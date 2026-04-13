"""Step 6: Stage 2 — GRPO with Verifiable Rewards. ~2-4 hours on RTX 4060."""
import torch, json, os, sys
from datasets import load_dataset, Dataset
sys.path.insert(0, os.path.dirname(__file__))
from shared_utils import extract_qa, extract_context, format_prompt, reward_function, SYSTEM_MSG_TRAIN

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
SFT_ADAPTER = "checkpoints/sft/final_adapter"
OUTPUT_DIR = "checkpoints/grpo"
NUM_GEN, MAX_TOK, MAX_PROMPT, LR = 4, 256, 768, 5e-6
MAX_TRAIN, TEMP = 2000, 0.7
os.makedirs(OUTPUT_DIR,exist_ok=True); os.makedirs("outputs",exist_ok=True)

print(f"Loading SFT model from {SFT_ADAPTER}...")
try:
    from unsloth import FastLanguageModel
    model,tokenizer = FastLanguageModel.from_pretrained(model_name=SFT_ADAPTER,
        max_seq_length=MAX_PROMPT+MAX_TOK,load_in_4bit=True)
    print("Loaded with Unsloth ✓")
except ImportError:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True),
        device_map="auto")
    model = PeftModel.from_pretrained(base,SFT_ADAPTER,is_trainable=True)
    print("Loaded with PEFT ✓")

dataset = load_dataset("wandb/finqa-data-processed")
train_raw = dataset["train"].shuffle(seed=42)
prompts, answers = [], []
for i in range(min(MAX_TRAIN,len(train_raw))):
    q,a = extract_qa(train_raw[i]); ctx = extract_context(train_raw[i])
    if not a.strip(): continue
    prompts.append(format_prompt(ctx,q,answer=None,max_context_chars=1000,mode="train")); answers.append(a)
print(f"Prepared {len(prompts)} prompts")

def finqa_reward_func(completions, **kwargs):
    gts = kwargs.get("ground_truth",[""]* len(completions))
    rewards = []
    for comp,gt in zip(completions,gts):
        if isinstance(comp,list):
            text = ""
            for msg in comp:
                if isinstance(msg,dict) and msg.get("role")=="assistant": text=msg.get("content","")
            comp = text
        elif isinstance(comp,dict): comp = comp.get("content",str(comp))
        rewards.append(reward_function(str(comp),str(gt)))
    return rewards

grpo_ds = Dataset.from_dict({"prompt":prompts,"ground_truth":answers})

from trl import GRPOConfig, GRPOTrainer
from training_logger import LiveLogCallback

trainer = GRPOTrainer(model=model,processing_class=tokenizer,
    args=GRPOConfig(output_dir=OUTPUT_DIR,num_train_epochs=1,per_device_train_batch_size=1,
        gradient_accumulation_steps=4,learning_rate=LR,lr_scheduler_type="cosine",
        num_generations=NUM_GEN,max_completion_length=MAX_TOK,max_prompt_length=MAX_PROMPT,
        temperature=TEMP,logging_steps=10,save_steps=100,save_total_limit=2,
        bf16=torch.cuda.is_bf16_supported(),fp16=not torch.cuda.is_bf16_supported(),
        report_to="none",remove_unused_columns=False),
    train_dataset=grpo_ds,reward_funcs=[finqa_reward_func],
    callbacks=[LiveLogCallback("grpo")])

print(f"\n{'='*55}\n  GRPO TRAINING — {len(prompts)} prompts × G={NUM_GEN}\n{'='*55}\n")
trainer.train()
model.save_pretrained(os.path.join(OUTPUT_DIR,"final_adapter"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR,"final_adapter"))
logs = [l for l in trainer.state.log_history if any(k in l for k in ["loss","reward","mean_reward"])]
json.dump(logs,open("outputs/grpo_training_log.json","w"),indent=2)
print(f"\n✓ GRPO complete → {OUTPUT_DIR}/final_adapter")
