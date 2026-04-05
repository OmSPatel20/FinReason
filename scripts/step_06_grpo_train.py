"""
Step 6: Stage 2 — GRPO Training with Verifiable Rewards.

The heart of the project. Uses TRL's GRPOTrainer.
Time: ~2-4 hours on RTX 4060.
"""
import torch, json, os, sys
from datasets import load_dataset, Dataset

sys.path.insert(0, os.path.dirname(__file__))
from shared_utils import (extract_qa, extract_context, format_prompt,
                           reward_function, SYSTEM_MSG)

# ============================================================
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
SFT_ADAPTER = "checkpoints/sft/final_adapter"
OUTPUT_DIR = "checkpoints/grpo"
NUM_GENERATIONS = 4
MAX_NEW_TOKENS = 256
MAX_PROMPT_LEN = 768
GRPO_EPOCHS = 1
LR = 5e-6
BATCH = 1
GRAD_ACCUM = 4
MAX_TRAIN = 2000
TEMPERATURE = 0.7
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load SFT model ---
print(f"Loading SFT model from {SFT_ADAPTER}...")
try:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=SFT_ADAPTER,
        max_seq_length=MAX_PROMPT_LEN + MAX_NEW_TOKENS,
        load_in_4bit=True,
    )
    print("Loaded with Unsloth ✓")
except ImportError:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True),
        device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(base, SFT_ADAPTER, is_trainable=True)
    print("Loaded with PEFT ✓")

print(f"VRAM: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")

# --- Build prompts ---
print("Preparing training prompts...")
dataset = load_dataset("ibm-research/finqa", trust_remote_code=True)
train_raw = dataset["train"].shuffle(seed=42)

prompts, answers = [], []
for i in range(min(MAX_TRAIN, len(train_raw))):
    q, a = extract_qa(train_raw[i])
    ctx = extract_context(train_raw[i])
    if not a.strip():
        continue
    prompts.append(format_prompt(ctx, q, answer=None, max_context_chars=1000))
    answers.append(a)
print(f"Prepared {len(prompts)} prompts.")

# --- Reward function wrapper for TRL ---
def finqa_reward_func(completions, **kwargs):
    gts = kwargs.get("ground_truth", [""] * len(completions))
    rewards = []
    for comp, gt in zip(completions, gts):
        if isinstance(comp, list):
            text = ""
            for msg in comp:
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    text = msg.get("content", "")
            comp = text
        elif isinstance(comp, dict):
            comp = comp.get("content", str(comp))
        rewards.append(reward_function(str(comp), str(gt)))
    return rewards

# --- Dataset ---
grpo_ds = Dataset.from_dict({"prompt": prompts, "ground_truth": answers})

# --- Train ---
from trl import GRPOConfig, GRPOTrainer

config = GRPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=GRPO_EPOCHS,
    per_device_train_batch_size=BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    num_generations=NUM_GENERATIONS,
    max_completion_length=MAX_NEW_TOKENS,
    max_prompt_length=MAX_PROMPT_LEN,
    temperature=TEMPERATURE,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    report_to="none",
    remove_unused_columns=False,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    args=config,
    train_dataset=grpo_ds,
    reward_funcs=[finqa_reward_func],
)

print(f"\n{'='*55}")
print(f"  STARTING GRPO — {len(prompts)} prompts × G={NUM_GENERATIONS}")
print(f"  Estimated: ~2-4 hours on RTX 4060")
print(f"{'='*55}\n")

trainer.train()

model.save_pretrained(os.path.join(OUTPUT_DIR, "final_adapter"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_adapter"))

logs = [l for l in trainer.state.log_history
        if any(k in l for k in ["loss", "reward", "mean_reward"])]
with open("outputs/grpo_training_log.json", "w") as f:
    json.dump(logs, f, indent=2)

print(f"\n✓ GRPO complete. Adapter → {OUTPUT_DIR}/final_adapter")
