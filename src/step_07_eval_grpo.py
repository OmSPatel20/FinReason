"""Step 7: Evaluate GRPO model — Checkpoint 3 (FINAL)."""
import torch, json, os, sys, re
from tqdm import tqdm
from datasets import load_dataset
sys.path.insert(0, os.path.dirname(__file__))
from shared_utils import extract_qa, extract_context, extract_final_answer, check_answer, compute_execution_accuracy, format_prompt

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER = "checkpoints/grpo/final_adapter"
NUM_TEST = 300
os.makedirs("outputs",exist_ok=True)
try:
    from unsloth import FastLanguageModel
    model,tokenizer = FastLanguageModel.from_pretrained(model_name=ADAPTER,max_seq_length=1280,load_in_4bit=True)
    FastLanguageModel.for_inference(model)
except ImportError:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True),
        device_map="auto")
    model = PeftModel.from_pretrained(base,ADAPTER); model.eval()

dataset = load_dataset("wandb/finqa-data-processed")
test = dataset["test"] if "test" in dataset else dataset["validation"]
if NUM_TEST and NUM_TEST < len(test): test = test.select(range(NUM_TEST))
preds, gts, log = [], [], []
think_count = 0
for i in tqdm(range(len(test)),desc="GRPO eval"):
    q,gt = extract_qa(test[i]); ctx = extract_context(test[i])
    ids = tokenizer(format_prompt(ctx,q,mode="grpo_eval"),return_tensors="pt",truncation=True,max_length=1024).to(model.device)
    with torch.no_grad():
        out = model.generate(**ids,max_new_tokens=256,do_sample=False,pad_token_id=tokenizer.eos_token_id)
    pred = tokenizer.decode(out[0][ids["input_ids"].shape[1]:],skip_special_tokens=True).strip()
    has_think = bool(re.search(r'<think>.*?</think>',pred,re.DOTALL))
    if has_think: think_count += 1
    final = extract_final_answer(pred); ok = check_answer(final,gt)
    preds.append(pred); gts.append(gt)
    log.append({"i":i,"q":q,"gt":gt,"raw":pred[:300],"pred":final,"think":has_think,"ok":ok})
acc = compute_execution_accuracy(preds,gts)
print(f"\n{'='*55}\n  GRPO: {acc:.4f} ({acc*100:.2f}%) | Think: {think_count}/{len(test)}\n{'='*55}")
all_acc = {"grpo":acc}
for n,p in [("zeroshot","outputs/zeroshot_results.json"),("sft","outputs/sft_results.json")]:
    if os.path.exists(p): all_acc[n] = json.load(open(p))["accuracy"]
for n in ["zeroshot","sft","grpo"]:
    if n in all_acc: print(f"  {n:>10s}: {all_acc[n]*100:.2f}%")
json.dump({"model":MODEL_NAME,"adapter":ADAPTER,"accuracy":acc,"think_rate":think_count/len(test),
    "n":len(preds),"results":log},open("outputs/grpo_results.json","w"),indent=2)
