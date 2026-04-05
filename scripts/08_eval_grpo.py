"""Step 8: Evaluate GRPO model (Checkpoint 3 of 3 — final model)."""
import torch, json, os, sys, re
from tqdm import tqdm
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import extract_qa, extract_context, build_prompt
from reward_utils import extract_final_answer, check_answer, compute_execution_accuracy

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER = "checkpoints/grpo/final_adapter"
NUM_TEST = 300
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model
try:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=ADAPTER, max_seq_length=1280, load_in_4bit=True)
    FastLanguageModel.for_inference(model)
except ImportError:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True),
        device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(base, ADAPTER); model.eval()

dataset = load_dataset("ibm-research/finqa", trust_remote_code=True)
test_data = dataset.get("test", dataset.get("validation"))
if NUM_TEST and NUM_TEST < len(test_data):
    test_data = test_data.select(range(NUM_TEST))

predictions, ground_truths, results_log = [], [], []
think_count = 0

for i in tqdm(range(len(test_data)), desc="GRPO eval"):
    q, gt = extract_qa(test_data[i])
    ctx = extract_context(test_data[i])
    prompt = build_prompt(ctx, q)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=1024).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    pred = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()

    has_think = bool(re.search(r'<think>.*?</think>', pred, re.DOTALL))
    if has_think:
        think_count += 1
    final = extract_final_answer(pred)
    correct = check_answer(final, gt)

    predictions.append(pred); ground_truths.append(gt)
    results_log.append({"i": i, "q": q, "gt": gt, "raw": pred[:300],
                        "pred": final, "think": has_think, "ok": correct})

acc = compute_execution_accuracy(predictions, ground_truths)

print(f"\n{'='*60}")
print(f"GRPO ACCURACY: {acc:.4f} ({acc*100:.2f}%)")
print(f"Think rate: {think_count}/{len(test_data)} ({100*think_count/len(test_data):.1f}%)")
print(f"{'='*60}")

# Compare all 3
print(f"\n{'='*60}")
print("FULL COMPARISON")
print(f"{'='*60}")
all_results = {"grpo": acc}
for name, path in [("zeroshot","zeroshot_results.json"),("sft","sft_results.json")]:
    p = f"{OUTPUT_DIR}/{path}"
    if os.path.exists(p):
        all_results[name] = json.load(open(p))["accuracy"]
for name in ["zeroshot", "sft", "grpo"]:
    if name in all_results:
        a = all_results[name]
        print(f"  {name:>10s}: {a:.4f} ({a*100:.2f}%) {'█' * int(a * 60)}")

with open(f"{OUTPUT_DIR}/grpo_results.json", "w") as f:
    json.dump({"model": MODEL_NAME, "adapter": ADAPTER, "accuracy": acc,
               "think_rate": think_count/len(test_data),
               "n": len(predictions), "results": results_log}, f, indent=2)
print(f"\n✓ Results saved to {OUTPUT_DIR}/grpo_results.json")
