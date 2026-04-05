"""Step 3: Zero-shot baseline on FinQA (Checkpoint 1 of 3).
THIS IS YOUR GO/NO-GO GATE. If accuracy < 2%, switch models.
"""
import torch, json, os, sys
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import extract_qa, extract_context, build_prompt
from reward_utils import extract_final_answer, check_answer, compute_execution_accuracy

# ---- CONFIG ----
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
NUM_TEST = 300
MAX_NEW_TOKENS = 128
OUTPUT_DIR = "outputs"
# ----------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True),
    device_map="auto", trust_remote_code=True)
model.eval()
print(f"VRAM: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")

dataset = load_dataset("ibm-research/finqa", trust_remote_code=True)
test_data = dataset.get("test", dataset.get("validation"))
if NUM_TEST and NUM_TEST < len(test_data):
    test_data = test_data.select(range(NUM_TEST))
print(f"Evaluating {len(test_data)} examples...\n")

predictions, ground_truths, results_log = [], [], []

for i in tqdm(range(len(test_data)), desc="Zero-shot"):
    q, gt = extract_qa(test_data[i])
    ctx = extract_context(test_data[i])
    prompt = build_prompt(ctx, q)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=1024).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    pred = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()
    final = extract_final_answer(pred)
    correct = check_answer(final, gt)

    predictions.append(pred)
    ground_truths.append(gt)
    results_log.append({"i": i, "q": q, "gt": gt, "pred": final, "ok": correct})

    if i < 5:
        print(f"  {'✓' if correct else '✗'} GT={gt} Pred={final}")

acc = compute_execution_accuracy(predictions, ground_truths)
print(f"\n{'='*50}")
print(f"ZERO-SHOT ACCURACY: {acc:.4f} ({acc*100:.2f}%)")
print(f"{'='*50}")

with open(f"{OUTPUT_DIR}/zeroshot_results.json", "w") as f:
    json.dump({"model": MODEL_NAME, "accuracy": acc, "n": len(predictions),
               "results": results_log}, f, indent=2)

if acc < 0.02:
    print("  TOO LOW. Switch to Qwen2.5-3B or Phi-4-mini.")
elif acc < 0.10:
    print("  Low but workable. SFT should help. Proceed.")
else:
    print("  Solid baseline. Proceed as planned.")
