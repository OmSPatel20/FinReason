"""
Step 2: Zero-shot baseline — Checkpoint 1 of 3.

GO/NO-GO GATE: If accuracy < 2%, switch to a bigger model.
"""
import torch, json, os, sys
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(__file__))
from shared_utils import (extract_qa, extract_context, extract_final_answer,
                           check_answer, compute_execution_accuracy, format_prompt)

# ============================================================
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
NUM_TEST = 300
MAX_NEW_TOKENS = 128
# ============================================================

os.makedirs("outputs", exist_ok=True)

print(f"Loading {MODEL_NAME} in 4-bit...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True),
    device_map="auto", trust_remote_code=True,
)
model.eval()
print(f"VRAM: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")

dataset = load_dataset("ibm-research/finqa", trust_remote_code=True)
test = dataset["test"] if "test" in dataset else dataset["validation"]
if NUM_TEST and NUM_TEST < len(test):
    test = test.select(range(NUM_TEST))
print(f"Evaluating {len(test)} examples zero-shot...\n")

predictions, ground_truths, log = [], [], []

for i in tqdm(range(len(test)), desc="Zero-shot"):
    q, gt = extract_qa(test[i])
    ctx = extract_context(test[i])
    prompt = format_prompt(ctx, q, answer=None)

    ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                    max_length=1024).to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    pred = tokenizer.decode(out[0][ids["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()

    final = extract_final_answer(pred)
    correct = check_answer(final, gt)
    predictions.append(pred)
    ground_truths.append(gt)
    log.append({"i": i, "q": q, "gt": gt, "pred": final, "ok": correct})

    if i < 5:
        print(f"  {'✓' if correct else '✗'} GT={gt} | Pred={final}")

acc = compute_execution_accuracy(predictions, ground_truths)
print(f"\n{'='*50}")
print(f"  ZERO-SHOT ACCURACY: {acc:.4f} ({acc*100:.2f}%)")
print(f"{'='*50}")

with open("outputs/zeroshot_results.json", "w") as f:
    json.dump({"model": MODEL_NAME, "accuracy": acc,
               "n": len(predictions), "results": log}, f, indent=2)

if acc < 0.02:
    print("\n  ✗ TOO LOW. Switch to Qwen2.5-3B or Phi-4-mini.")
elif acc < 0.10:
    print("\n  ~ Low but workable. SFT should help. Proceed.")
else:
    print("\n  ✓ Solid baseline. Proceed as planned.")
