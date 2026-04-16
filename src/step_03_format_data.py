"""Step 3: Format FinQA into chat-template data for SFT WITH think tags."""
import json, os, sys, random
from datasets import load_dataset
sys.path.insert(0, os.path.dirname(__file__))
from shared_utils import extract_qa, extract_context, format_prompt

random.seed(42)
dataset = load_dataset("wandb/finqa-data-processed")
train_raw = dataset["train"]

formatted = []
for i in range(len(train_raw)):
    ex = train_raw[i]
    q, a = extract_qa(ex)
    ctx = extract_context(ex)
    if not a.strip():
        continue

    # Build <think> trace from the program column
    program = ex.get("program", "")
    if program and str(program).strip():
        think_answer = f"<think>{program}</think>\n{a}"
    else:
        think_answer = a

    formatted.append({
        "text": format_prompt(ctx, q, answer=think_answer, mode="train"),
        "question": q,
        "answer": a,
    })

random.shuffle(formatted)
split = int(0.9 * len(formatted))
os.makedirs("data", exist_ok=True)
json.dump(formatted[:split], open("data/train_sft.json", "w"))
json.dump(formatted[split:], open("data/val_sft.json", "w"))
print(f"Train: {split} | Val: {len(formatted) - split}")
print(f"\nSample with <think>:")
print(f"  {formatted[0]['text'][:400]}...")
print("✓ Saved to data/")