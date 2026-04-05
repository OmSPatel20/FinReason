"""Step 4: Format FinQA into chat-template training data for SFT."""
import os, sys, json, random
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import extract_qa, extract_context, build_training_text

random.seed(42)
dataset = load_dataset("ibm-research/finqa", trust_remote_code=True)
train_raw = dataset["train"]

formatted = []
for i in range(len(train_raw)):
    q, a = extract_qa(train_raw[i])
    ctx = extract_context(train_raw[i])
    if not a.strip():
        continue
    formatted.append({"text": build_training_text(ctx, q, a), "question": q, "answer": a})

random.shuffle(formatted)
split = int(0.9 * len(formatted))
train_data = formatted[:split]
val_data = formatted[split:]

os.makedirs("data", exist_ok=True)
with open("data/train_sft.json", "w") as f:
    json.dump(train_data, f, indent=2)
with open("data/val_sft.json", "w") as f:
    json.dump(val_data, f, indent=2)

print(f"Train: {len(train_data)} | Val: {len(val_data)}")
print(f"\nSample prompt (first 200 chars):\n{train_data[0]['text'][:200]}...")
print("\n✓ Data saved to data/train_sft.json and data/val_sft.json")
