"""Step 3: Format FinQA into chat-template data for SFT."""
import json, os, sys, random
from datasets import load_dataset
sys.path.insert(0, os.path.dirname(__file__))
from shared_utils import extract_qa, extract_context, format_prompt

random.seed(42)
dataset = load_dataset("wandb/finqa-data-processed")
train_raw = dataset["train"]
formatted = []
for i in range(len(train_raw)):
    q,a = extract_qa(train_raw[i]); ctx = extract_context(train_raw[i])
    if not a.strip(): continue
    formatted.append({"text":format_prompt(ctx,q,answer=a),"question":q,"answer":a})
random.shuffle(formatted)
split = int(0.9*len(formatted))
os.makedirs("data",exist_ok=True)
json.dump(formatted[:split],open("data/train_sft.json","w"))
json.dump(formatted[split:],open("data/val_sft.json","w"))
print(f"Train: {split} | Val: {len(formatted)-split}")
print(f"Sample: {formatted[0]['text'][:200]}...")
print("✓ Saved to data/")
