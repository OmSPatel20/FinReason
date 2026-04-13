"""Step 1: Download FinQA and explore its structure."""
from datasets import load_dataset
from collections import Counter
import random, re, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from shared_utils import extract_qa, extract_context

random.seed(42)
print("Downloading FinQA...")
dataset = load_dataset("wandb/finqa-data-processed")
for s in dataset: print(f"  {s}: {len(dataset[s])}")
train = dataset["train"]
print(f"Columns: {train.column_names}\n")
for i, idx in enumerate(random.sample(range(len(train)), 3)):
    q, a = extract_qa(train[idx])
    print(f"  Example {i+1}: Q={q[:60]}... A={a}")
types = Counter()
for ex in train:
    _, a = extract_qa(ex)
    try: float(re.sub(r'[,$%]','',a.strip())); types["numeric"]+=1
    except ValueError: types["yes/no" if a.strip().lower() in ("yes","no") else "text"]+=1
print(f"\nAnswer types:")
for t,c in types.most_common(): print(f"  {t}: {c} ({100*c/len(train):.1f}%)")
os.makedirs("data", exist_ok=True)
json.dump({"columns":train.column_names,"types":dict(types)},open("data/format_info.json","w"),indent=2)
print("✓ Done")
