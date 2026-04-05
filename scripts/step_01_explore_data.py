"""
Step 1: Download FinQA and explore its structure.
"""
from datasets import load_dataset
from collections import Counter
import random, re, json, os, sys

sys.path.insert(0, os.path.dirname(__file__))
from shared_utils import extract_qa, extract_context

random.seed(42)

print("Downloading FinQA from HuggingFace...")
dataset = load_dataset("ibm-research/finqa", trust_remote_code=True)

print(f"\nSplits:")
for s in dataset:
    print(f"  {s}: {len(dataset[s])} examples")
print(f"Columns: {dataset['train'].column_names}")

# --- 5 random examples ---
train = dataset["train"]
indices = random.sample(range(len(train)), 5)
print(f"\n{'='*60}\n  5 RANDOM EXAMPLES\n{'='*60}")
for i, idx in enumerate(indices):
    ex = train[idx]
    q, a = extract_qa(ex)
    ctx = extract_context(ex)
    print(f"\n  Example {i+1} (idx {idx}):")
    print(f"    Q: {q}")
    print(f"    A: {a}")
    print(f"    Context: {len(ctx)} chars")
    if "qa" in ex and isinstance(ex["qa"], dict):
        prog = ex["qa"].get("program", "N/A")
        print(f"    Program: {prog}")

# --- Answer analysis ---
print(f"\n{'='*60}\n  ANSWER ANALYSIS\n{'='*60}")
types = Counter()
ctx_lens = []
for ex in train:
    _, a = extract_qa(ex)
    ctx_lens.append(len(extract_context(ex)))
    cleaned = re.sub(r'[,$%]', '', a.strip())
    try:
        float(cleaned)
        types["numeric"] += 1
    except ValueError:
        if a.strip().lower() in ("yes", "no"):
            types["yes/no"] += 1
        else:
            types["text"] += 1

print(f"  Total: {len(train)}")
for t, c in types.most_common():
    print(f"    {t}: {c} ({100*c/len(train):.1f}%)")
print(f"  Context: mean={sum(ctx_lens)/len(ctx_lens):.0f} chars, max={max(ctx_lens)}")

os.makedirs("data", exist_ok=True)
with open("data/format_info.json", "w") as f:
    json.dump({"columns": dataset["train"].column_names,
               "answer_types": dict(types), "num_train": len(train)}, f, indent=2)

print(f"\n✓ Data exploration complete. Saved info to data/format_info.json")
