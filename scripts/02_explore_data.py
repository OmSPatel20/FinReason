"""Step 2: Explore FinQA — understand answer types, lengths, formats."""
import os, sys, re, random, json
from collections import Counter
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import extract_qa, extract_context

random.seed(42)
dataset = load_dataset("ibm-research/finqa", trust_remote_code=True)
train = dataset["train"]

print(f"Columns: {train.column_names}")

# Show 5 random examples
print("\n" + "=" * 60)
print("5 RANDOM EXAMPLES")
print("=" * 60)
for i, idx in enumerate(random.sample(range(len(train)), 5)):
    q, a = extract_qa(train[idx])
    ctx = extract_context(train[idx])
    print(f"\nEx {i+1} (idx {idx}): Q: {q}")
    print(f"  A: {a}  |  Context: {len(ctx)} chars")

# Answer type analysis
print("\n" + "=" * 60)
print("ANSWER TYPE ANALYSIS")
print("=" * 60)
types = Counter()
ctx_lens = []
for ex in train:
    _, a = extract_qa(ex)
    ctx_lens.append(len(extract_context(ex)))
    cleaned = re.sub(r'[,$%]', '', a.strip())
    try:
        float(cleaned); types["numeric"] += 1
    except ValueError:
        types["yes/no" if a.strip().lower() in ("yes","no") else "text"] += 1

for t, c in types.most_common():
    print(f"  {t}: {c} ({100*c/len(train):.1f}%)")
print(f"\nContext length: mean={sum(ctx_lens)/len(ctx_lens):.0f}, max={max(ctx_lens)}")

os.makedirs("data", exist_ok=True)
with open("data/format_info.json", "w") as f:
    json.dump({"columns": train.column_names, "types": dict(types),
               "n_train": len(train)}, f, indent=2)
print("✓ Saved data/format_info.json")
