"""Step 1: Download FinQA dataset from HuggingFace."""
from datasets import load_dataset

print("Downloading FinQA (this takes ~30 seconds)...")
dataset = load_dataset("ibm-research/finqa", trust_remote_code=True)

print(f"\nSplits:")
for split in dataset:
    print(f"  {split}: {len(dataset[split])} examples")
print(f"Columns: {dataset['train'].column_names}")

ex = dataset["train"][0]
print(f"\nFirst example keys: {list(ex.keys())}")
for k, v in ex.items():
    print(f"  {k}: {str(v)[:120]}...")

print("\n✓ Dataset downloaded and cached.")
