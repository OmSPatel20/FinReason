"""
run_scale.py — Train a specific model scale end-to-end.

Usage:
    python src/run_scale.py 1.5B
    python src/run_scale.py 3B
    python src/run_scale.py 7B

Runs: zero-shot → format data → SFT → eval SFT → GRPO → eval GRPO → figures
All results saved to outputs/<SCALE>/ and checkpoints/<SCALE>/
"""
import subprocess, sys, os, json, shutil

SCALES = {
    "1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
    "3B":   "Qwen/Qwen2.5-3B-Instruct",
    "7B":   "Qwen/Qwen2.5-7B-Instruct",
}

if len(sys.argv) < 2 or sys.argv[1] not in SCALES:
    print(f"Usage: python src/run_scale.py <{'|'.join(SCALES.keys())}>")
    sys.exit(1)

scale = sys.argv[1]
model_name = SCALES[scale]

print(f"\n{'='*55}")
print(f"  FinReason — Running {scale} ({model_name})")
print(f"{'='*55}\n")

# Create output dirs
os.makedirs(f"outputs/{scale}", exist_ok=True)
os.makedirs(f"results/{scale}", exist_ok=True)
os.makedirs(f"checkpoints/{scale}/sft", exist_ok=True)
os.makedirs(f"checkpoints/{scale}/grpo", exist_ok=True)

# Patch all scripts to use this model
scripts = [
    "src/step_02_zeroshot_baseline.py",
    "src/step_03_format_data.py",
    "src/sft_train.py",
    "src/step_05_eval_sft.py",
    "src/grpo_train.py",
    "src/step_07_eval_grpo.py",
]

# Also patch output/checkpoint paths
for script in scripts:
    if not os.path.exists(script):
        continue
    with open(script, 'r') as f:
        content = f.read()

    # Backup original
    backup = script + ".bak"
    if not os.path.exists(backup):
        with open(backup, 'w') as f:
            f.write(content)

    # Replace model name
    for old_model in SCALES.values():
        content = content.replace(f'"{old_model}"', f'"{model_name}"')

    # Replace output paths
    content = content.replace('"outputs/zeroshot_results.json"', f'"outputs/{scale}/zeroshot_results.json"')
    content = content.replace('"outputs/sft_results.json"', f'"outputs/{scale}/sft_results.json"')
    content = content.replace('"outputs/grpo_results.json"', f'"outputs/{scale}/grpo_results.json"')
    content = content.replace('"outputs/grpo_training_log.json"', f'"outputs/{scale}/grpo_training_log.json"')

    # Replace checkpoint paths
    content = content.replace('"checkpoints/sft"', f'"checkpoints/{scale}/sft"')
    content = content.replace('"checkpoints/grpo"', f'"checkpoints/{scale}/grpo"')
    content = content.replace('"checkpoints/sft/final_adapter"', f'"checkpoints/{scale}/sft/final_adapter"')
    content = content.replace('"checkpoints/grpo/final_adapter"', f'"checkpoints/{scale}/grpo/final_adapter"')

    with open(script, 'w') as f:
        f.write(content)

print(f"  Patched {len(scripts)} scripts for {scale}\n")

# Run pipeline
steps = [
    ("Zero-shot baseline", "python src/step_02_zeroshot_baseline.py"),
    ("Format SFT data", "python src/step_03_format_data.py"),
    ("SFT training", "python src/sft_train.py"),
    ("Eval SFT", "python src/step_05_eval_sft.py"),
    ("GRPO training", "python src/grpo_train.py"),
    ("Eval GRPO", "python src/step_07_eval_grpo.py"),
]

for name, cmd in steps:
    print(f"\n{'='*55}")
    print(f"  [{scale}] {name}")
    print(f"{'='*55}\n")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\n  ✗ {name} failed! Stopping.")
        sys.exit(1)

# Generate figures for this scale
print(f"\n  Generating figures for {scale}...")
# Quick figure gen with correct paths
fig_script = f"""
import json, os
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use("Agg")

OUT = "outputs/{scale}"
RES = "results/{scale}"
os.makedirs(RES, exist_ok=True)

def load(f):
    p = os.path.join(OUT, f)
    return json.load(open(p)) if os.path.exists(p) else None

zs = load("zeroshot_results.json")
sft = load("sft_results.json")
grpo = load("grpo_results.json")
grpo_log = load("grpo_training_log.json")

# Fig 1: Accuracy
fig, ax = plt.subplots(figsize=(7,4.5))
names, accs, colors = [], [], []
for n, d, c in [("Zero-Shot",zs,"#94a3b8"),("SFT",sft,"#3b82f6"),("SFT+GRPO",grpo,"#10b981")]:
    if d: names.append(n); accs.append(d["accuracy"]*100); colors.append(c)
if accs:
    bars = ax.bar(names, accs, color=colors, edgecolor="black", lw=0.5, width=0.5)
    for b, a in zip(bars, accs):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3, f"{{a:.1f}}%", ha="center", fontweight="bold")
    ax.set_ylabel("Execution Accuracy (%)"); ax.set_title(f"FinQA — {scale} Model")
    ax.set_ylim(0, max(accs)*1.3); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{{RES}}/fig1_accuracy.png", dpi=200)

# Fig 2: GRPO curves
if grpo_log:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    losses = [(e.get("step",i), e["loss"]) for i, e in enumerate(grpo_log) if "loss" in e]
    rewards = [(e.get("step",i), e.get("reward", e.get("mean_reward", e.get("reward/mean"))))
               for i, e in enumerate(grpo_log)]
    rewards = [(s, r) for s, r in rewards if r is not None]
    if losses:
        a1.plot(*zip(*losses), color="#ef4444", lw=1.5)
        a1.set_xlabel("Step"); a1.set_ylabel("Loss"); a1.set_title(f"GRPO Loss ({scale})"); a1.grid(alpha=0.3)
    if rewards:
        a2.plot(*zip(*rewards), color="#10b981", lw=1.5)
        a2.set_xlabel("Step"); a2.set_ylabel("Reward"); a2.set_title(f"Mean Reward ({scale})"); a2.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{{RES}}/fig2_grpo_curves.png", dpi=200)

print(f"✓ Figures saved to {{RES}}/")
"""

with open("/tmp/gen_figs.py", "w") as f:
    f.write(fig_script)
subprocess.run("python /tmp/gen_figs.py", shell=True)

# Restore original scripts
for script in scripts:
    backup = script + ".bak"
    if os.path.exists(backup):
        shutil.move(backup, script)

print(f"\n{'='*55}")
print(f"  ✓ {scale} COMPLETE")
print(f"  Results:     outputs/{scale}/")
print(f"  Figures:     results/{scale}/")
print(f"  Checkpoints: checkpoints/{scale}/")
print(f"{'='*55}")
