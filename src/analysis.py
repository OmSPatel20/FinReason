"""
Step 8: Generate all figures and analysis for the final report.
"""
import json, os, re
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use("Agg")
import numpy as np

OUT = "outputs"

def load(f):
    p = os.path.join(OUT, f)
    return json.load(open(p)) if os.path.exists(p) else None

zs = load("zeroshot_results.json")
sft = load("sft_results.json")
grpo = load("grpo_results.json")
grpo_log = load("grpo_training_log.json")

# --- Fig 1: Accuracy bar chart ---
fig, ax = plt.subplots(figsize=(7, 4.5))
names, accs, colors = [], [], []
for n, d, c in [("Zero-Shot", zs, "#94a3b8"),
                ("SFT", sft, "#3b82f6"),
                ("SFT + GRPO", grpo, "#10b981")]:
    if d:
        names.append(n); accs.append(d["accuracy"]*100); colors.append(c)

bars = ax.bar(names, accs, color=colors, edgecolor="black", linewidth=0.5, width=0.5)
for b, a in zip(bars, accs):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
            f"{a:.1f}%", ha="center", fontweight="bold", fontsize=11)
ax.set_ylabel("Execution Accuracy (%)"); ax.set_title("FinQA: Three Training Stages")
ax.set_ylim(0, max(accs)*1.25 if accs else 100); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(f"{OUT}/fig1_accuracy.png", dpi=200)
print("✓ fig1_accuracy.png")

# --- Fig 2: GRPO training curves ---
if grpo_log:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    steps = [e.get("step", i) for i, e in enumerate(grpo_log)]
    losses = [(s, e["loss"]) for s, e in zip(steps, grpo_log) if "loss" in e]
    rewards = [(s, e.get("reward", e.get("mean_reward", e.get("reward/mean"))))
               for s, e in zip(steps, grpo_log)]
    rewards = [(s, r) for s, r in rewards if r is not None]

    if losses:
        a1.plot(*zip(*losses), color="#ef4444", lw=1.5)
        a1.set_xlabel("Step"); a1.set_ylabel("Loss"); a1.set_title("GRPO Loss")
        a1.grid(alpha=0.3)
    if rewards:
        a2.plot(*zip(*rewards), color="#10b981", lw=1.5)
        a2.set_xlabel("Step"); a2.set_ylabel("Reward"); a2.set_title("Mean Reward")
        a2.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{OUT}/fig2_grpo_curves.png", dpi=200)
    print("✓ fig2_grpo_curves.png")

# --- Fig 3: SFT vs GRPO per-example ---
if sft and grpo:
    sm = {r["i"]: r["ok"] for r in sft["results"]}
    gm = {r["i"]: r["ok"] for r in grpo["results"]}
    common = set(sm) & set(gm)
    cats = {"Both ✓": 0, "Both ✗": 0, "SFT✓ GRPO✗": 0, "SFT✗ GRPO✓": 0}
    for idx in common:
        s, g = sm[idx], gm[idx]
        if s and g: cats["Both ✓"] += 1
        elif not s and not g: cats["Both ✗"] += 1
        elif s: cats["SFT✓ GRPO✗"] += 1
        else: cats["SFT✗ GRPO✓"] += 1

    fig, ax = plt.subplots(figsize=(7, 4))
    cc = ["#10b981", "#ef4444", "#f59e0b", "#3b82f6"]
    ax.bar(cats.keys(), cats.values(), color=cc, edgecolor="black", lw=0.5)
    for i, v in enumerate(cats.values()):
        ax.text(i, v+0.5, str(v), ha="center", fontweight="bold")
    ax.set_ylabel("Count"); ax.set_title("Per-Example: SFT vs GRPO")
    plt.xticks(fontsize=9); plt.tight_layout()
    plt.savefig(f"{OUT}/fig3_error_analysis.png", dpi=200)
    print("✓ fig3_error_analysis.png")

# --- Fig 4: Think vs no-think ---
if grpo:
    wt = [r for r in grpo["results"] if r.get("think")]
    nt = [r for r in grpo["results"] if not r.get("think")]
    at = sum(r["ok"] for r in wt)/max(len(wt),1)*100
    an = sum(r["ok"] for r in nt)/max(len(nt),1)*100

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["With <think>", "Without"], [at, an],
           color=["#8b5cf6","#94a3b8"], edgecolor="black", lw=0.5)
    ax.text(0, at+0.5, f"{at:.1f}%\n(n={len(wt)})", ha="center", fontsize=9)
    ax.text(1, an+0.5, f"{an:.1f}%\n(n={len(nt)})", ha="center", fontsize=9)
    ax.set_ylabel("Accuracy (%)"); ax.set_title("Does Reasoning Help?")
    ax.grid(axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig(f"{OUT}/fig4_think_analysis.png", dpi=200)
    print("✓ fig4_think_analysis.png")

# --- Qualitative examples ---
if grpo:
    print(f"\n{'='*55}")
    print("  QUALITATIVE EXAMPLES (with <think>)")
    print(f"{'='*55}")
    exs = [r for r in grpo["results"] if r.get("think")][:3]
    for e in exs:
        print(f"\n  Q: {e['q']}")
        print(f"  GT: {e['gt']}")
        print(f"  Output: {e['raw'][:250]}...")
        print(f"  Extracted: {e['pred']}")
        print(f"  {'✓' if e['ok'] else '✗'}")

print(f"\n✓ All figures saved to {OUT}/")
