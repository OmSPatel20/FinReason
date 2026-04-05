# FinReason

**Teaching a Small Language Model Financial Numerical Reasoning via Reinforcement Learning with Verifiable Rewards**

Course: EGN 6217 — Applied Deep Learning, University of Florida, Spring 2026

---

## What This Is

A two-stage pipeline (SFT → GRPO) that teaches Qwen2.5-1.5B to answer
numerical questions over financial reports, where the RL reward is simply
"did you get the right number." Includes a Streamlit demo app.

---

## How to Run

### Option A: Run everything at once

```bash
# 1. Create environment
conda create -n finreason python=3.11 -y
conda activate finreason

# 2. Install dependencies
pip install -r requirements.txt
pip install unsloth          # optional but recommended (saves ~30% VRAM)

# 3. Run the full pipeline
chmod +x run.sh
./run.sh

# 4. Launch the demo
streamlit run scripts/app.py
```

### Option B: Run step by step (recommended first time)

```bash
# Setup
conda create -n finreason python=3.11 -y
conda activate finreason
pip install -r requirements.txt

# Step 0: Verify GPU works
python scripts/step_00_check_gpu.py

# Step 1: Download FinQA and explore it
python scripts/step_01_explore_data.py

# Self-test: Make sure reward function works
python scripts/shared_utils.py

# Step 2: Zero-shot baseline (GO/NO-GO gate)
#   If accuracy < 2%, switch model. See Troubleshooting.
python scripts/step_02_zeroshot_baseline.py

# Step 3: Format data for SFT
python scripts/step_03_format_data.py

# Step 4: SFT training (~1-2 hours on RTX 4060)
python scripts/step_04_sft_train.py

# Step 5: Evaluate SFT model
python scripts/step_05_eval_sft.py

# Step 6: GRPO training (~2-4 hours on RTX 4060)
python scripts/step_06_grpo_train.py

# Step 7: Evaluate GRPO model (final results!)
python scripts/step_07_eval_grpo.py

# Step 8: Generate all figures for the report
python scripts/step_08_analysis.py

# Launch interactive demo
streamlit run scripts/app.py
```

---

## Project Structure

```
finreason/
├── run.sh                          # One-click full pipeline
├── requirements.txt                # Python dependencies
├── README.md                       # You are here
│
├── scripts/
│   ├── shared_utils.py             # Shared: reward function, data extraction, metrics
│   ├── step_00_check_gpu.py        # Verify GPU + deps
│   ├── step_01_explore_data.py     # Download & explore FinQA
│   ├── step_02_zeroshot_baseline.py  # Checkpoint 1: base model
│   ├── step_03_format_data.py      # Format data for SFT
│   ├── step_04_sft_train.py        # Stage 1: SFT with QLoRA
│   ├── step_05_eval_sft.py         # Checkpoint 2: SFT model
│   ├── step_06_grpo_train.py       # Stage 2: GRPO with verifiable rewards
│   ├── step_07_eval_grpo.py        # Checkpoint 3: GRPO model (FINAL)
│   ├── step_08_analysis.py         # Generate figures for report
│   └── app.py                      # Streamlit demo app
│
├── data/                           # Created by step 01 and 03
├── checkpoints/sft/                # SFT adapter (created by step 04)
├── checkpoints/grpo/               # GRPO adapter (created by step 06)
└── outputs/                        # Results JSONs + figures
    ├── zeroshot_results.json
    ├── sft_results.json
    ├── grpo_results.json
    ├── grpo_training_log.json
    ├── fig1_accuracy.png
    ├── fig2_grpo_curves.png
    ├── fig3_error_analysis.png
    └── fig4_think_analysis.png
```

---

## Hardware Requirements

| Setup | What works |
|-------|-----------|
| **RTX 4060 (8 GB)** | Full pipeline. Batch=1, QLoRA, 4-bit quantization |
| **Colab Free (T4 16 GB)** | Faster. Can increase batch size to 2 |
| **Colab Pro (A100 40 GB)** | Best. Can use Qwen2.5-3B, batch=4, G=8 |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `step_00` fails: no CUDA | Install CUDA toolkit or use Colab |
| Zero-shot accuracy = 0% | Model too weak. Change `MODEL_NAME` to `Qwen/Qwen2.5-3B-Instruct` in ALL scripts |
| OOM during SFT | Reduce `MAX_SEQ_LEN` to 512 in `step_04` |
| OOM during GRPO | Reduce `NUM_GENERATIONS` to 2, `MAX_NEW_TOKENS` to 128 in `step_06` |
| SFT loss doesn't decrease | Print a formatted prompt from `step_03` and check it looks right |
| GRPO rewards always 0 | SFT accuracy too low. Train SFT for more epochs first |
| Unsloth won't install | Use WSL2 on Windows, or just skip it (scripts fall back to PEFT) |
| TRL API error | Check `pip show trl` version. Try `reward_fn` instead of `reward_funcs` |
| Streamlit won't start | Run from project root: `streamlit run scripts/app.py` |

---

## Google Colab Quick Start

```python
# Cell 1: Install
!pip install unsloth transformers trl peft bitsandbytes accelerate datasets streamlit

# Cell 2: Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 3: Clone or upload the finreason/ folder, then cd into it
%cd /content/finreason

# Cell 4: Run steps
!python scripts/step_00_check_gpu.py
!python scripts/step_01_explore_data.py
# ... etc
```

On Colab T4: increase `BATCH` to 2, `NUM_GENERATIONS` to 8.
On Colab A100: use `Qwen/Qwen2.5-3B-Instruct`, `BATCH`=4.
