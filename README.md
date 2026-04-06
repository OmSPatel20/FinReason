# FinReason

**Teaching a Small Language Model Financial Numerical Reasoning via Reinforcement Learning with Verifiable Rewards**

EGN 6217 — Applied Deep Learning | University of Florida | Spring 2026

---

## Project Description

FinReason applies Group Relative Policy Optimization (GRPO) with verifiable rewards to teach Qwen2.5-1.5B-Instruct how to answer numerical questions over real financial reports (S&P 500 earnings filings). The model is trained in two stages — supervised fine-tuning (SFT) to learn table comprehension, followed by GRPO reinforcement learning where the reward is simply "did you get the right number." The project includes a Streamlit demo for interactive financial Q&A.

## Installation and Setup

```bash
# 1. Clone
git clone https://github.com/OmSPatel20/FinReason.git
cd FinReason

# 2. Create environment
conda create -n finreason python=3.11 -y
conda activate finreason

# 3. Install dependencies
pip install -r requirements.txt
pip install unsloth    # optional but recommended (saves ~30% VRAM)
```

**Hardware:** RTX 4060 (8 GB VRAM) or Google Colab T4/A100.

## How to Run

### Setup notebook (verify everything works)

```bash
# From project root:
jupyter notebook notebooks/setup.ipynb
```

Run all cells. This downloads FinQA, explores the data, tests the reward function, and runs a quick zero-shot baseline.

### Full training pipeline

```bash
# Option A: Run everything at once
chmod +x run.sh
./run.sh

# Option B: Step by step
python src/step_00_check_gpu.py       # Verify GPU
python src/step_01_explore_data.py    # Download & explore
python src/step_02_zeroshot.py        # Checkpoint 1
python src/step_03_format_data.py     # Prepare SFT data
python src/sft_train.py               # Stage 1: SFT (~1-2 hrs)
python src/step_05_eval_sft.py        # Checkpoint 2
python src/grpo_train.py              # Stage 2: GRPO (~2-4 hrs)
python src/step_07_eval_grpo.py       # Checkpoint 3
python src/analysis.py                # Generate figures
```

### Interactive demo

```bash
streamlit run ui/app.py
```

## Dataset

**FinQA** (Chen et al., EMNLP 2021) — 8,281 QA pairs over 2,789 S&P 500 earnings reports.  
Source: [HuggingFace — ibm-research/finqa](https://huggingface.co/datasets/ibm-research/finqa)  
License: Creative Commons Attribution 4.0  
No manual download needed — the setup notebook pulls it automatically.

**TAT-QA** (Zhu et al., ACL 2021) — 16,552 questions for out-of-distribution evaluation.  
Source: [HuggingFace — next-tat/TAT-QA](https://huggingface.co/datasets/next-tat/TAT-QA)

## Project Structure

```
FinReason/
├── README.md
├── requirements.txt
├── .gitignore
├── run.sh                    # One-click pipeline
├── data/                     # FinQA data (auto-downloaded)
├── notebooks/
│   └── setup.ipynb           # Environment + data exploration
├── src/
│   ├── shared_utils.py       # Reward function, extraction, metrics
│   ├── sft_train.py          # Stage 1: SFT training
│   ├── grpo_train.py         # Stage 2: GRPO training
│   └── analysis.py           # Figure generation
├── ui/
│   └── app.py                # Streamlit demo
├── results/                  # Figures and outputs
├── checkpoints/              # Model adapters (not in git)
└── docs/                     # Architecture diagrams
```

## Author

**Om Sanjaykumar Patel**  
MS in AI Systems, University of Florida  
Email: <your-email>  
GitHub: <your-github>
