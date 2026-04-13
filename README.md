<p align="center">
  <h1 align="center">📊 FinReason</h1>
  <p align="center">
    <strong>Teaching a Small Language Model Financial Numerical Reasoning<br>via Reinforcement Learning with Verifiable Rewards</strong>
  </p>
  <p align="center">
    EGN 6217 — Applied Deep Learning · University of Florida · Spring 2026
  </p>
</p>

---

## What Is This Project?

FinReason takes a small, open-source language model (**Qwen2.5-1.5B-Instruct** — only 1.5 billion parameters) and teaches it to answer numerical questions about financial reports. Think questions like:

> *"What was the year-over-year change in operating income from 2018 to 2019?"*

The model needs to read a financial table, find the right numbers, and do the math — subtraction, division, percentage change. This is something even GPT-4 gets wrong ~40% of the time on the FinQA benchmark.

**The key idea:** Instead of just showing the model correct answers (supervised fine-tuning), we also train it with **reinforcement learning** where it gets a simple reward: **right number = +1, wrong number = 0**. This is called **GRPO (Group Relative Policy Optimization) with Verifiable Rewards** — the same technique behind DeepSeek-R1, one of the most capable reasoning models in 2025.

---

## Why Does This Matter?

**The problem is real.** Financial analysts at banks, audit firms, and regulators spend hours manually extracting and computing figures from earnings reports. LLMs promise to automate this, but they hallucinate numbers, confuse table rows, and skip arithmetic steps.

**The gap is measurable.** On the FinQA benchmark, even frontier LLMs hit only ~60% accuracy. Human financial experts score ~91%. That 30-point gap is the problem this project attacks.

**The technique is cutting-edge.** GRPO with verifiable rewards is the #1 technique in LLM post-training right now (2025-2026). It's how DeepSeek-R1, OpenAI's o-series, and other reasoning models are built. Applying it to financial domain reasoning is novel — all published GRPO work targets math competitions and code, not real-world financial documents.

**The constraint is practical.** The entire pipeline runs on a single consumer GPU with 8 GB VRAM. No datacenter needed. This proves the technique works under real hardware constraints, which matters for anyone thinking about on-device or cost-constrained deployment.

---

## How It Works

### The Two-Stage Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  FinQA Dataset (8,281 QA pairs from S&P 500 earnings reports)       │
│       │                                                             │
│       ▼                                                             │
│  ┌──────────────────────────────────┐                               │
│  │  STAGE 1: Supervised Fine-Tuning │  ← Teaches the model to      │
│  │  (SFT with QLoRA)               │    read financial tables and  │
│  │  ~1-2 hours on GPU              │    produce short answers      │
│  └──────────────┬───────────────────┘                               │
│                 │                                                    │
│                 ▼                                                    │
│  ┌──────────────────────────────────┐     ┌───────────────────┐     │
│  │  STAGE 2: GRPO Reinforcement    │◄────│  Reward Function  │     │
│  │  Learning                        │────►│  correct = +1     │     │
│  │  ~2-4 hours on GPU              │     │  wrong   =  0     │     │
│  └──────────────┬───────────────────┘     │  <think> = +0.2   │     │
│                 │                          └───────────────────┘     │
│                 ▼                                                    │
│  Three Model Checkpoints Compared:                                  │
│    1. Zero-Shot (no training)     → ~0-5% accuracy                  │
│    2. After SFT                   → ~20-30% accuracy                │
│    3. After SFT + GRPO           → ~30-45% accuracy                 │
│                                                                     │
│  Streamlit Demo App (paste data or upload PDF, ask questions)       │
└─────────────────────────────────────────────────────────────────────┘
```

### What GRPO Does (in plain English)

For each financial question, the model generates **4 different answers**. A simple rule-based function checks each one — did you get the right number? The answers that scored above the group average get reinforced (model learns to produce more of these). The ones below average get suppressed. No neural reward model, no human feedback — just "is the number correct?"

Over thousands of questions, the model discovers that **reasoning step-by-step before answering** leads to more correct answers. It starts producing `<think>` traces on its own — showing its work like a human analyst would.

---

## Dataset

### FinQA (Primary — Training & Evaluation)

| Property | Details |
|----------|---------|
| **Source** | [wandb/finqa-data-processed](https://huggingface.co/datasets/wandb/finqa-data-processed) on HuggingFace |
| **Origin** | Chen et al., EMNLP 2021. Created by financial experts at UPenn. |
| **Size** | 6,624 training + 1,657 test QA pairs |
| **Content** | Real S&P 500 earnings reports with tables, text, questions, and verified numerical answers |
| **Columns** | `query`, `context`, `exe_ans`, `table`, `pre_text`, `post_text`, `program` |
| **License** | Creative Commons 4.0 — fully public, no access restrictions |
| **Download** | Automatic via `datasets` library. ~32 MB total. |

### TAT-QA (Secondary — Out-of-Distribution Testing)

| Property | Details |
|----------|---------|
| **Source** | [next-tat/TAT-QA](https://huggingface.co/datasets/next-tat/TAT-QA) on HuggingFace |
| **Size** | 16,552 questions over 2,757 financial report contexts |
| **Purpose** | Tests whether GRPO-trained reasoning generalizes to unseen question types |

### Why These Datasets?

Financial QA is ideal for GRPO because **every answer is a number that can be verified**. You don't need a neural reward model or human annotators to judge quality — a Python function can check if `306.2 == 306.2`. That's what makes the verifiable reward possible.

---

## Technology Stack

| Component | Library | Why |
|-----------|---------|-----|
| Base model | Qwen2.5-1.5B-Instruct | Largest model that fits in 8GB VRAM with 4-bit quantization |
| Quantization | bitsandbytes (NF4 4-bit) | Compresses model from ~3GB to ~0.8GB in VRAM |
| Fine-tuning | PEFT (QLoRA, rank 16) | Trains only ~1% of parameters — fast, memory-efficient |
| SFT trainer | TRL SFTTrainer | HuggingFace's standard supervised fine-tuning |
| GRPO trainer | TRL GRPOTrainer | HuggingFace's implementation of the GRPO algorithm |
| Inference | HuggingFace Transformers | Model loading, tokenization, generation |
| Data | HuggingFace Datasets | One-line dataset download |
| Memory optimization | Unsloth (optional) | Saves ~30% VRAM, faster training |
| Web interface | Streamlit | Interactive demo with PDF upload |
| PDF extraction | pdfplumber + pytesseract | Handles both digital and scanned PDFs |
| Live monitoring | Custom LiveLogCallback | Real-time training dashboard |
| Plotting | matplotlib | Generates all report figures |

---

## Project Structure

```
FinReason/
│
├── README.md                           ← You are here
├── TECHNICAL_REFERENCE.md              ← Deep dive: how every file and function works
├── requirements.txt                    ← All Python dependencies
├── .gitignore                          ← Keeps checkpoints and cache out of git
│
├── src/                                ← All source code
│   ├── shared_utils.py                 ← Core module: reward function, metrics, data extraction
│   ├── training_logger.py              ← Live log callback for training dashboard
│   ├── pdf_extractor.py                ← OCR + multi-currency PDF processing (25+ currencies)
│   │
│   ├── step_00_check_gpu.py            ← Verify GPU, CUDA, all packages
│   ├── step_01_explore_data.py         ← Download FinQA, analyze answer types
│   ├── step_02_zeroshot_baseline.py    ← Checkpoint 1: base model (no training)
│   ├── step_03_format_data.py          ← Format data into chat template for SFT
│   ├── sft_train.py                    ← Stage 1: SFT with QLoRA (~1-2 hours)
│   ├── step_05_eval_sft.py             ← Checkpoint 2: SFT model evaluation
│   ├── grpo_train.py                   ← Stage 2: GRPO with verifiable rewards (~2-4 hours)
│   ├── step_07_eval_grpo.py            ← Checkpoint 3: GRPO model evaluation (final)
│   └── analysis.py                     ← Generate all 4 report figures
│
├── ui/                                 ← User interfaces
│   ├── app.py                          ← Streamlit demo (interactive Q&A + PDF upload)
│   └── training_monitor.py             ← Live training dashboard (loss, reward, VRAM)
│
├── notebooks/                          ← Jupyter notebooks
│   ├── setup.ipynb                     ← Environment check + data exploration (required for submission)
│   └── colab_run.ipynb                 ← Full pipeline notebook for Google Colab
│
├── docs/                               ← Reports and documentation
│   ├── technical_blueprint.pdf         ← Deliverable 1.2 blueprint
│   └── deliverable2_report.pdf         ← Deliverable 2 implementation report
│
├── data/                               ← Auto-populated by scripts
│   ├── train_sft.json                  ← Formatted training data (created by step_03)
│   ├── val_sft.json                    ← Formatted validation data (created by step_03)
│   └── format_info.json                ← Dataset metadata (created by step_01)
│
├── results/                            ← Figures and visualizations
│   ├── fig1_accuracy.png               ← 3-checkpoint accuracy comparison
│   ├── fig2_grpo_curves.png            ← GRPO loss and reward during training
│   ├── fig3_error_analysis.png         ← SFT vs GRPO per-example comparison
│   ├── fig4_think_analysis.png         ← Does reasoning help? With vs without <think>
│   └── data_exploration.png            ← Answer type and context length distributions
│
├── outputs/                            ← JSON results from evaluation
│   ├── zeroshot_results.json           ← Checkpoint 1 detailed results
│   ├── sft_results.json                ← Checkpoint 2 detailed results
│   ├── grpo_results.json               ← Checkpoint 3 detailed results
│   └── grpo_training_log.json          ← Training metrics over time
│
└── checkpoints/                        ← Model weights (not in git — too large)
    ├── sft/final_adapter/              ← SFT LoRA adapter (~10-30 MB)
    └── grpo/final_adapter/             ← GRPO LoRA adapter (~10-30 MB)
```

---

## Installation & Setup

### Prerequisites

- **Python 3.11**
- **NVIDIA GPU** with CUDA support
- **conda** (Anaconda or Miniconda)
- **Tesseract OCR** (only needed for PDF extraction feature)
  - Windows: Download from [github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)
  - Linux: `sudo apt install tesseract-ocr poppler-utils`
  - Mac: `brew install tesseract poppler`

### Install

```bash
git clone https://github.com/OmSPatel20/FinReason.git
cd FinReason
conda create -n finreason python=3.11 -y
conda activate finreason
pip install -r requirements.txt
pip install unsloth    # optional but recommended — saves ~30% VRAM
```

---

## How to Run

### Local Machine (step by step)

```bash
conda activate finreason

python src/step_00_check_gpu.py            # Verify GPU          (~10 sec)
python src/shared_utils.py                 # Test reward function (~5 sec, must show all ✓)
python src/step_01_explore_data.py         # Download FinQA      (~2 min)
python src/step_02_zeroshot_baseline.py    # Checkpoint 1        (~20 min)
python src/step_03_format_data.py          # Format data         (~1 min)
python src/sft_train.py                    # Stage 1: SFT        (~1-2 hrs)
python src/step_05_eval_sft.py             # Checkpoint 2        (~20 min)
python src/grpo_train.py                   # Stage 2: GRPO       (~2-4 hrs)
python src/step_07_eval_grpo.py            # Checkpoint 3        (~20 min)
python src/analysis.py                     # Generate figures     (~1 min)
```

### Google Colab

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Runtime → Change runtime type → **T4 GPU**
3. Upload `notebooks/colab_run.ipynb` or run these cells:

```python
!pip install -q torch torchvision torchaudio
!pip install -q transformers trl peft bitsandbytes accelerate datasets mergekit
!pip install -q matplotlib seaborn pandas numpy tqdm unsloth

!git clone https://github.com/OmSPatel20/FinReason.git
%cd FinReason

!python src/step_00_check_gpu.py
!python src/shared_utils.py
!python src/step_01_explore_data.py
!python src/step_02_zeroshot_baseline.py
!python src/step_03_format_data.py
!python src/sft_train.py
!python src/step_05_eval_sft.py
!python src/grpo_train.py
!python src/step_07_eval_grpo.py
!python src/analysis.py
```

### Launch the Demo

```bash
streamlit run ui/app.py                    # Interactive Q&A demo
streamlit run ui/training_monitor.py       # Live training dashboard (during training)
```

### Run the Setup Notebook (required for course submission)

```bash
jupyter notebook notebooks/setup.ipynb     # Run all cells, save with outputs
```

---

## Hardware Requirements

### Minimum (what this project was built for)

| Component | Spec |
|-----------|------|
| GPU | NVIDIA RTX 4060 Laptop (8 GB VRAM) |
| CPU | Intel i7-14th Gen HX |
| RAM | 16 GB |
| Storage | ~10 GB free (model cache + data + checkpoints) |
| OS | Windows 11 / Linux / macOS (with NVIDIA GPU) |

### Recommended (faster training)

| Component | Spec |
|-----------|------|
| GPU | Google Colab T4 (16 GB) or A100 (40 GB) |
| Training time | T4: ~3-4 hrs total · A100: ~1-2 hrs total |

### If Hardware Were No Constraint

With an A100 (80 GB) or H100 cluster, the project could be scaled to:
- **Qwen2.5-7B or 14B** as the base model (much stronger reasoning baseline)
- **Group size G=16 or 32** for better GRPO signal
- **Full FinQA + TAT-QA + ConvFinQA** combined training (~30K examples)
- **Multi-turn financial dialogue** (conversational QA over reports)
- **Full fine-tuning** instead of QLoRA (higher quality but 10x more VRAM)
- **Longer context (4096+ tokens)** to handle complete financial filings without truncation
- **DPO/RLHF** on top of GRPO for better output formatting and safety alignment
- **Multi-lingual training** on IFRS filings from Europe, Asia, and emerging markets

---

## Key Results (Expected)

| Checkpoint | Execution Accuracy | Think Rate | Description |
|------------|-------------------|------------|-------------|
| Zero-Shot | 0-5% | 0% | Base model with no training |
| SFT | 18-30% | 0-5% | After supervised fine-tuning |
| **SFT + GRPO** | **25-40%** | **20-50%** | After reinforcement learning |

The main finding: GRPO with verifiable rewards measurably improves financial numerical reasoning beyond what SFT alone achieves, and the model develops emergent step-by-step reasoning (the `<think>` traces) without being explicitly forced to.

---

## References

1. Z. Chen et al., "FinQA: A Dataset of Numerical Reasoning over Financial Data," *Proc. EMNLP*, 2021.
2. DeepSeek-AI, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning," *arXiv:2501.12948*, 2025. Published in *Nature*, 2025.
3. Z. Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models," *arXiv:2402.03300*, 2024.
4. T. Dettmers et al., "QLoRA: Efficient Finetuning of Quantized Language Models," *Proc. NeurIPS*, 2023.
5. L. von Werra et al., "TRL: Transformer Reinforcement Learning," GitHub, 2022.
6. F. Zhu et al., "TAT-QA: A Question Answering Benchmark on a Hybrid of Tabular and Textual Content in Finance," *Proc. ACL*, 2021.
7. J. Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models," *Proc. NeurIPS*, 2022.

---

## Author

**Om Sanjaykumar Patel**
MS in AI Systems, University of Florida
GitHub: [OmSPatel20](https://github.com/OmSPatel20)

---

## License

This project is for academic purposes (EGN 6217 coursework). The FinQA dataset is licensed under Creative Commons Attribution 4.0. Model weights are subject to Qwen's license terms.
