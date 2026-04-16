<p align="center">
  <h1 align="center">📊 FinReason</h1>
  <p align="center">
    <strong>How Small Can You Go? A Multi-Scale Study of GRPO<br>for Financial Numerical Reasoning</strong>
  </p>
  <p align="center">
    EGN 6217 — Applied Deep Learning · University of Florida · Spring 2026
  </p>
</p>

---

## What Is This Project?

FinReason systematically tests whether **GRPO (Group Relative Policy Optimization) with verifiable rewards** — the same reinforcement learning technique behind DeepSeek-R1 — can teach small language models to answer numerical questions about financial reports.

The key question: **how small can the model be and still benefit from GRPO?**

I test three scales — **1.5B, 3B, and 7B parameters** — using the same two-stage pipeline (SFT → GRPO), the same dataset (FinQA), and the same reward function. The scale where GRPO starts producing gains over SFT is the **minimum viable scale for RL-based financial reasoning**.


---

## Why Does This Matter?

**The problem is real.** Financial analysts spend hours extracting and computing figures from earnings reports. LLMs promise to automate this, but they hallucinate numbers, confuse table rows, and skip arithmetic steps. Even frontier LLMs hit only ~60% accuracy on FinQA; human experts score ~91%.

**The technique is cutting-edge.** GRPO with verifiable rewards is the #1 technique in LLM post-training (2025-2026). DeepSeek-R1, OpenAI's o-series, and other reasoning models all use it. Applying it to financial reasoning at multiple scales is novel.

**The constraint is practical.** Running GRPO on small models matters for on-device deployment, cost-constrained environments, and organizations that can't afford A100 clusters. Understanding the minimum viable scale informs real deployment decisions.

---

## How It Works

### The Two-Stage Pipeline (Applied at Each Scale)

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  FinQA Dataset (6,624 train + 1,657 test QA pairs)                  │
│       │                                                             │
│       ▼                                                             │
│  ┌──────────────────────────────────┐                               │
│  │  STAGE 1: Supervised Fine-Tuning │  ← Teaches the model to      │
│  │  (SFT with QLoRA)               │    read financial tables,     │
│  │  + Chain-of-thought warmup       │    produce <think> reasoning  │
│  └──────────────┬───────────────────┘                               │
│                 │                                                    │
│                 ▼                                                    │
│  ┌──────────────────────────────────┐     ┌───────────────────┐     │
│  │  STAGE 2: GRPO Reinforcement    │◄────│  Reward Function  │     │
│  │  Learning                        │────►│  correct = +1     │     │
│  │  4 completions per question      │     │  wrong   =  0     │     │
│  └──────────────┬───────────────────┘     │  <think> = +0.2   │     │
│                 │                          └───────────────────┘     │
│                 ▼                                                    │
│  Three Checkpoints Compared at Each Scale:                          │
│    1. Zero-Shot (no training)                                       │
│    2. After SFT                                                     │
│    3. After SFT + GRPO                                              │
│                                                                     │
│  Repeated for: 1.5B → 3B → 7B                                      │
│  The scale where GRPO > SFT = minimum viable scale                  │
└─────────────────────────────────────────────────────────────────────┘
```

### What GRPO Does

For each financial question, the model generates **4 different answers**. A rule-based function checks each: did you get the right number? Answers above the group average get reinforced; answers below get suppressed. No neural reward model, no human feedback — just "is the number correct?"

### The Reward Function

```python
def reward_function(model_output, ground_truth):
    correct = check_answer(extract_final_answer(model_output), ground_truth)  # ±1% tolerance
    has_think = bool(re.search(r'<think>.*?</think>', model_output))
    return (1.0 if correct else 0.0) + (0.2 if has_think else 0.0)
```

40 lines of Python. No neural network. Handles 25+ currency formats ($, €, £, RM, ₹, ¥, etc.).

---

## Multi-Scale Results

| Scale | Zero-Shot | SFT | GRPO | GRPO − SFT (Δ) | Think Rate |
|---|---|---|---|---|---|
| **1.5B** | 3.7% | 7.0% | 6.7% | −0.3% | 0% |
| **3B** | TBD | TBD | TBD | TBD | TBD |
| **7B** | TBD | TBD | TBD | TBD | TBD |

**Key finding at 1.5B:** GRPO does not improve over SFT. The primary bottleneck is reward sparsity — with SFT accuracy at only 7%, GRPO rarely sees correct completions during sampling (mean reward 0.06), producing near-zero gradient signal.

**Research question:** Does 3B provide enough capacity for GRPO to cross the threshold?

---

## Dataset

### FinQA (Primary)

| Property | Details |
|----------|---------|
| **Source** | [wandb/finqa-data-processed](https://huggingface.co/datasets/wandb/finqa-data-processed) on HuggingFace |
| **Origin** | Chen et al., EMNLP 2021. Created by financial experts at UPenn. |
| **Size** | 6,624 training + 1,657 test QA pairs |
| **Content** | Real S&P 500 earnings reports with tables, text, questions, and verified answers |
| **Columns** | `query`, `context`, `exe_ans`, `table`, `pre_text`, `post_text`, `program` |
| **License** | Creative Commons 4.0 |

The `program` column contains computation steps (e.g., `subtract(1452.4, 1146.2)`) which we convert to `<think>` reasoning traces during SFT training.

### Cross-Currency Testing

We additionally test on the **J.P. Morgan Chase Bank Berhad Q4 2024** annual report (Malaysian Ringgit, MFRS standards) to evaluate cross-currency generalization.

---

## Technology Stack

| Component | Library | Why |
|-----------|---------|-----|
| Base models | Qwen2.5 (1.5B / 3B / 7B) | Best open models at each scale |
| Quantization | bitsandbytes (NF4 4-bit) | Fits larger models in limited VRAM |
| Fine-tuning | PEFT (QLoRA, rank 16) | Trains ~1% of parameters |
| SFT | TRL SFTTrainer | Standard supervised fine-tuning |
| GRPO | TRL GRPOTrainer | DeepSeek-R1's RL algorithm |
| Inference | HuggingFace Transformers | Model loading, generation |
| Data | HuggingFace Datasets | One-line dataset download |
| Web interface | Streamlit | Interactive multi-scale demo |
| PDF extraction | pdfplumber + pytesseract | Handles digital + scanned PDFs |
| Plotting | matplotlib | Generates all report figures |

---

## Project Structure

```
FinReason/
│
├── README.md                                ← You are here
├── TECHNICAL_REFERENCE.md                   ← How every file and function works
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── shared_utils.py                      ← Reward function, metrics, extraction
│   ├── training_logger.py                   ← Live log callback
│   ├── pdf_extractor.py                     ← OCR + multi-currency (25+ currencies)
│   ├── run_scale.py                         ← One-command full pipeline per scale
│   │
│   ├── step_00_check_gpu.py                 ← Verify GPU + packages
│   ├── step_01_explore_data.py              ← Download + explore FinQA
│   ├── step_02_zeroshot_baseline.py         ← Checkpoint 1
│   ├── step_03_format_data.py               ← Format data with <think> tags
│   ├── sft_train.py                         ← Stage 1: SFT
│   ├── step_05_eval_sft.py                  ← Checkpoint 2
│   ├── grpo_train.py                        ← Stage 2: GRPO
│   ├── step_07_eval_grpo.py                 ← Checkpoint 3
│   └── analysis.py                          ← Figure generation
│
├── ui/
│   ├── app.py                               ← Streamlit demo (multi-scale + comparison)
│   └── training_monitor.py                  ← Live training dashboard
│
├── notebooks/
│   ├── setup.ipynb                          ← Environment + data exploration
│   └── colab_run.ipynb                      ← Full Colab pipeline
│
├── docs/
│   ├── technical_blueprint.pdf              ← Deliverable 1.2
│   └── deliverable2_report.pdf              ← Deliverable 2
│
├── outputs/                                 ← Results JSONs (per scale)
│   ├── 1.5B/
│   │   ├── zeroshot_results.json
│   │   ├── sft_results.json
│   │   └── grpo_results.json
│   ├── 3B/
│   │   └── ...
│   └── 7B/
│       └── ...
│
├── results/                                 ← Figures (per scale)
│   ├── 1.5B/fig1_accuracy.png, fig2_grpo_curves.png
│   ├── 3B/...
│   └── 7B/...
│
├── checkpoints/                             ← Model adapters (per scale, not in git)
│   ├── 1.5B/sft/final_adapter, grpo/final_adapter
│   ├── 3B/sft/final_adapter, grpo/final_adapter
│   └── 7B/sft/final_adapter, grpo/final_adapter
│
└── data/                                    ← Auto-populated by scripts
    ├── train_sft.json
    └── val_sft.json
```

---

## Installation

```bash
git clone https://github.com/OmSPatel20/FinReason.git
cd FinReason
conda create -n finreason python=3.11 -y
conda activate finreason
pip install -r requirements.txt
```

---

## How to Run

### Option 1: One Command Per Scale (recommended)

```bash
python src/run_scale.py 1.5B    # ~5 hours on A100
python src/run_scale.py 3B      # ~5 hours on A100
python src/run_scale.py 7B      # ~8 hours on A100
```

Each command runs the **entire pipeline** for that scale: zero-shot → SFT → GRPO → eval → figures. Results saved in `outputs/<scale>/` and `checkpoints/<scale>/`.

### Option 2: Step by Step

```bash
python src/step_00_check_gpu.py
python src/shared_utils.py
python src/step_01_explore_data.py
python src/step_02_zeroshot_baseline.py
python src/step_03_format_data.py
python src/sft_train.py
python src/step_05_eval_sft.py
python src/grpo_train.py
python src/step_07_eval_grpo.py
python src/analysis.py
```

### Google Colab

```python
# Cell 1: Install
!pip uninstall -y -q torch torchvision torchaudio
!pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
!pip install transformers==4.48.0 trl==0.15.2 peft accelerate bitsandbytes datasets matplotlib pandas numpy tqdm seaborn
!pip uninstall -y wandb

# Cell 2: Clone
!git clone https://github.com/OmSPatel20/FinReason.git
%cd FinReason

# Cell 3: Run each scale
!python src/run_scale.py 1.5B
!python src/run_scale.py 3B
!python src/run_scale.py 7B

# Cell 4: Save to Drive
from google.colab import drive
drive.mount('/content/drive')
!cp -r outputs /content/drive/MyDrive/FinReason_Results/
!cp -r results /content/drive/MyDrive/FinReason_Results/
!cp -r checkpoints /content/drive/MyDrive/FinReason_Results/
```

### Launch the Demo

```bash
streamlit run ui/app.py
```

Features:
- **Model Scale selector** — switch between 1.5B, 3B, 7B
- **Training Stage selector** — switch between Zero-Shot, SFT, GRPO
- **PDF upload** — test on any financial report in any currency
- **Reasoning display** — shows `<think>` traces when the model uses them
- **Cross-Scale Comparison** — table and chart showing all scales side by side

---

## Hardware

### What We Used

| Hardware | Role | Time |
|----------|------|------|
| HP Omen RTX 4060 (8GB) | Development, testing, Streamlit demo | — |
| Google Colab Pro A100 (80GB) | Training all three scales | ~18 hrs total |

### Per-Scale Training Time (A100)

| Scale | SFT | GRPO | Total |
|---|---|---|---|
| 1.5B | ~1.5 hrs | ~2-3 hrs | ~4 hrs |
| 3B | ~2-3 hrs | ~3-4 hrs | ~6 hrs |
| 7B | ~3-4 hrs | ~4-5 hrs | ~8 hrs |

### If Hardware Were No Constraint

With an H100 cluster, the study could be extended to:
- **14B and 32B** scales (complete the scaling curve)
- **Group size G=16 or 32** for stronger GRPO signal
- **Full fine-tuning** instead of QLoRA
- **Multi-lingual training** on IFRS filings from Europe, Asia, and emerging markets
- **Longer context (8K+ tokens)** to handle complete financial filings without truncation

---

## References

1. Z. Chen et al., "FinQA: A Dataset of Numerical Reasoning over Financial Data," *EMNLP*, 2021.
2. DeepSeek-AI, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning," *arXiv:2501.12948*, 2025.
3. Z. Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models," *arXiv:2402.03300*, 2024.
4. T. Dettmers et al., "QLoRA: Efficient Finetuning of Quantized Language Models," *NeurIPS*, 2023.
5. Z. Liu et al., "Fin-R1: A Large Language Model for Financial Reasoning through RL," *arXiv:2503.16252*, 2025.
6. L. Qian et al., "Fin-o1: On the Transferability of Reasoning-Enhanced LLMs to Finance," *arXiv:2502.08127*, 2025.
7. F. Zhu et al., "TAT-QA: A Question Answering Benchmark on Hybrid Tabular and Textual Content in Finance," *ACL*, 2021.
8. J. Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models," *NeurIPS*, 2022.
9. L. von Werra et al., "TRL: Transformer Reinforcement Learning," GitHub, 2022.

---

## Author

**Om Sanjaykumar Patel**
MS in AI Systems, University of Florida
GitHub: [OmSPatel20](https://github.com/OmSPatel20)

---

## License

Academic use (EGN 6217 coursework). FinQA dataset: CC-BY 4.0. Model weights: Qwen license terms.
