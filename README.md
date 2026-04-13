# FinReason

**Teaching a Small Language Model Financial Numerical Reasoning via Reinforcement Learning with Verifiable Rewards**

EGN 6217 — Applied Deep Learning | University of Florida | Spring 2026

## Quick Start

```bash
conda create -n finreason python=3.11 -y
conda activate finreason
pip install -r requirements.txt
pip install unsloth                                    # optional, saves VRAM
# For PDF support (Windows: download Tesseract installer from github.com/UB-Mannheim/tesseract/wiki)
# Linux: sudo apt install tesseract-ocr poppler-utils

# Run step by step:
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

# Launch demo:
streamlit run ui/app.py

# Live training monitor (run in separate terminal during training):
streamlit run ui/training_monitor.py
```

## Project Structure

```
FinReason/
├── README.md
├── TECHNICAL_REFERENCE.md          # How everything works (read this)
├── requirements.txt
├── .gitignore
├── notebooks/setup.ipynb           # Required: environment + data exploration
├── src/
│   ├── shared_utils.py             # Reward function, metrics, extraction
│   ├── training_logger.py          # Live log callback for training
│   ├── pdf_extractor.py            # OCR + multi-currency PDF extraction
│   ├── step_00_check_gpu.py
│   ├── step_01_explore_data.py
│   ├── step_02_zeroshot_baseline.py
│   ├── step_03_format_data.py
│   ├── sft_train.py                # Stage 1: SFT
│   ├── step_05_eval_sft.py
│   ├── grpo_train.py               # Stage 2: GRPO
│   ├── step_07_eval_grpo.py
│   └── analysis.py
├── ui/
│   ├── app.py                      # Streamlit demo
│   └── training_monitor.py         # Live training dashboard
├── docs/technical_blueprint.pdf
├── data/
├── results/
└── checkpoints/
```

## Dataset

**FinQA** — 8,281 QA pairs over S&P 500 earnings reports. Auto-downloaded.
**TAT-QA** — 16,552 questions for OOD evaluation. Optional.

## Author

Om — MS in AI Systems, University of Florida
