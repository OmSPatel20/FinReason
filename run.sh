#!/bin/bash
# =============================================================
#  FinReason — Master Run Script
#  Run: chmod +x run.sh && ./run.sh
# =============================================================
set -e

echo "=============================================="
echo "  FinReason — Full Pipeline"
echo "=============================================="

echo "[0/8] Checking GPU..."
python src/step_00_check_gpu.py

echo "[1/8] Downloading & exploring FinQA..."
python src/step_01_explore_data.py

echo "[Self-test] Reward function..."
python src/shared_utils.py

echo "[2/8] Zero-shot baseline (Checkpoint 1)..."
python src/step_02_zeroshot_baseline.py

echo "[3/8] Formatting SFT data..."
python src/step_03_format_data.py

echo "[4/8] SFT training (~1-2 hrs)..."
python src/sft_train.py

echo "[5/8] Evaluating SFT (Checkpoint 2)..."
python src/step_05_eval_sft.py

echo "[6/8] GRPO training (~2-4 hrs)..."
python src/grpo_train.py

echo "[7/8] Evaluating GRPO (Checkpoint 3)..."
python src/step_07_eval_grpo.py

echo "[8/8] Generating figures..."
python src/analysis.py

echo "=============================================="
echo "  Done! Results in results/ | Demo: streamlit run ui/app.py"
echo "=============================================="
