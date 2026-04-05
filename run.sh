#!/bin/bash
# =============================================================
#  FinReason — Master Run Script
#
#  HOW TO RUN:
#    chmod +x run.sh
#    ./run.sh
#
#  Or run steps individually (see below).
# =============================================================

set -e  # Stop on first error

echo "=============================================="
echo "  FinReason — Full Pipeline"
echo "=============================================="
echo ""

# Step 0: Check GPU and dependencies
echo "[Step 0/8] Checking GPU and dependencies..."
python scripts/step_00_check_gpu.py
echo ""

# Step 1: Download and explore FinQA
echo "[Step 1/8] Downloading and exploring FinQA..."
python scripts/step_01_explore_data.py
echo ""

# Self-test: Verify reward function works
echo "[Self-test] Testing reward function..."
python scripts/shared_utils.py
echo ""

# Step 2: Zero-shot baseline (CHECKPOINT 1)
echo "[Step 2/8] Zero-shot baseline evaluation..."
python scripts/step_02_zeroshot_baseline.py
echo ""

# Step 3: Format data for SFT
echo "[Step 3/8] Formatting data for SFT..."
python scripts/step_03_format_data.py
echo ""

# Step 4: SFT training (~1-2 hours)
echo "[Step 4/8] SFT training (Stage 1)..."
echo "  This takes ~1-2 hours on RTX 4060."
python scripts/step_04_sft_train.py
echo ""

# Step 5: Evaluate SFT (CHECKPOINT 2)
echo "[Step 5/8] Evaluating SFT model..."
python scripts/step_05_eval_sft.py
echo ""

# Step 6: GRPO training (~2-4 hours)
echo "[Step 6/8] GRPO training (Stage 2)..."
echo "  This takes ~2-4 hours on RTX 4060."
python scripts/step_06_grpo_train.py
echo ""

# Step 7: Evaluate GRPO (CHECKPOINT 3)
echo "[Step 7/8] Evaluating GRPO model..."
python scripts/step_07_eval_grpo.py
echo ""

# Step 8: Generate figures
echo "[Step 8/8] Generating analysis figures..."
python scripts/step_08_analysis.py
echo ""

echo "=============================================="
echo "  ✓ Pipeline complete!"
echo ""
echo "  Results:  outputs/*.json"
echo "  Figures:  outputs/*.png"
echo "  Demo:     streamlit run scripts/app.py"
echo "=============================================="
