"""
training_monitor.py — Live Training Dashboard

Run in a SEPARATE terminal while training is running:
    streamlit run ui/training_monitor.py

Shows real-time:
  - Loss curve (SFT or GRPO)
  - Reward curve (GRPO only)
  - Current step, epoch, learning rate
  - Estimated time remaining
  - GPU VRAM usage
  - Recent log entries table
  - Auto-refreshes every 5 seconds
"""
import streamlit as st
import json
import os
import time
import glob

st.set_page_config(page_title="FinReason Training Monitor", page_icon="📈", layout="wide")

# =====================================================================
# CONFIG
# =====================================================================
SFT_LOG_DIR = "checkpoints/sft"
GRPO_LOG_DIR = "checkpoints/grpo"
SFT_LOG_FILE = "checkpoints/sft/training_log.json"
GRPO_LOG_FILE = "outputs/grpo_training_log.json"
REFRESH_INTERVAL = 5  # seconds

# =====================================================================
# HEADER
# =====================================================================
st.title("📈 FinReason — Live Training Monitor")
st.caption("Auto-refreshes every 5 seconds. Run this in a separate terminal while training.")

# =====================================================================
# GPU STATUS
# =====================================================================
def get_gpu_stats():
    """Get current GPU utilization."""
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            name = torch.cuda.get_device_name(0)
            return {
                "name": name,
                "used_gb": allocated,
                "total_gb": total,
                "pct": (allocated / total) * 100,
            }
    except Exception:
        pass

    # Fallback: try nvidia-smi
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            used = float(parts[1]) / 1024
            total = float(parts[2]) / 1024
            return {
                "name": parts[0],
                "used_gb": used,
                "total_gb": total,
                "pct": (used / total) * 100,
            }
    except Exception:
        pass

    return None


gpu = get_gpu_stats()
if gpu:
    col_gpu1, col_gpu2, col_gpu3 = st.columns(3)
    col_gpu1.metric("GPU", gpu["name"])
    col_gpu2.metric("VRAM Used", f"{gpu['used_gb']:.1f} / {gpu['total_gb']:.1f} GB")
    col_gpu3.metric("VRAM %", f"{gpu['pct']:.0f}%")
    st.progress(min(gpu["pct"] / 100, 1.0))
else:
    st.info("GPU stats not available (training may not be running)")

st.markdown("---")

# =====================================================================
# DETECT WHICH TRAINING IS RUNNING
# =====================================================================
def find_latest_trainer_state():
    """Find the most recently modified trainer_state.json."""
    patterns = [
        "checkpoints/sft/*/trainer_state.json",
        "checkpoints/sft/trainer_state.json",
        "checkpoints/grpo/*/trainer_state.json",
        "checkpoints/grpo/trainer_state.json",
    ]
    all_states = []
    for pattern in patterns:
        all_states.extend(glob.glob(pattern))

    if not all_states:
        return None, None

    latest = max(all_states, key=os.path.getmtime)
    stage = "SFT" if "sft" in latest else "GRPO"
    return latest, stage


def load_log(path):
    """Load a training log JSON file."""
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def load_trainer_state(path):
    """Load HuggingFace trainer_state.json."""
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


# =====================================================================
# DISPLAY TRAINING STATUS
# =====================================================================

# Check for active training
state_path, active_stage = find_latest_trainer_state()
trainer_state = load_trainer_state(state_path)

# Also check for saved logs
sft_log = load_log(SFT_LOG_FILE)
grpo_log = load_log(GRPO_LOG_FILE)

# Determine what to show
tab1, tab2 = st.tabs(["🔧 SFT (Stage 1)", "🧠 GRPO (Stage 2)"])

# =====================================================================
# SFT TAB
# =====================================================================
with tab1:
    # Try trainer_state first (live), then saved log
    entries = []

    if trainer_state and active_stage == "SFT":
        entries = trainer_state.get("log_history", [])
        st.success("🟢 SFT training is LIVE")
    elif sft_log:
        entries = sft_log
        st.info("SFT training completed. Showing saved log.")
    else:
        st.warning("No SFT training data found. Start training with: `python src/sft_train.py`")

    if entries:
        # Extract metrics
        train_losses = [(e.get("step", i), e["loss"])
                        for i, e in enumerate(entries) if "loss" in e and "eval_loss" not in e]
        eval_losses = [(e.get("step", i), e["eval_loss"])
                       for i, e in enumerate(entries) if "eval_loss" in e]
        lrs = [(e.get("step", i), e["learning_rate"])
               for i, e in enumerate(entries) if "learning_rate" in e]

        # Status metrics
        col1, col2, col3, col4 = st.columns(4)

        if train_losses:
            latest_loss = train_losses[-1][1]
            first_loss = train_losses[0][1]
            col1.metric("Current Loss", f"{latest_loss:.4f}",
                        delta=f"{latest_loss - first_loss:.4f}")

        if eval_losses:
            col2.metric("Eval Loss", f"{eval_losses[-1][1]:.4f}")

        current_step = entries[-1].get("step", len(entries))
        col3.metric("Step", current_step)

        if trainer_state:
            max_steps = trainer_state.get("max_steps", "?")
            epoch = trainer_state.get("epoch", "?")
            col4.metric("Epoch", f"{epoch:.1f}" if isinstance(epoch, float) else epoch)

        # Loss chart
        if train_losses:
            import pandas as pd
            st.subheader("Training Loss")

            df_train = pd.DataFrame(train_losses, columns=["Step", "Train Loss"])
            chart_data = df_train.set_index("Step")

            if eval_losses:
                df_eval = pd.DataFrame(eval_losses, columns=["Step", "Eval Loss"])
                chart_data = chart_data.join(df_eval.set_index("Step"), how="outer")

            st.line_chart(chart_data)

        # Learning rate
        if lrs:
            st.subheader("Learning Rate Schedule")
            df_lr = pd.DataFrame(lrs, columns=["Step", "LR"])
            st.line_chart(df_lr.set_index("Step"))

        # Recent entries table
        with st.expander("📋 Recent Log Entries"):
            recent = entries[-20:]
            st.json(recent)


# =====================================================================
# GRPO TAB
# =====================================================================
with tab2:
    entries = []

    if trainer_state and active_stage == "GRPO":
        entries = trainer_state.get("log_history", [])
        st.success("🟢 GRPO training is LIVE")
    elif grpo_log:
        entries = grpo_log
        st.info("GRPO training completed. Showing saved log.")
    else:
        st.warning("No GRPO training data found. Start training with: `python src/grpo_train.py`")

    if entries:
        # Extract GRPO-specific metrics
        losses = [(e.get("step", i), e["loss"])
                  for i, e in enumerate(entries) if "loss" in e]

        # GRPO logs rewards under various keys depending on TRL version
        rewards = []
        for i, e in enumerate(entries):
            r = e.get("reward", e.get("mean_reward",
                e.get("reward/mean", e.get("rewards/mean", None))))
            if r is not None:
                rewards.append((e.get("step", i), r))

        # Completion lengths
        lengths = []
        for i, e in enumerate(entries):
            l = e.get("mean_completion_length",
                e.get("completion_length/mean", None))
            if l is not None:
                lengths.append((e.get("step", i), l))

        # Status metrics
        col1, col2, col3, col4 = st.columns(4)

        if losses:
            col1.metric("Loss", f"{losses[-1][1]:.4f}")
        if rewards:
            first_r = rewards[0][1]
            latest_r = rewards[-1][1]
            col2.metric("Mean Reward", f"{latest_r:.3f}",
                        delta=f"{latest_r - first_r:+.3f}")

        current_step = entries[-1].get("step", len(entries))
        col3.metric("Step", current_step)

        if rewards:
            # Estimate accuracy from reward (reward ~1.0 = correct)
            est_acc = latest_r / 1.2 * 100  # max reward is 1.2
            col4.metric("Est. Accuracy", f"~{est_acc:.0f}%")

        # Charts
        import pandas as pd

        if losses and rewards:
            st.subheader("Loss & Reward")
            c1, c2 = st.columns(2)

            with c1:
                df_loss = pd.DataFrame(losses, columns=["Step", "Loss"])
                st.line_chart(df_loss.set_index("Step"))

            with c2:
                df_reward = pd.DataFrame(rewards, columns=["Step", "Mean Reward"])
                st.line_chart(df_reward.set_index("Step"))

        elif losses:
            st.subheader("Loss")
            df_loss = pd.DataFrame(losses, columns=["Step", "Loss"])
            st.line_chart(df_loss.set_index("Step"))

        # Completion length chart
        if lengths:
            st.subheader("Average Completion Length (tokens)")
            df_len = pd.DataFrame(lengths, columns=["Step", "Tokens"])
            st.line_chart(df_len.set_index("Step"))
            st.caption(
                "Increasing length = model learning to produce longer reasoning. "
                "Decreasing = model getting more concise. "
                "Spike then decrease = model tried verbose outputs, learned they don't help."
            )

        # Recent entries
        with st.expander("📋 Recent Log Entries"):
            recent = entries[-20:]
            st.json(recent)


# =====================================================================
# RESULTS COMPARISON (if evaluations are done)
# =====================================================================
st.markdown("---")
st.subheader("📊 Evaluation Results")

results = {}
for name, path in [("Zero-Shot", "outputs/zeroshot_results.json"),
                    ("SFT", "outputs/sft_results.json"),
                    ("GRPO", "outputs/grpo_results.json")]:
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
            results[name] = data["accuracy"]
        except (json.JSONDecodeError, KeyError):
            pass

if results:
    cols = st.columns(len(results))
    for col, (name, acc) in zip(cols, results.items()):
        col.metric(name, f"{acc*100:.1f}%")

    # Bar chart
    import pandas as pd
    df = pd.DataFrame({"Model": list(results.keys()),
                        "Accuracy (%)": [v*100 for v in results.values()]})
    st.bar_chart(df.set_index("Model"))
else:
    st.info("No evaluation results yet. Run the eval scripts first.")


# =====================================================================
# AUTO-REFRESH
# =====================================================================
st.markdown("---")
auto_refresh = st.checkbox("🔄 Auto-refresh (every 5 seconds)", value=True)
if auto_refresh:
    time.sleep(REFRESH_INTERVAL)
    st.rerun()
