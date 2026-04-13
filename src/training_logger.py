"""
training_logger.py — Custom callback that writes live training logs.

Add this to SFT and GRPO trainers to enable the live dashboard.

Usage in sft_train.py:
    from training_logger import LiveLogCallback
    trainer = SFTTrainer(..., callbacks=[LiveLogCallback("sft")])

Usage in grpo_train.py:
    from training_logger import LiveLogCallback
    trainer = GRPOTrainer(..., callbacks=[LiveLogCallback("grpo")])
"""
import json
import os
import time
from transformers import TrainerCallback


class LiveLogCallback(TrainerCallback):
    """
    Writes training metrics to a JSON file after every log step.
    The Streamlit training_monitor.py reads this file for live updates.
    """

    def __init__(self, stage: str = "sft"):
        """
        Args:
            stage: "sft" or "grpo" — determines which log file to write
        """
        self.stage = stage
        self.log_file = f"outputs/live_{stage}_log.json"
        self.entries = []
        self.start_time = None

        os.makedirs("outputs", exist_ok=True)

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        self.entries = []
        self._write_status("running")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        entry = {
            "step": state.global_step,
            "epoch": round(state.epoch, 2) if state.epoch else 0,
            "timestamp": time.time(),
            "elapsed_sec": round(time.time() - self.start_time, 1) if self.start_time else 0,
        }

        # Copy all logged metrics
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                entry[key] = round(value, 6) if isinstance(value, float) else value

        # Add estimated time remaining
        if state.max_steps and state.global_step > 0:
            elapsed = time.time() - self.start_time
            steps_done = state.global_step
            steps_left = state.max_steps - steps_done
            sec_per_step = elapsed / steps_done
            eta_sec = steps_left * sec_per_step
            entry["eta_minutes"] = round(eta_sec / 60, 1)
            entry["progress_pct"] = round(100 * steps_done / state.max_steps, 1)

        self.entries.append(entry)
        self._save()

    def on_train_end(self, args, state, control, **kwargs):
        self._write_status("completed")

    def _save(self):
        """Write log entries to disk."""
        try:
            with open(self.log_file, "w") as f:
                json.dump({
                    "stage": self.stage,
                    "status": "running",
                    "num_entries": len(self.entries),
                    "entries": self.entries,
                }, f, indent=2)
        except IOError:
            pass

    def _write_status(self, status):
        """Write a status marker."""
        try:
            with open(self.log_file, "w") as f:
                json.dump({
                    "stage": self.stage,
                    "status": status,
                    "num_entries": len(self.entries),
                    "entries": self.entries,
                }, f, indent=2)
        except IOError:
            pass
