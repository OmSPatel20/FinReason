"""
FinReason — Streamlit Demo App

Run: streamlit run scripts/app.py

Shows:
  - Upload or paste financial data
  - Ask a question
  - See the model's reasoning and answer
  - Compare base vs SFT vs GRPO outputs
"""
import streamlit as st
import torch
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from shared_utils import (extract_final_answer, check_answer, reward_function,
                           format_prompt, SYSTEM_MSG)

st.set_page_config(page_title="FinReason", page_icon="📊", layout="wide")

st.title("📊 FinReason")
st.markdown("**Teaching a Small LLM Financial Reasoning via GRPO**")

# --- Sidebar: Model selection ---
st.sidebar.header("Model")
model_choice = st.sidebar.radio(
    "Select checkpoint:",
    ["Zero-Shot (base)", "SFT", "SFT + GRPO (final)"],
    index=2,
)

adapter_map = {
    "Zero-Shot (base)": None,
    "SFT": "checkpoints/sft/final_adapter",
    "SFT + GRPO (final)": "checkpoints/grpo/final_adapter",
}

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
adapter_path = adapter_map[model_choice]


@st.cache_resource
def load_model(base_name, adapter):
    """Load model once and cache it."""
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_name, quantization_config=bnb,
        device_map="auto", trust_remote_code=True,
    )

    if adapter and os.path.exists(adapter):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        st.sidebar.success(f"Loaded adapter: {adapter}")
    elif adapter:
        st.sidebar.warning(f"Adapter not found: {adapter}")

    model.eval()
    return model, tokenizer


# --- Main area ---
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("Financial Data")

    # Sample data for quick testing
    sample_context = """| | 2019 | 2018 | 2017 |
| Revenue | $1,452.4 | $1,146.2 | $1,036.9 |
| Operating Income | $312.5 | $287.1 | $245.8 |
| Net Income | $198.3 | $172.4 | $156.2 |
| Total Assets | $3,421.7 | $3,102.5 | $2,876.3 |

The company reported strong revenue growth driven by increased demand
in the North American market. Operating margins improved by 150 basis
points year-over-year."""

    context = st.text_area(
        "Paste financial table/text here:",
        value=sample_context,
        height=250,
    )

    question = st.text_input(
        "Question:",
        value="What was the year-over-year change in revenue from 2018 to 2019?",
    )

    ground_truth = st.text_input(
        "Ground truth answer (optional, for checking):",
        value="306.2",
    )

with col2:
    st.subheader("Model Output")

    if st.button("🔍 Ask FinReason", type="primary", use_container_width=True):
        if not context.strip() or not question.strip():
            st.error("Please enter both financial data and a question.")
        else:
            with st.spinner(f"Running {model_choice}..."):
                model, tokenizer = load_model(MODEL_NAME, adapter_path)
                prompt = format_prompt(context, question, answer=None)

                ids = tokenizer(
                    prompt, return_tensors="pt",
                    truncation=True, max_length=1024,
                ).to(model.device)

                with torch.no_grad():
                    out = model.generate(
                        **ids,
                        max_new_tokens=256,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                raw = tokenizer.decode(
                    out[0][ids["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                ).strip()

            # Display output
            final = extract_final_answer(raw)
            has_think = bool(re.search(r'<think>.*?</think>', raw, re.DOTALL))

            st.markdown(f"**Extracted Answer:** `{final}`")

            if has_think:
                think_match = re.search(r'<think>(.*?)</think>', raw, re.DOTALL)
                if think_match:
                    st.info(f"**Reasoning:**\n{think_match.group(1).strip()}")

            # Check against ground truth
            if ground_truth.strip():
                correct = check_answer(final, ground_truth)
                r = reward_function(raw, ground_truth)
                if correct:
                    st.success(f"✓ Correct! (Reward: {r:.1f})")
                else:
                    st.error(f"✗ Wrong. Expected: {ground_truth} (Reward: {r:.1f})")

            with st.expander("Full raw output"):
                st.code(raw, language=None)

# --- Results comparison ---
st.markdown("---")
st.subheader("📈 Results Comparison")

results = {}
for name, path in [("Zero-Shot", "outputs/zeroshot_results.json"),
                    ("SFT", "outputs/sft_results.json"),
                    ("SFT + GRPO", "outputs/grpo_results.json")]:
    if os.path.exists(path):
        with open(path) as f:
            results[name] = json.load(f)

if results:
    cols = st.columns(len(results))
    for col, (name, data) in zip(cols, results.items()):
        acc = data["accuracy"] * 100
        col.metric(name, f"{acc:.1f}%")

    # Show figures if they exist
    fig_col1, fig_col2 = st.columns(2)
    if os.path.exists("outputs/fig1_accuracy.png"):
        fig_col1.image("outputs/fig1_accuracy.png", caption="Accuracy Comparison")
    if os.path.exists("outputs/fig2_grpo_curves.png"):
        fig_col2.image("outputs/fig2_grpo_curves.png", caption="GRPO Training Curves")

    fig_col3, fig_col4 = st.columns(2)
    if os.path.exists("outputs/fig3_error_analysis.png"):
        fig_col3.image("outputs/fig3_error_analysis.png", caption="Error Analysis")
    if os.path.exists("outputs/fig4_think_analysis.png"):
        fig_col4.image("outputs/fig4_think_analysis.png", caption="Reasoning Analysis")
else:
    st.info("No results yet. Run the training pipeline first.")

st.markdown("---")
st.caption("FinReason — EGN 6217 Applied Deep Learning, Spring 2026")
