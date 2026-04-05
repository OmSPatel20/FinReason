"""
FinReason — Streamlit Demo App
Run: streamlit run app.py
"""
import streamlit as st
import torch
import json
import os
import re
import sys

sys.path.insert(0, "scripts")
from reward_utils import extract_final_answer, check_answer
from data_utils import build_prompt

st.set_page_config(page_title="FinReason", page_icon="📊", layout="wide")

st.title("📊 FinReason")
st.markdown("**Financial Numerical Reasoning via GRPO with Verifiable Rewards**")
st.markdown("---")

# ---- Sidebar: Model selection ----
st.sidebar.header("Model Selection")
model_choice = st.sidebar.radio(
    "Which checkpoint to use?",
    ["GRPO (SFT + RL)", "SFT Only", "Base (Zero-shot)"],
    index=0,
)

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
adapter_map = {
    "GRPO (SFT + RL)": "checkpoints/grpo/final_adapter",
    "SFT Only": "checkpoints/sft/final_adapter",
    "Base (Zero-shot)": None,
}
adapter_path = adapter_map[model_choice]

# ---- Load model (cached) ----
@st.cache_resource
def load_model(model_name, adapter):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb, device_map="auto", trust_remote_code=True)

    if adapter and os.path.exists(adapter):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        st.sidebar.success(f"Loaded adapter: {adapter}")
    elif adapter:
        st.sidebar.warning(f"Adapter not found: {adapter}. Using base model.")

    model.eval()
    return model, tokenizer


with st.spinner("Loading model... (first time takes ~30s)"):
    model, tokenizer = load_model(MODEL_NAME, adapter_path)

# ---- Main interface ----
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Financial Context")
    default_context = """| | 2019 | 2018 | 2017 |
|---|---|---|---|
| Revenue | $1,452.4 | $1,146.2 | $1,036.9 |
| Operating Income | $312.5 | $287.1 | $251.3 |
| Net Income | $198.7 | $176.4 | $155.2 |

The company reported strong revenue growth driven by increased demand
in the enterprise segment. Operating margins expanded by 150 basis points
year-over-year."""

    context = st.text_area("Paste financial data (table + text):", value=default_context,
                           height=250)

    default_question = "What was the year-over-year change in revenue from 2018 to 2019?"
    question = st.text_input("Question:", value=default_question)

    ground_truth = st.text_input("Ground truth answer (optional, for checking):", value="306.2")

    run_btn = st.button("🔍 Ask FinReason", type="primary", use_container_width=True)

with col2:
    st.subheader("Model Output")

    if run_btn and context and question:
        prompt = build_prompt(context, question)

        with st.spinner("Generating answer..."):
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                               max_length=1024).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=256, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
            raw_output = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                          skip_special_tokens=True).strip()

        # Display raw output
        st.markdown("**Raw Model Output:**")
        # Highlight <think> blocks
        if "<think>" in raw_output:
            think_match = re.search(r'<think>(.*?)</think>', raw_output, re.DOTALL)
            if think_match:
                st.info(f"💭 **Reasoning:** {think_match.group(1).strip()}")
            after = re.sub(r'<think>.*?</think>', '', raw_output, flags=re.DOTALL).strip()
            st.success(f"📊 **Answer:** {after}")
        else:
            st.success(f"📊 **Answer:** {raw_output}")

        # Extract and check
        final = extract_final_answer(raw_output)
        st.markdown(f"**Extracted answer:** `{final}`")

        if ground_truth.strip():
            correct = check_answer(final, ground_truth)
            if correct:
                st.balloons()
                st.success(f"✅ Correct! (Ground truth: {ground_truth})")
            else:
                st.error(f"❌ Wrong. Ground truth: {ground_truth}, Got: {final}")

    elif run_btn:
        st.warning("Please enter both context and question.")

# ---- Results dashboard ----
st.markdown("---")
st.subheader("📈 Results Dashboard")

results_files = {
    "Zero-Shot": "outputs/zeroshot_results.json",
    "SFT": "outputs/sft_results.json",
    "GRPO": "outputs/grpo_results.json",
}

accs = {}
for name, path in results_files.items():
    if os.path.exists(path):
        with open(path) as f:
            accs[name] = json.load(f)["accuracy"]

if accs:
    cols = st.columns(len(accs))
    for i, (name, acc) in enumerate(accs.items()):
        with cols[i]:
            st.metric(name, f"{acc*100:.1f}%",
                      delta=f"+{(acc - list(accs.values())[0])*100:.1f}%" if i > 0 else None)

    # Show figures if they exist
    fig_col1, fig_col2 = st.columns(2)
    if os.path.exists("outputs/fig1_accuracy.png"):
        with fig_col1:
            st.image("outputs/fig1_accuracy.png", caption="Accuracy Comparison")
    if os.path.exists("outputs/fig4_think_analysis.png"):
        with fig_col2:
            st.image("outputs/fig4_think_analysis.png", caption="Reasoning Analysis")
else:
    st.info("No results yet. Run the training pipeline first (Steps 3-8).")

# ---- Footer ----
st.markdown("---")
st.caption("FinReason — EGN 6217 Applied Deep Learning, Spring 2026 — University of Florida")
