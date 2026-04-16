"""
FinReason — Interactive Demo with Multi-Scale Comparison
Run: streamlit run ui/app.py
"""
import streamlit as st
import torch, json, os, re, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from shared_utils import extract_final_answer, check_answer, reward_function, format_prompt

st.set_page_config(page_title="FinReason", page_icon="📊", layout="wide")
st.title("📊 FinReason")
st.markdown("**Financial Numerical Reasoning via GRPO — Multi-Scale Study**")

# =====================================================================
# SIDEBAR — Model Scale + Checkpoint Selection
# =====================================================================
st.sidebar.header("1. Model Scale")
model_scale = st.sidebar.radio(
    "Select model size:",
    ["1.5B (Qwen2.5-1.5B)", "3B (Qwen2.5-3B)", "7B (Qwen2.5-7B)"],
    index=1,
)

SCALE_MAP = {
    "1.5B (Qwen2.5-1.5B)": {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "tag": "1.5B",
        "sft": "checkpoints/1.5B/sft/final_adapter",
        "grpo": "checkpoints/1.5B/grpo/final_adapter",
        "results_dir": "outputs/1.5B",
    },
    "3B (Qwen2.5-3B)": {
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "tag": "3B",
        "sft": "checkpoints/3B/sft/final_adapter",
        "grpo": "checkpoints/3B/grpo/final_adapter",
        "results_dir": "outputs/3B",
    },
    "7B (Qwen2.5-7B)": {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "tag": "7B",
        "sft": "checkpoints/7B/sft/final_adapter",
        "grpo": "checkpoints/7B/grpo/final_adapter",
        "results_dir": "outputs/7B",
    },
}

scale_cfg = SCALE_MAP[model_scale]
MODEL_NAME = scale_cfg["model"]

st.sidebar.header("2. Training Stage")
stage_choice = st.sidebar.radio(
    "Select checkpoint:",
    ["Zero-Shot (base)", "SFT", "SFT + GRPO"],
    index=2,
)

adapter_map = {
    "Zero-Shot (base)": None,
    "SFT": scale_cfg["sft"],
    "SFT + GRPO": scale_cfg["grpo"],
}

# Also support old flat checkpoint paths (backward compatible)
FLAT_ADAPTERS = {
    "SFT": "checkpoints/sft/final_adapter",
    "SFT + GRPO": "checkpoints/grpo/final_adapter",
}

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Active:** {scale_cfg['tag']} / {stage_choice}")

# =====================================================================
# MODEL LOADING
# =====================================================================
@st.cache_resource
def load_model(base, adapter):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(base)
    mdl = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32)
    if adapter:
        adapter_path = None
        if os.path.exists(adapter):
            adapter_path = adapter
        # Fallback: check flat checkpoint path
        elif stage_choice in FLAT_ADAPTERS and os.path.exists(FLAT_ADAPTERS[stage_choice]):
            adapter_path = FLAT_ADAPTERS[stage_choice]

        if adapter_path:
            from peft import PeftModel
            mdl = PeftModel.from_pretrained(mdl, adapter_path)
            st.sidebar.success(f"Loaded: {adapter_path}")
        else:
            st.sidebar.warning(f"Adapter not found. Running base model.")
    mdl.eval()
    return mdl, tok

# =====================================================================
# MAIN AREA — Input + Output
# =====================================================================
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("Financial Data")
    input_method = st.radio("Input method:", ["Paste text", "Upload PDF"], horizontal=True)

    if input_method == "Upload PDF":
        uploaded = st.file_uploader("Upload a financial PDF", type=["pdf"])
        if uploaded:
            try:
                from pdf_extractor import extract_report_for_qa
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                    f.write(uploaded.read())
                    tmp_path = f.name
                result = extract_report_for_qa(tmp_path, max_pages=15)
                st.info(f"Detected: {result['currency']['currency']} | {result['table_pages']} table pages")
                if result["ready_for_model"]:
                    context = result["ready_for_model"][0]["text"]
                else:
                    context = ""
                    st.warning("No table pages found")
                os.unlink(tmp_path)
            except ImportError:
                st.error("Install: pip install pdfplumber pdf2image pytesseract")
                context = ""
        else:
            context = ""
    else:
        context = st.text_area("Paste financial table/text:", height=250, value=(
            "| | 2019 | 2018 | 2017 |\n"
            "| Revenue | $1,452.4 | $1,146.2 | $1,036.9 |\n"
            "| Operating Income | $312.5 | $287.1 | $245.8 |\n"
            "| Net Income | $198.3 | $172.4 | $156.2 |\n\n"
            "The company reported strong revenue growth driven by "
            "increased demand in the North American market."))

    question = st.text_input("Question:",
        value="What was the year-over-year change in revenue from 2018 to 2019?")
    ground_truth = st.text_input("Ground truth (optional):", value="306.2")

with col2:
    st.subheader("Model Output")
    st.caption(f"Using: **{scale_cfg['tag']}** / {stage_choice}")

    if st.button("🔍 Ask FinReason", type="primary", use_container_width=True):
        if not context.strip() or not question.strip():
            st.error("Enter both financial data and a question.")
        else:
            with st.spinner(f"Running {scale_cfg['tag']} {stage_choice}..."):
                model, tokenizer = load_model(MODEL_NAME, adapter_map[stage_choice])
                prompt = format_prompt(context, question, mode="grpo_eval")
                ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                                max_length=1024).to(model.device)
                with torch.no_grad():
                    out = model.generate(**ids, max_new_tokens=256, do_sample=False,
                                         pad_token_id=tokenizer.eos_token_id)
                raw = tokenizer.decode(out[0][ids["input_ids"].shape[1]:],
                                       skip_special_tokens=True).strip()

            # Display results
            final = extract_final_answer(raw)
            st.markdown(f"**Answer:** `{final}`")

            # Show reasoning if present
            think_match = re.search(r'<think>(.*?)</think>', raw, re.DOTALL)
            if think_match:
                st.info(f"**Reasoning:**\n{think_match.group(1).strip()}")

            # Check against ground truth
            if ground_truth.strip():
                ok = check_answer(final, ground_truth)
                r = reward_function(raw, ground_truth)
                if ok:
                    st.success(f"✓ Correct (R={r:.1f})")
                else:
                    st.error(f"✗ Wrong (R={r:.1f})")

            # Full output
            st.subheader("Full Model Output")
            st.code(raw, language=None)

# =====================================================================
# RESULTS — Per-Scale Dashboard
# =====================================================================
st.markdown("---")
st.subheader(f"📈 Results — {scale_cfg['tag']}")

# Try scale-specific results first, then flat results
results = {}
results_dir = scale_cfg["results_dir"]
for name, filename in [("Zero-Shot", "zeroshot_results.json"),
                         ("SFT", "sft_results.json"),
                         ("GRPO", "grpo_results.json")]:
    # Check scale-specific path
    path = os.path.join(results_dir, filename)
    if not os.path.exists(path):
        # Fallback to flat path
        path = os.path.join("outputs", filename)
    if os.path.exists(path):
        try:
            results[name] = json.load(open(path))["accuracy"]
        except (json.JSONDecodeError, KeyError):
            pass

if results:
    cols = st.columns(len(results))
    for col, (n, a) in zip(cols, results.items()):
        col.metric(n, f"{a*100:.1f}%")

# Show figures (scale-specific or flat)
for fig_name in ["fig1_accuracy.png", "fig2_grpo_curves.png",
                  "fig3_error_analysis.png", "fig4_think_analysis.png"]:
    fig_path = os.path.join("results", scale_cfg["tag"], fig_name)
    if not os.path.exists(fig_path):
        fig_path = os.path.join("results", fig_name)
    if os.path.exists(fig_path):
        st.image(fig_path)

# =====================================================================
# CROSS-SCALE COMPARISON
# =====================================================================
st.markdown("---")
st.subheader("🔬 Cross-Scale Comparison")
st.markdown("*How does model scale affect GRPO for financial reasoning?*")

comparison_data = {}
for scale_name, cfg in SCALE_MAP.items():
    tag = cfg["tag"]
    scale_results = {}
    for stage_name, filename in [("Zero-Shot", "zeroshot_results.json"),
                                   ("SFT", "sft_results.json"),
                                   ("GRPO", "grpo_results.json")]:
        # Check scale-specific path
        path = os.path.join(cfg["results_dir"], filename)
        if not os.path.exists(path):
            path = os.path.join("outputs", filename) if tag == "1.5B" else ""
        if os.path.exists(path):
            try:
                scale_results[stage_name] = json.load(open(path))["accuracy"] * 100
            except (json.JSONDecodeError, KeyError):
                pass
    if scale_results:
        comparison_data[tag] = scale_results

if comparison_data:
    # Build comparison table
    st.markdown("### Execution Accuracy by Scale and Training Stage")

    # Table header
    header = "| Scale |"
    separator = "|---|"
    for stage in ["Zero-Shot", "SFT", "GRPO", "GRPO - SFT (Δ)"]:
        header += f" {stage} |"
        separator += "---|"

    rows = []
    for tag in ["1.5B", "3B", "7B"]:
        if tag not in comparison_data:
            continue
        d = comparison_data[tag]
        zs = d.get("Zero-Shot", "-")
        sft = d.get("SFT", "-")
        grpo = d.get("GRPO", "-")
        if isinstance(sft, float) and isinstance(grpo, float):
            delta = f"{grpo - sft:+.1f}%"
        else:
            delta = "-"
        zs_str = f"{zs:.1f}%" if isinstance(zs, float) else zs
        sft_str = f"{sft:.1f}%" if isinstance(sft, float) else sft
        grpo_str = f"{grpo:.1f}%" if isinstance(grpo, float) else grpo
        rows.append(f"| **{tag}** | {zs_str} | {sft_str} | {grpo_str} | {delta} |")

    table = header + "\n" + separator + "\n" + "\n".join(rows)
    st.markdown(table)

    # Bar chart comparison
    import pandas as pd
    chart_data = []
    for tag, stages in comparison_data.items():
        for stage, acc in stages.items():
            chart_data.append({"Scale": tag, "Stage": stage, "Accuracy (%)": acc})

    if chart_data:
        df = pd.DataFrame(chart_data)
        # Pivot for grouped display
        pivot = df.pivot(index="Scale", columns="Stage", values="Accuracy (%)")
        # Reorder
        col_order = [c for c in ["Zero-Shot", "SFT", "GRPO"] if c in pivot.columns]
        pivot = pivot[col_order]
        row_order = [r for r in ["1.5B", "3B", "7B"] if r in pivot.index]
        pivot = pivot.reindex(row_order)
        st.bar_chart(pivot)

    st.markdown("""
    **Key Research Question:** *Does GRPO improve over SFT at each scale?*

    Look at the **GRPO - SFT (Δ)** column:
    - **Positive Δ** = GRPO helped at this scale
    - **Negative Δ** = GRPO didn't help (reward too sparse)
    - The scale where Δ turns positive is the **minimum viable scale for GRPO**
    """)
else:
    st.info("No results found yet. Run the training pipeline for each model scale.")
    st.markdown("""
    **Expected folder structure for multi-scale results:**
    ```
    outputs/
    ├── 1.5B/
    │   ├── zeroshot_results.json
    │   ├── sft_results.json
    │   └── grpo_results.json
    ├── 3B/
    │   ├── zeroshot_results.json
    │   ├── sft_results.json
    │   └── grpo_results.json
    └── 7B/
        ├── zeroshot_results.json
        ├── sft_results.json
        └── grpo_results.json
    ```
    """)

# =====================================================================
# FOOTER
# =====================================================================
st.markdown("---")
st.caption("FinReason — EGN 6217 Applied Deep Learning, Spring 2026 | Om Sanjaykumar Patel")
