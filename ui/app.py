"""
FinReason — Interactive Demo
Run: streamlit run ui/app.py
"""
import streamlit as st
import torch, json, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from shared_utils import extract_final_answer, check_answer, reward_function, format_prompt

st.set_page_config(page_title="FinReason", page_icon="📊", layout="wide")
st.title("📊 FinReason")
st.markdown("**Financial Numerical Reasoning via GRPO with Verifiable Rewards**")

# Sidebar
st.sidebar.header("Model Checkpoint")
model_choice = st.sidebar.radio("Select:", ["Zero-Shot (base)", "SFT", "SFT + GRPO (final)"], index=2)
adapter_map = {"Zero-Shot (base)": None, "SFT": "checkpoints/sft/final_adapter", "SFT + GRPO (final)": "checkpoints/grpo/final_adapter"}
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

@st.cache_resource
def load_model(base, adapter):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(base)
    mdl = AutoModelForCausalLM.from_pretrained(base,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True),
        device_map="auto")
    if adapter and os.path.exists(adapter):
        from peft import PeftModel
        mdl = PeftModel.from_pretrained(mdl, adapter)
        st.sidebar.success(f"Loaded: {adapter}")
    elif adapter:
        st.sidebar.warning(f"Not found: {adapter}")
    mdl.eval()
    return mdl, tok

# Main area
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
                    f.write(uploaded.read()); tmp_path = f.name
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
        context = st.text_area("Paste financial table/text:", height=250, value=
            "| | 2024 RM'000 | 2023 RM'000 |\n| Demand deposits | 12,825,346 | 14,082,264 |\n"
            "| Fixed deposits | 183,660 | 104,226 |\n| Total | 13,009,006 | 14,186,490 |")

    question = st.text_input("Question:", value="What was the change in demand deposits from 2023 to 2024?")
    ground_truth = st.text_input("Ground truth (optional):", value="-1256918")

with col2:
    st.subheader("Model Output")
    if st.button("🔍 Ask FinReason", type="primary", use_container_width=True):
        if not context.strip() or not question.strip():
            st.error("Enter both financial data and a question.")
        else:
            with st.spinner(f"Running {model_choice}..."):
                model, tokenizer = load_model(MODEL_NAME, adapter_map[model_choice])
                prompt = format_prompt(context, question)
                ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
                with torch.no_grad():
                    out = model.generate(**ids, max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.eos_token_id)
                raw = tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            final = extract_final_answer(raw)
            st.markdown(f"**Answer:** `{final}`")
            if re.search(r'<think>(.*?)</think>', raw, re.DOTALL):
                st.info(f"**Reasoning:**\n{re.search(r'<think>(.*?)</think>', raw, re.DOTALL).group(1).strip()}")
            if ground_truth.strip():
                ok = check_answer(final, ground_truth)
                r = reward_function(raw, ground_truth)
                (st.success if ok else st.error)(f"{'✓ Correct' if ok else '✗ Wrong'} (R={r:.1f})")
            with st.expander("Raw output"): st.code(raw)

# Results dashboard
st.markdown("---")
st.subheader("📈 Results")
results = {}
for name, path in [("Zero-Shot","outputs/zeroshot_results.json"),("SFT","outputs/sft_results.json"),("GRPO","outputs/grpo_results.json")]:
    if os.path.exists(path):
        results[name] = json.load(open(path))["accuracy"]
if results:
    cols = st.columns(len(results))
    for col,(n,a) in zip(cols, results.items()): col.metric(n, f"{a*100:.1f}%")
    for fig in ["results/fig1_accuracy.png","results/fig2_grpo_curves.png","results/fig3_error_analysis.png","results/fig4_think_analysis.png"]:
        if os.path.exists(fig): st.image(fig)
