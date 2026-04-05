"""Step 6: Evaluate SFT model (Checkpoint 2 of 3)."""
import torch, json, os, sys
from tqdm import tqdm
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import extract_qa, extract_context, build_prompt
from reward_utils import extract_final_answer, check_answer, compute_execution_accuracy

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER = "checkpoints/sft/final_adapter"
NUM_TEST = 300
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model
try:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=ADAPTER, max_seq_length=1024, load_in_4bit=True)
    FastLanguageModel.for_inference(model)
    print("Loaded with Unsloth ✓")
except ImportError:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True),
        device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(base, ADAPTER)
    model.eval()
    print("Loaded with PEFT ✓")

dataset = load_dataset("ibm-research/finqa", trust_remote_code=True)
test_data = dataset.get("test", dataset.get("validation"))
if NUM_TEST and NUM_TEST < len(test_data):
    test_data = test_data.select(range(NUM_TEST))

predictions, ground_truths, results_log = [], [], []

for i in tqdm(range(len(test_data)), desc="SFT eval"):
    q, gt = extract_qa(test_data[i])
    ctx = extract_context(test_data[i])
    prompt = build_prompt(ctx, q)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=1024).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    pred = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()
    final = extract_final_answer(pred)
    correct = check_answer(final, gt)
    predictions.append(pred); ground_truths.append(gt)
    results_log.append({"i": i, "q": q, "gt": gt, "pred": final, "ok": correct})

acc = compute_execution_accuracy(predictions, ground_truths)
print(f"\n{'='*50}")
print(f"SFT ACCURACY: {acc:.4f} ({acc*100:.2f}%)")

zs_path = f"{OUTPUT_DIR}/zeroshot_results.json"
if os.path.exists(zs_path):
    zs_acc = json.load(open(zs_path))["accuracy"]
    print(f"Zero-shot was: {zs_acc:.4f} → Improvement: +{(acc-zs_acc)*100:.2f}%")
print(f"{'='*50}")

with open(f"{OUTPUT_DIR}/sft_results.json", "w") as f:
    json.dump({"model": MODEL_NAME, "adapter": ADAPTER, "accuracy": acc,
               "n": len(predictions), "results": results_log}, f, indent=2)
