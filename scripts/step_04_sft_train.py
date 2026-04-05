"""
Step 4: Stage 1 — Supervised Fine-Tuning with QLoRA.

Teaches the model to read financial tables and produce short answers.
Time: ~1-2 hours on RTX 4060.
"""
import torch, json, os
from datasets import Dataset

# ============================================================
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "checkpoints/sft"
NUM_EPOCHS = 3
LR = 2e-4
MAX_SEQ_LEN = 1024
LORA_R = 16
LORA_ALPHA = 32
BATCH = 1
GRAD_ACCUM = 8
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open("data/train_sft.json") as f:
    train_data = json.load(f)
with open("data/val_sft.json") as f:
    val_data = json.load(f)
print(f"Train: {len(train_data)} | Val: {len(val_data)}")

# --- Try Unsloth first ---
try:
    from unsloth import FastLanguageModel
    print("Using Unsloth ✓")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME, max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True, dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
        bias="none", use_gradient_checkpointing="unsloth",
    )
except ImportError:
    print("Unsloth not found. Using HuggingFace PEFT.")
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True),
        device_map="auto", trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
        bias="none", task_type="CAUSAL_LM",
    ))

model.print_trainable_parameters()
print(f"VRAM: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")

# Build HF datasets
train_ds = Dataset.from_list([{"text": d["text"]} for d in train_data])
val_ds = Dataset.from_list([{"text": d["text"]} for d in val_data])

from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model, tokenizer=tokenizer,
    train_dataset=train_ds, eval_dataset=val_ds,
    args=SFTConfig(
        output_dir=OUTPUT_DIR, num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR, lr_scheduler_type="cosine", warmup_ratio=0.05,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=20, save_steps=200,
        eval_strategy="steps", eval_steps=200,
        save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        max_seq_length=MAX_SEQ_LEN, dataset_text_field="text",
        report_to="none", dataloader_pin_memory=False,
    ),
)

print(f"\n{'='*50}")
print(f"  STARTING SFT — ~1-2 hours on RTX 4060")
print(f"{'='*50}\n")

trainer.train()

model.save_pretrained(os.path.join(OUTPUT_DIR, "final_adapter"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_adapter"))

logs = [l for l in trainer.state.log_history if "loss" in l or "eval_loss" in l]
with open(os.path.join(OUTPUT_DIR, "training_log.json"), "w") as f:
    json.dump(logs, f, indent=2)

print(f"\n✓ SFT complete. Adapter → {OUTPUT_DIR}/final_adapter")
