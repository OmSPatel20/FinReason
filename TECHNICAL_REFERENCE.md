# FinReason — Complete Technical Reference

**What every file does, what every function does, what every library does, and why.**

Read this before your professor asks you anything.

---

## Table of Contents

1. [Project Overview — The Big Picture](#1-project-overview)
2. [File-by-File Breakdown](#2-file-by-file-breakdown)
3. [Libraries Used and Why](#3-libraries-used-and-why)
4. [Key Concepts You Must Understand](#4-key-concepts-you-must-understand)
5. [Data Flow — What Happens When You Run Each Step](#5-data-flow)
6. [What to Say When Your Professor Asks](#6-what-to-say)

---

## 1. Project Overview

### What the project does in one sentence

Takes a small language model (Qwen2.5-1.5B), teaches it to answer numerical questions over financial reports using supervised fine-tuning, then improves it further using reinforcement learning (GRPO) where the reward is simply "did you get the right number."

### The two-stage pipeline

```
Stage 1 (SFT):   FinQA data → format as chat → QLoRA fine-tune → model learns table reading + answer format
Stage 2 (GRPO):  Same data → model generates 4 answers per question → reward function scores each →
                  correct answers get reinforced, wrong ones get suppressed → model learns to reason
```

### The three checkpoints you compare

```
Checkpoint 1:  Base model zero-shot (no training at all)          → probably ~5-10% accuracy
Checkpoint 2:  After SFT (supervised fine-tuning)                 → probably ~20-35% accuracy  
Checkpoint 3:  After SFT + GRPO (reinforcement learning)          → hopefully ~30-45% accuracy
```

The whole point of the project is measuring whether Stage 2 (GRPO) adds value on top of Stage 1 (SFT).

---

## 2. File-by-File Breakdown

### `requirements.txt`

```
torch>=2.1.0              # PyTorch — the deep learning framework everything runs on
torchvision>=0.16.0       # Image utilities for PyTorch (not heavily used here)
transformers>=4.46.0      # HuggingFace Transformers — loads and runs LLMs
trl>=0.12.0               # Transformer Reinforcement Learning — provides SFTTrainer and GRPOTrainer
peft>=0.13.0              # Parameter-Efficient Fine-Tuning — implements LoRA/QLoRA
bitsandbytes>=0.44.0      # 4-bit quantization — makes 1.5B model fit in 8GB VRAM
accelerate>=1.0.0         # HuggingFace Accelerate — handles GPU placement automatically
datasets                  # HuggingFace Datasets — downloads FinQA with one line
matplotlib                # Plotting — generates the 4 figures for your report
seaborn                   # Pretty plots on top of matplotlib
pandas                    # Data manipulation (used lightly)
numpy                     # Numerical operations
tqdm                      # Progress bars during evaluation loops
streamlit                 # Web UI framework — builds the interactive demo
pdfplumber                # Extracts text from text-based PDFs
pdf2image                 # Converts PDF pages to images (for OCR)
pytesseract               # OCR — reads text from scanned PDF images
```

---

### `src/shared_utils.py` — THE MOST IMPORTANT FILE

This file is imported by every other script. It contains:

#### `extract_qa(example)` — Data extraction
```python
# FinQA on HuggingFace has different format versions.
# Some have example["question"] and example["answer"] directly.
# Others nest it inside example["qa"]["question"] and example["qa"]["exe_ans"].
# This function handles both so every other script doesn't need to care.
```

#### `extract_context(example)` — Financial table extraction
```python
# Pulls out the financial data (table + surrounding text) from a FinQA example.
# Tables come as 2D arrays like [["", "2019", "2018"], ["Revenue", "1452", "1146"]].
# This function flattens them into pipe-delimited text: " | 2019 | 2018"
# Why pipes? Because the LLM needs text, not arrays. Pipes preserve column alignment.
```

#### `format_prompt(context, question, answer)` — Chat template
```python
# Formats everything into Qwen's chat template:
#   <|im_start|>system
#   You are a financial analyst expert...
#   <|im_end|>
#   <|im_start|>user
#   Financial Data: [table + text]
#   Question: What was the YoY change?
#   <|im_end|>
#   <|im_start|>assistant
#   306.2<|im_end|>
#
# WHY THIS FORMAT: Qwen2.5 was trained with this exact template.
# Using a different format means the model doesn't "recognize" the structure
# and performance drops. Every model has its own template.
```

#### `extract_number(text)` — Number parsing
```python
# Financial numbers come in many formats. This function handles ALL of them:
#   "$1,452.4"      → 1452.4      (strip $, remove commas)
#   "1.45B"         → 1450000000  (suffix multiplier)
#   "1,452 million" → 1452000000  (word multiplier)  
#   "(3.5)"         → -3.5        (accounting-style negatives)
#   "45.2%"         → 45.2        (strip %)
#   "RM 12,825"     → 12825       (strip currency symbols)
#
# WHY THIS MATTERS: The GRPO reward function calls this to check if the
# model's answer is correct. If this function can't parse "1.45B" correctly,
# the model gets reward=0 for a correct answer, and GRPO learns garbage.
```

#### `check_answer(prediction, ground_truth)` — Relaxed accuracy
```python
# Compares model's answer to ground truth with ±1% tolerance.
#   check_answer("42.5", "42.0") → True   (within 1%)
#   check_answer("50", "42")     → False  (19% off)
#
# WHY ±1%: Financial computations involve rounding. If the gold answer is
# 306.2 and the model says 306.19, that's correct. Exact match would
# penalize valid answers.
#
# For non-numeric answers (yes/no, text spans): exact match after lowercasing.
```

#### `extract_final_answer(model_output)` — Answer extraction
```python
# The GRPO-trained model might output:
#   "<think>Revenue 2019 = 1452.4, Revenue 2018 = 1146.2.
#    Change = 1452.4 - 1146.2 = 306.2</think>
#    Answer: 306.2"
#
# This function:
#   1. Looks for </think> tag, takes everything after it
#   2. Strips prefixes like "Answer:", "The answer is"
#   3. Takes the first line (models sometimes ramble)
#   4. Returns just "306.2"
```

#### `reward_function(model_output, ground_truth)` — THE GRPO REWARD
```python
# This is the ENTIRE reinforcement learning signal. No neural network.
# No human preferences. Just rules:
#
#   +1.0  if the extracted answer matches ground truth (within ±1%)
#    0.0  if it's wrong
#   +0.2  bonus if the output contains <think>...</think> tags
#
# WHY +0.2 FOR THINKING: We want the model to show its work.
# Without this bonus, the model might learn to just guess numbers
# without reasoning. The bonus nudges it toward step-by-step thinking.
#
# Max reward per completion: 1.2
# This function is called by GRPO for every single completion during training.
```

#### `compute_execution_accuracy(predictions, ground_truths)` — Metric
```python
# Simple: what % of predictions are correct?
# This is the number you report in your results table.
```

---

### `src/step_00_check_gpu.py` — Environment check

What it does: Verifies CUDA is available, prints GPU name and VRAM, checks all required packages are installed, tests if Unsloth is available.

Why it exists: If this fails, nothing else will work. Run it first, always.

Key output to look for: "✓ All checks passed"

---

### `src/step_01_explore_data.py` — Data exploration

What it does:
1. Downloads FinQA from HuggingFace (`datasets.load_dataset("wandb/finqa-data-processed")`)
2. Prints 5 random examples (question, answer, context)
3. Classifies answers into types (numeric / yes-no / text)
4. Saves metadata to `data/format_info.json`

Why it exists: You need to understand what the data looks like before training. Is it mostly numeric? (Yes — ~80%). How long are contexts? (Mean ~1500 chars). This shapes every decision.

---

### `src/step_02_zeroshot_baseline.py` — Checkpoint 1

What it does:
1. Loads Qwen2.5-1.5B in 4-bit quantization
2. For each test example: formats the prompt, generates an answer, checks correctness
3. Reports zero-shot accuracy
4. Saves results to `outputs/zeroshot_results.json`

Key code explained:
```python
# BitsAndBytesConfig — tells HuggingFace to load the model in 4 bits instead of 16
# This is what makes a 1.5B model fit in 8GB VRAM
# NF4 = "Normal Float 4-bit" — a specific quantization format that preserves accuracy
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,           # Use 4-bit weights
    bnb_4bit_quant_type="nf4",   # NF4 format (better than basic int4)
    bnb_4bit_compute_dtype=torch.float16,  # Compute in FP16 (faster than FP32)
    bnb_4bit_use_double_quant=True,  # Quantize the quantization constants too (saves more)
)
```

GO/NO-GO gate: If accuracy < 2%, the model literally cannot read financial tables. Switch to a larger model.

---

### `src/step_03_format_data.py` — Data formatting

What it does:
1. Loads FinQA training split
2. Formats each example into the chat template (system + context + question + answer)
3. Shuffles and splits 90/10 into train and validation
4. Saves to `data/train_sft.json` and `data/val_sft.json`

Why the chat template matters: The model was pretrained with `<|im_start|>` and `<|im_end|>` tokens. If you don't use them, the model doesn't understand where the question ends and the answer begins.

---

### `src/sft_train.py` — Stage 1: Supervised Fine-Tuning

This is the longest and most important training script. Here's what every part does:

#### Model loading with QLoRA
```python
# QLoRA = Quantized Low-Rank Adaptation
# Instead of training all 1.5 billion parameters (impossible on 8GB),
# we freeze the original weights AND quantize them to 4-bit,
# then add tiny trainable "adapter" matrices to specific layers.
#
# LoRA rank=16 means each adapter matrix is 16 dimensions wide.
# That's ~0.5-2% of total parameters being trained.
# The rest is frozen and compressed.

# target_modules — WHICH layers get LoRA adapters:
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
    "gate_proj", "up_proj", "down_proj",       # MLP layers
]
# WHY THESE: These are the layers where most of the "reasoning" happens.
# Attention layers decide WHAT to look at. MLP layers decide WHAT TO DO with it.
```

#### Training configuration
```python
# per_device_train_batch_size=1    — only 1 example per GPU step (VRAM limit)
# gradient_accumulation_steps=8    — accumulate 8 steps before updating weights
#                                    effective batch size = 1 × 8 = 8
# learning_rate=2e-4               — how big each weight update is
#                                    too high = training explodes
#                                    too low = training takes forever
# lr_scheduler_type="cosine"       — learning rate starts at 2e-4, slowly decreases
#                                    to near-zero following a cosine curve
# warmup_ratio=0.05                — first 5% of training uses a tiny learning rate
#                                    (prevents early instability)
# num_train_epochs=3               — see all training data 3 times
```

#### What SFTTrainer does under the hood
```
For each batch:
  1. Tokenize the formatted text (context + question + answer)
  2. Forward pass: model predicts each next token
  3. Loss = cross-entropy between predicted tokens and actual answer tokens
  4. Backward pass: compute gradients (only for LoRA parameters)
  5. Update LoRA weights
```

Output: Saves LoRA adapter weights to `checkpoints/sft/final_adapter/` (~10-30 MB, not the full model)

---

### `src/step_05_eval_sft.py` — Checkpoint 2

Same as step_02 (zero-shot eval), but loads the SFT adapter on top of the base model:
```python
# PeftModel.from_pretrained(base_model, adapter_path)
# This "stacks" the tiny LoRA weights on top of the frozen base model.
# During inference, the adapter modifies the base model's outputs.
```

Compares accuracy with zero-shot to measure SFT improvement.

---

### `src/grpo_train.py` — Stage 2: GRPO (THE CORE)

This is the reinforcement learning stage. Here's the algorithm step by step:

#### What GRPO does (in plain English)
```
For each financial question:
  1. GENERATE: Model produces 4 different answers (by sampling with temperature=0.7)
     Example outputs for "What was revenue change?":
       Completion 1: "306.2"           ← correct
       Completion 2: "1452.4"          ← wrong (that's raw revenue, not the change)
       Completion 3: "<think>1452.4 - 1146.2 = 306.2</think> 306.2"  ← correct + reasoning
       Completion 4: "approximately 300" ← wrong (outside 1% tolerance)

  2. REWARD: Score each completion
       Rewards: [1.0, 0.0, 1.2, 0.0]

  3. ADVANTAGE: Z-score normalize within the group
       Mean = 0.55, Std = 0.57
       Advantages: [+0.79, -0.96, +1.14, -0.96]
       (Positive = better than average, Negative = worse than average)

  4. UPDATE: Adjust model weights
       - Completion 3 (highest advantage) → model learns to produce MORE outputs like this
       - Completions 2 & 4 (negative advantage) → model learns to produce FEWER outputs like this
       - This is the policy gradient: ∇ = advantage × ∇log(probability of this completion)
```

#### Why GRPO instead of PPO
```
PPO (what OpenAI used for ChatGPT):
  - Needs a separate "critic" model (same size as the main model)
  - That means 2× the VRAM — impossible on 8GB
  
GRPO (what DeepSeek used for R1):
  - No critic model
  - Uses the GROUP of completions to estimate the baseline (via z-score)
  - Same result, half the VRAM
```

#### Key hyperparameters
```python
NUM_GENERATIONS = 4     # G = group size. More = better signal, more VRAM
MAX_NEW_TOKENS = 256    # Allow long outputs for <think> reasoning
TEMPERATURE = 0.7       # How "creative" the sampling is
                        # 0.0 = always pick the most likely token (deterministic)
                        # 1.0 = sample proportionally to probabilities (diverse)
                        # 0.7 = moderate diversity (need different completions for GRPO)
LR = 5e-6              # 40× lower than SFT learning rate
                        # RL is unstable — tiny steps prevent collapse
```

#### The reward function wrapper for TRL
```python
def finqa_reward_func(completions, **kwargs):
    # TRL's GRPOTrainer calls this function during training.
    # It passes a list of generated completions + the ground truth via kwargs.
    # We score each completion using our rule-based reward_function.
    # Returns a list of floats — one reward per completion.
```

Output: Saves GRPO-trained adapter to `checkpoints/grpo/final_adapter/`

---

### `src/step_07_eval_grpo.py` — Checkpoint 3 (FINAL)

Same evaluation loop as steps 02 and 05, but also:
- Counts how many outputs contain `<think>` reasoning traces
- Reports "think rate" — what % of answers include step-by-step reasoning
- Prints the full 3-checkpoint comparison table

This is the money shot of your project. The bar chart of all 3 checkpoints is Figure 1 in your report.

---

### `src/analysis.py` — Figure generation

Generates 4 PNG figures from the saved JSON results:

**fig1_accuracy.png** — Bar chart: zero-shot vs SFT vs GRPO accuracy
- Uses matplotlib bar chart with custom colors
- This is the MAIN result of your project

**fig2_grpo_curves.png** — Training curves during GRPO
- Left: loss over training steps (should decrease)
- Right: mean reward over training steps (should increase)
- If reward is flat = GRPO didn't learn anything

**fig3_error_analysis.png** — Per-example comparison SFT vs GRPO
- Categories: Both correct, Both wrong, SFT✓ GRPO✗, SFT✗ GRPO✓
- The "SFT✗ GRPO✓" bar = examples that GRPO specifically fixed

**fig4_think_analysis.png** — Does reasoning help?
- Compares accuracy on examples WHERE the model used `<think>` vs where it didn't
- If `<think>` accuracy > non-`<think>` accuracy → reasoning actually helped

---

### `src/pdf_extractor.py` — PDF processing for real-world reports

This file lets you test the trained model on any financial PDF from any country.

#### `extract_text_from_pdf(pdf_path)` — Smart extraction
```python
# Step 1: Try pdfplumber (direct text extraction — fast, clean)
# Step 2: If a page returns empty text → it's scanned → fall back to OCR
# Step 3: OCR uses pdf2image (converts page to image) + pytesseract (reads the image)
#
# WHY BOTH: Your JP Morgan Malaysia PDF is scanned (Canon scanner).
# A typical SEC filing from EDGAR is digital text. This handles both.
```

#### `find_table_pages(pages)` — Table detection
```python
# Scans each page's text looking for lines with 2+ numbers.
# If a page has 3+ such lines AND mentions a currency → it's a financial table page.
# This filters out the 80 pages of legal boilerplate and finds the 20 pages that matter.
```

#### `detect_currency(text)` — Auto-detection
```python
# Checks for 25+ currency symbols and codes:
# RM, USD, EUR, GBP, INR, JPY, CNY, SGD, HKD, AUD, CHF, etc.
# Also detects scale: "'000" means thousands, "in millions", etc.
# Your JP Morgan PDF says "RM'000" → detects Malaysian Ringgit, scale=thousands
```

---

### `ui/app.py` — Streamlit demo

What users see:
- Left panel: text area to paste financial data + question input
- Right panel: model's answer, reasoning trace (if any), correctness check
- Sidebar: toggle between Zero-Shot / SFT / GRPO checkpoints
- Bottom: results dashboard with all 4 figures

How it works under the hood:
```python
@st.cache_resource  # Load model ONCE, keep in memory
def load_model(base_name, adapter):
    # Same loading code as the eval scripts
    # Uses 4-bit quantization
    # Applies the selected LoRA adapter (SFT or GRPO)

# When user clicks "Ask FinReason":
# 1. format_prompt(context, question)  ← from shared_utils
# 2. tokenizer(prompt)                 ← convert text to token IDs
# 3. model.generate()                  ← run the LLM
# 4. extract_final_answer()            ← parse the output
# 5. Display answer + reasoning + correctness
```

---

### `notebooks/setup.ipynb` — Required for submission

Jupyter notebook that proves your environment works. Contains:
1. Environment check (GPU, packages)
2. FinQA download and loading
3. Data structure exploration
4. Answer type analysis with plots
5. Reward function self-test
6. Model loading test (4-bit)
7. Quick 5-example zero-shot test

**You MUST run this and save with outputs visible before submitting.**

---

## 3. Libraries Used and Why

| Library | What it does | Why we need it |
|---------|-------------|----------------|
| **PyTorch** | Deep learning framework | Everything runs on PyTorch tensors and autograd |
| **Transformers** | Load/run pretrained LLMs | Provides `AutoModelForCausalLM` to load Qwen2.5 |
| **TRL** | RL training for LLMs | Provides `SFTTrainer` (Stage 1) and `GRPOTrainer` (Stage 2) |
| **PEFT** | Parameter-efficient fine-tuning | Implements LoRA — trains 1% of parameters instead of 100% |
| **bitsandbytes** | 4-bit quantization | Makes 1.5B model fit in 8GB VRAM (normally needs ~12GB) |
| **Accelerate** | GPU management | Automatically places model layers on GPU, handles memory |
| **Datasets** | Data loading | One-line download: `load_dataset("wandb/finqa-data-processed")` |
| **Unsloth** | Optimized training | 30-40% VRAM savings + faster training. Optional but recommended |
| **Streamlit** | Web UI | Builds interactive demo with ~100 lines of Python |
| **pdfplumber** | PDF text extraction | Reads text from digital PDFs |
| **pdf2image** | PDF to image | Converts scanned PDF pages to images for OCR |
| **pytesseract** | OCR | Reads text from images (scanned documents) |
| **matplotlib** | Plotting | Generates the 4 figures for the report |

---

## 4. Key Concepts You Must Understand

### QLoRA (Quantized Low-Rank Adaptation)

**Problem:** Fine-tuning a 1.5B model normally needs ~12GB VRAM. You have 8GB.

**Solution (two tricks combined):**

Trick 1 — **Quantization (the Q):** Compress model weights from 16-bit to 4-bit. Each weight goes from using 2 bytes to 0.5 bytes. Model size drops from ~3GB to ~0.8GB in VRAM.

Trick 2 — **LoRA (the LoRA):** Don't train the compressed weights. Instead, add tiny "adapter" matrices to each layer. If a layer's weight matrix is 2048×2048, LoRA adds two small matrices: 2048×16 and 16×2048. Only these are trained. That's 65,536 trainable parameters instead of 4,194,304. Reduction: 98.4%.

Combined: The model fits in VRAM (quantization) and trains efficiently (LoRA).

### GRPO vs PPO vs SFT

| Method | How it learns | Needs reward model? | Needs critic? | VRAM cost |
|--------|--------------|--------------------|----|-----|
| **SFT** | "Copy this answer" | No | No | 1× |
| **PPO** | "This answer is good/bad" (from reward model) | Yes (neural net) | Yes (same size as model) | 4× |
| **GRPO** | "This answer is better than those" (from verifier) | No (rule-based) | No | 1.5× |

GRPO is feasible on consumer hardware because it eliminates both the reward model and the critic.

### Verifiable Rewards (RLVR)

Traditional RLHF: A human looks at the model's output and says "this is good" or "this is bad." Then you train a neural network to predict those preferences. That's the reward model.

RLVR: Skip all of that. The answer to "What is 1452.4 - 1146.2?" is 306.2. Period. A Python function can check this. No humans needed, no neural reward model needed.

This is why financial QA is perfect for GRPO: every answer is a number that can be verified.

### Why SFT Before GRPO (the Cold Start Problem)

GRPO needs the model to produce SOME correct answers during sampling. If it generates 4 answers and all are wrong, reward = [0, 0, 0, 0], advantages = [0, 0, 0, 0], gradient = zero. Nothing is learned.

SFT gets the model from "complete nonsense" to "occasionally correct." GRPO then amplifies the "occasionally correct" into "usually correct."

DeepSeek-R1 called this the "cold start" — their production pipeline also started with SFT before RL.

---

## 5. Data Flow

```
FinQA (HuggingFace)
    │
    ▼
step_01: Download, explore, save format_info.json
    │
    ▼
step_02: Zero-shot eval → zeroshot_results.json (Checkpoint 1)
    │
    ▼
step_03: Format into chat template → train_sft.json, val_sft.json
    │
    ▼
sft_train: QLoRA fine-tuning → checkpoints/sft/final_adapter/
    │
    ▼
step_05: Eval SFT → sft_results.json (Checkpoint 2)
    │
    ▼
grpo_train: GRPO with verifiable rewards → checkpoints/grpo/final_adapter/
    │
    ▼
step_07: Eval GRPO → grpo_results.json (Checkpoint 3)
    │
    ▼
analysis: Generate fig1-4 from all JSON results
    │
    ▼
app.py: Interactive demo (loads any checkpoint, accepts PDF uploads)
```

---

## 6. What to Say When Your Professor Asks

**"Why GRPO and not PPO?"**
> PPO requires a separate critic network the same size as the policy model. On 8GB VRAM, I can't fit two copies of the model. GRPO estimates the baseline from the group of sampled completions instead — same learning signal, half the memory. This is what DeepSeek used for R1.

**"Why not just use SFT?"**
> SFT treats a wrong number and a correct number the same during training — both are just sequences of tokens with cross-entropy loss. GRPO explicitly rewards correct answers and penalizes wrong ones. The reward function checks the actual numerical value, not just the token sequence.

**"Why Qwen2.5-1.5B?"**
> It's the largest model that fits in 8GB VRAM with 4-bit quantization while still leaving room for training. Smaller models (500M) lack the capacity for reasoning. Larger models (7B) need 24GB VRAM.

**"What's the reward function?"**
> Rule-based. Extract the number from the model's output, compare to ground truth within ±1% tolerance. Correct = +1, wrong = 0, plus a 0.2 bonus for showing reasoning in think tags. No neural network, no human annotation.

**"How do you handle different currencies?"**
> The number parsing function strips currency symbols ($, €, £, RM, ₹, ¥) and handles multiplier suffixes (K, M, B, T, million, billion). The PDF extractor auto-detects the report's currency and scale. Training is on US financial data (FinQA), but the numerical reasoning transfers across currencies because math is math — percentages and ratios work the same in RM as in USD.

**"What if GRPO doesn't improve over SFT?"**
> That's a valid result. The project is designed as a controlled experiment with a clear research question. If GRPO doesn't help, I analyze why — reward sparsity, model too small, domain too different from math benchmarks. A negative result with good analysis is still a strong project.

**"What's the think tag?"**
> The prompt asks the model to reason step-by-step inside `<think>...</think>` XML tags before giving the final answer. The reward function gives a +0.2 bonus for using this format. Over training, the model learns that reasoning leads to higher rewards, so it starts showing its work — similar to chain-of-thought prompting, but learned through RL rather than prompted.
