"""
shared_utils.py — Used by every script in the project.
Contains: data extraction, prompt formatting, reward function, metrics.
"""
import re, os
from typing import Optional

# === DATA EXTRACTION ===
def extract_qa(example):
    # wandb/finqa-data-processed: columns are query, exe_ans, output, context
    if "query" in example and "exe_ans" in example:
        return str(example["query"]), str(example["exe_ans"])
    if "query" in example and "output" in example:
        return str(example["query"]), str(example["output"])
    if "question" in example and "answer" in example:
        return str(example["question"]), str(example["answer"])
    if "qa" in example and isinstance(example["qa"], dict):
        qa = example["qa"]
        return qa.get("question", ""), str(qa.get("exe_ans", qa.get("answer", "")))
    return str(example.get("input", "")), str(example.get("output", ""))

def extract_context(example):
    if "context" in example:
        return str(example["context"])
    parts = []
    for key in ["pre_text", "table", "post_text"]:
        if key not in example: continue
        val = example[key]
        if isinstance(val, list):
            for item in val:
                if isinstance(item, list):
                    parts.append(" | ".join(str(c) for c in item))
                else:
                    parts.append(str(item))
        else:
            parts.append(str(val))
    return "\n".join(parts)

# === PROMPT FORMATTING ===
SYSTEM_MSG_TRAIN = (
    "You are a financial analyst expert. Given financial data (tables and text), "
    "answer the question with a precise numerical answer. "
    "Think step by step in <think> tags, then give your final answer."
)

SYSTEM_MSG_EVAL = (
    "You are a financial analyst. Given financial data, answer with ONLY "
    "a single number or short phrase. No explanations. No words. Just the answer."
)

def format_prompt(context, question, answer=None, max_context_chars=1200, mode="train"):
    """
    mode="train" → uses think-tag prompt (for SFT training and GRPO)
    mode="eval"  → uses strict short-answer prompt (for zero-shot and SFT eval)
    mode="grpo_eval" → uses think-tag prompt (for GRPO eval, model learned to use them)
    """
    if len(context) > max_context_chars:
        context = context[:max_context_chars] + "\n[...truncated...]"
    sys_msg = SYSTEM_MSG_TRAIN if mode in ("train", "grpo_eval") else SYSTEM_MSG_EVAL
    prompt = (
        f"<|im_start|>system\n{sys_msg}<|im_end|>\n"
        f"<|im_start|>user\nFinancial Data:\n{context}\n\n"
        f"Question: {question}<|im_end|>\n<|im_start|>assistant\n"
    )
    if answer is not None:
        prompt += f"{answer}<|im_end|>"
    return prompt

# === NUMBER EXTRACTION ===
def extract_number(text):
    if text is None: return None
    text = str(text).strip()
    if not text: return None
    text = re.sub(r'[$€£¥₹]', '', text).strip()
    text = re.sub(r'^(?:RM|SGD|HKD|AUD|INR|CNY|JPY|GBP|EUR|CHF|CAD|AED|SAR)\s*', '', text, flags=re.IGNORECASE).strip()
    m = re.match(r'^\(([0-9,.\s]+)\)$', text)
    if m: text = '-' + m.group(1)
    text = text.replace(',', '')
    for word, mult in {'thousand':1e3,'thousands':1e3,'million':1e6,'millions':1e6,
                        'billion':1e9,'billions':1e9,'trillion':1e12,'trillions':1e12}.items():
        if word in text.lower():
            try: return float(text.lower().replace(word,'').strip()) * mult
            except ValueError: pass
    sm = re.match(r'^([+-]?[0-9.]+)\s*([KkMmBbTt])$', text)
    if sm:
        try: return float(sm.group(1)) * {'K':1e3,'M':1e6,'B':1e9,'T':1e12}.get(sm.group(2).upper(),1)
        except ValueError: pass
    text = text.rstrip('%').strip()
    try: return float(text)
    except ValueError: pass
    nm = re.search(r'[+-]?[0-9]+\.?[0-9]*', text)
    if nm:
        try: return float(nm.group())
        except ValueError: pass
    return None

# === ANSWER CHECKING ===
def normalize_text(text):
    return re.sub(r'\s+', ' ', str(text).lower().strip().replace(',','')).rstrip('.')

def check_answer(prediction, ground_truth, tolerance=0.01):
    pred_num = extract_number(prediction)
    gt_num = extract_number(ground_truth)
    if pred_num is not None and gt_num is not None:
        if gt_num == 0: return abs(pred_num) < 0.001
        return abs(pred_num - gt_num) / abs(gt_num) <= tolerance
    return normalize_text(prediction) == normalize_text(ground_truth)

def extract_final_answer(model_output):
    text = model_output.strip()
    m = re.search(r'</think>\s*(.*)', text, re.DOTALL)
    if m: text = m.group(1).strip()
    for prefix in ["Answer:","The answer is:","The answer is","Final answer:","A:","Result:"]:
        if text.lower().startswith(prefix.lower()): text = text[len(prefix):].strip()
    lines = text.strip().split('\n')
    if lines: text = lines[0].strip()
    return text.rstrip('.').strip()

# === REWARD FUNCTION ===
def reward_function(model_output, ground_truth):
    reward = 0.0
    has_think = bool(re.search(r'<think>.*?</think>', model_output, re.DOTALL))
    final = extract_final_answer(model_output)
    if check_answer(final, ground_truth): reward += 1.0
    if has_think: reward += 0.2
    return reward

def compute_execution_accuracy(predictions, ground_truths):
    assert len(predictions) == len(ground_truths)
    return sum(check_answer(extract_final_answer(p), g) for p, g in zip(predictions, ground_truths)) / len(predictions)

# === SELF-TEST ===
if __name__ == "__main__":
    tests = [("42.5","42.5",True),("$1,452.4","1452.4",True),("1.45B","1450000000",True),
             ("(3.5)","-3.5",True),("45.2%","45.2",True),("50","42",False),
             ("<think>306.2</think>\n306.2","306.2",True),("Answer: 306.2","306.2",True),
             ("RM 12,825","12825",True),("1,452 million","1452000000",True)]
    passed = sum(1 for o,g,e in tests if check_answer(extract_final_answer(o),g)==e)
    for o,g,e in tests:
        ok = check_answer(extract_final_answer(o),g)==e
        print(f"  {'✓' if ok else '✗'} '{o[:40]:40s}' vs '{g}'")
    print(f"\n  {passed}/{len(tests)} passed")
