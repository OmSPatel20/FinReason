"""
shared_utils.py — Shared across ALL scripts.

Contains:
  - FinQA data extraction (handles multiple HuggingFace format versions)
  - Reward function (the GRPO verifier)
  - Number extraction & normalization
  - Execution accuracy metric
  - Prompt formatting
"""
import re
import os
from typing import Optional

# =====================================================================
# DATA EXTRACTION — handles different FinQA HuggingFace formats
# =====================================================================

def extract_qa(example: dict) -> tuple:
    """Extract (question, answer) from a FinQA example."""
    # Format 1: flat columns
    if "question" in example and "answer" in example:
        return str(example["question"]), str(example["answer"])
    # Format 2: nested qa dict
    if "qa" in example and isinstance(example["qa"], dict):
        qa = example["qa"]
        q = qa.get("question", "")
        a = str(qa.get("exe_ans", qa.get("answer", "")))
        return q, a
    # Format 3: input/output
    if "input" in example:
        return str(example["input"]), str(example.get("output", ""))
    return "", ""


def extract_context(example: dict) -> str:
    """Extract financial context (table + text) from a FinQA example."""
    if "context" in example:
        return str(example["context"])
    parts = []
    for key in ["pre_text", "table", "post_text"]:
        if key not in example:
            continue
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


# =====================================================================
# PROMPT FORMATTING
# =====================================================================

SYSTEM_MSG = (
    "You are a financial analyst expert. Given financial data (tables and text), "
    "answer the question with a precise numerical answer. "
    "Think step by step in <think> tags, then give your final answer."
)


def format_prompt(context: str, question: str, answer: str = None,
                  max_context_chars: int = 1200) -> str:
    """
    Format into Qwen chat template.
    If answer is provided, includes it (for SFT training).
    If answer is None, leaves it open (for inference).
    """
    if len(context) > max_context_chars:
        context = context[:max_context_chars] + "\n[...truncated...]"

    prompt = (
        f"<|im_start|>system\n{SYSTEM_MSG}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Financial Data:\n{context}\n\n"
        f"Question: {question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    if answer is not None:
        prompt += f"{answer}<|im_end|>"
    return prompt


# =====================================================================
# NUMBER EXTRACTION — handles messy financial output formats
# =====================================================================

def extract_number(text: str) -> Optional[float]:
    """
    Parse a numeric value from financial text.
    Handles: $1,452.4 | 1.45B | (3.5) | 45.2% | 1,452 million
    """
    if text is None:
        return None
    text = str(text).strip()
    if not text:
        return None

    text = re.sub(r'[$€£¥]', '', text).strip()

    # Accounting negatives: (123) -> -123
    m = re.match(r'^\(([0-9,.\s]+)\)$', text)
    if m:
        text = '-' + m.group(1)

    text = text.replace(',', '')

    # Word multipliers
    word_mult = {
        'thousand': 1e3, 'thousands': 1e3,
        'million': 1e6, 'millions': 1e6,
        'billion': 1e9, 'billions': 1e9,
        'trillion': 1e12, 'trillions': 1e12,
    }
    lower = text.lower()
    for word, mult in word_mult.items():
        if word in lower:
            try:
                return float(lower.replace(word, '').strip()) * mult
            except ValueError:
                pass

    # Suffix multipliers: 1.45M, 1.45B
    sm = re.match(r'^([+-]?[0-9.]+)\s*([KkMmBbTt])$', text)
    if sm:
        suffix_map = {'K': 1e3, 'M': 1e6, 'B': 1e9, 'T': 1e12}
        try:
            return float(sm.group(1)) * suffix_map.get(sm.group(2).upper(), 1)
        except ValueError:
            pass

    # Strip trailing %
    text = text.rstrip('%').strip()

    try:
        return float(text)
    except ValueError:
        pass

    # Last resort: find any number
    nm = re.search(r'[+-]?[0-9]+\.?[0-9]*', text)
    if nm:
        try:
            return float(nm.group())
        except ValueError:
            pass
    return None


# =====================================================================
# ANSWER CHECKING
# =====================================================================

def normalize_text(text: str) -> str:
    text = str(text).lower().strip().replace(',', '')
    text = re.sub(r'\s+', ' ', text).rstrip('.')
    return text


def check_answer(prediction: str, ground_truth: str,
                 tolerance: float = 0.01) -> bool:
    """
    Relaxed accuracy:
    - Numeric: within ±tolerance
    - Text: exact match after normalization
    """
    pred_num = extract_number(prediction)
    gt_num = extract_number(ground_truth)

    if pred_num is not None and gt_num is not None:
        if gt_num == 0:
            return abs(pred_num) < 0.001
        return abs(pred_num - gt_num) / abs(gt_num) <= tolerance

    return normalize_text(prediction) == normalize_text(ground_truth)


def extract_final_answer(model_output: str) -> str:
    """Extract answer from model output, handling <think> blocks and prefixes."""
    text = model_output.strip()

    # After </think> tag
    m = re.search(r'</think>\s*(.*)', text, re.DOTALL)
    if m:
        text = m.group(1).strip()

    # Remove common prefixes
    for prefix in ["Answer:", "The answer is:", "The answer is",
                   "Final answer:", "A:", "Result:"]:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()

    # Take first line only
    lines = text.strip().split('\n')
    if lines:
        text = lines[0].strip()

    return text.rstrip('.').strip()


# =====================================================================
# REWARD FUNCTION — used by GRPO
# =====================================================================

def reward_function(model_output: str, ground_truth: str) -> float:
    """
    GRPO reward:
      +1.0  correct answer
       0.0  wrong answer
      +0.2  bonus for <think> reasoning
    """
    reward = 0.0
    has_think = bool(re.search(r'<think>.*?</think>', model_output, re.DOTALL))
    final = extract_final_answer(model_output)

    if check_answer(final, ground_truth):
        reward += 1.0
    if has_think:
        reward += 0.2
    return reward


def compute_execution_accuracy(predictions: list, ground_truths: list) -> float:
    assert len(predictions) == len(ground_truths)
    correct = sum(
        check_answer(extract_final_answer(p), g)
        for p, g in zip(predictions, ground_truths)
    )
    return correct / len(predictions)


# =====================================================================
# SELF-TEST
# =====================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("  SHARED UTILITIES — SELF-TEST")
    print("=" * 50)

    tests = [
        ("42.5", "42.5", True),
        ("$1,452.4", "1452.4", True),
        ("1.45B", "1450000000", True),
        ("(3.5)", "-3.5", True),
        ("45.2%", "45.2", True),
        ("yes", "Yes", True),
        ("50", "42", False),
        ("<think>1452.4 - 1146.2 = 306.2</think>\n306.2", "306.2", True),
        ("Answer: 306.2", "306.2", True),
        ("The answer is 42.5", "42.5", True),
        ("1,452 million", "1452000000", True),
    ]

    passed = 0
    for output, gt, expected in tests:
        final = extract_final_answer(output)
        result = check_answer(final, gt)
        ok = result == expected
        passed += ok
        r = reward_function(output, gt)
        print(f"  {'✓' if ok else '✗'} '{output[:45]:45s}' vs '{gt:15s}'"
              f" → correct={result} R={r:.1f}")

    print(f"\n  {passed}/{len(tests)} passed")
    if passed < len(tests):
        print("  ✗ FIX FAILURES BEFORE TRAINING")
    else:
        print("  ✓ All good. Reward function ready.")
    print("=" * 50)
