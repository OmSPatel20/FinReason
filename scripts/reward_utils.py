"""
reward_utils.py — Reward function for FinQA GRPO training.
+1.0 correct answer, 0.0 wrong, +0.2 bonus for <think> reasoning.

RUN THIS FILE DIRECTLY to self-test:  python scripts/reward_utils.py
"""
import re
from typing import Optional


def extract_number(text: str) -> Optional[float]:
    """Extract a numeric value from messy financial text."""
    if text is None:
        return None
    text = str(text).strip()
    if not text:
        return None

    text = re.sub(r'[$€£¥]', '', text).strip()

    # Accounting-style negatives: (123) -> -123
    paren = re.match(r'^\(([0-9,.\s]+)\)$', text)
    if paren:
        text = '-' + paren.group(1)

    text = text.replace(',', '')

    # Word multipliers
    for word, mult in {'thousand': 1e3, 'thousands': 1e3, 'million': 1e6,
                        'millions': 1e6, 'billion': 1e9, 'billions': 1e9,
                        'trillion': 1e12, 'trillions': 1e12}.items():
        if word in text.lower():
            cleaned = text.lower().replace(word, '').strip()
            try:
                return float(cleaned) * mult
            except ValueError:
                pass

    # Suffix multipliers: 1.45M, 1.45B
    sfx = re.match(r'^([+-]?[0-9.]+)\s*([KkMmBbTt])$', text)
    if sfx:
        mult_map = {'K': 1e3, 'M': 1e6, 'B': 1e9, 'T': 1e12}
        try:
            return float(sfx.group(1)) * mult_map.get(sfx.group(2).upper(), 1)
        except ValueError:
            pass

    text = text.rstrip('%').strip()

    try:
        return float(text)
    except ValueError:
        pass

    # Last resort: find any number
    m = re.search(r'[+-]?[0-9]+\.?[0-9]*', text)
    if m:
        try:
            return float(m.group())
        except ValueError:
            pass
    return None


def normalize_text(text: str) -> str:
    text = str(text).lower().strip().replace(',', '')
    text = re.sub(r'\s+', ' ', text).rstrip('.')
    return text


def check_answer(prediction: str, ground_truth: str, tolerance: float = 0.01) -> bool:
    """Check if prediction matches ground truth (±1% for numbers, exact for text)."""
    pred_num = extract_number(prediction)
    gt_num = extract_number(ground_truth)

    if pred_num is not None and gt_num is not None:
        if gt_num == 0:
            return abs(pred_num) < 0.001
        return abs(pred_num - gt_num) / abs(gt_num) <= tolerance

    return normalize_text(prediction) == normalize_text(ground_truth)


def extract_final_answer(model_output: str) -> str:
    """Extract the final answer from model output, handling <think> blocks."""
    text = model_output.strip()

    # If there's a </think> tag, take everything after it
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


def reward_function(model_output: str, ground_truth: str) -> float:
    """Compute GRPO reward. Correct=+1.0, Wrong=0.0, Think bonus=+0.2."""
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
    correct = sum(check_answer(extract_final_answer(p), g)
                  for p, g in zip(predictions, ground_truths))
    return correct / len(predictions)


# ---- SELF TEST ----
if __name__ == "__main__":
    print("=" * 50)
    print("REWARD FUNCTION SELF-TEST")
    print("=" * 50)
    tests = [
        ("42.5", "42.5", True),
        ("42.5", "42.0", True),
        ("50.0", "42.0", False),
        ("$1,452.4", "1452.4", True),
        ("1452.4 million", "1452400000", True),
        ("1.45B", "1450000000", True),
        ("(3.5)", "-3.5", True),
        ("45.2%", "45.2", True),
        ("yes", "Yes", True),
        ("<think>Revenue changed by 306.2</think>\n306.2", "306.2", True),
        ("Answer: 306.2", "306.2", True),
        ("The answer is 42.5", "42.5", True),
    ]
    passed = 0
    for out, gt, expected in tests:
        final = extract_final_answer(out)
        result = check_answer(final, gt)
        ok = result == expected
        passed += ok
        r = reward_function(out, gt)
        print(f"  {'✓' if ok else '✗ FAIL'} | '{out[:45]}' vs '{gt}' → "
              f"extracted='{final}' correct={result} R={r:.1f}")
    print(f"\n  {passed}/{len(tests)} passed")
    if passed == len(tests):
        print("  All good. Proceed to training.")
    else:
        print("  FIX FAILURES BEFORE TRAINING.")
