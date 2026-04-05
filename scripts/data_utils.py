"""
data_utils.py — Shared data extraction functions for FinQA.
Every script imports from here so column-name logic lives in ONE place.
"""


def extract_qa(example):
    """Extract (question, answer) from a FinQA example, regardless of format."""
    if "question" in example and "answer" in example:
        return example["question"], str(example["answer"])
    if "qa" in example:
        qa = example["qa"]
        if isinstance(qa, dict):
            return qa.get("question", ""), str(qa.get("exe_ans", ""))
    if "input" in example:
        return example["input"], str(example.get("output", example.get("target", "")))
    return "", ""


def extract_context(example):
    """Extract the financial context (table + text) from a FinQA example."""
    if "context" in example:
        return str(example["context"])
    parts = []
    for key in ["pre_text", "table", "post_text"]:
        if key in example:
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


def truncate_context(context, max_chars=1200):
    """Truncate context to fit within token limits, keeping table intact."""
    if len(context) <= max_chars:
        return context
    return context[:max_chars] + "\n[...truncated...]"


SYSTEM_MSG = (
    "You are a financial analyst expert. Given financial data (tables and text), "
    "answer the question with a precise numerical answer. "
    "Think step by step in <think> tags, then give your final answer."
)


def build_prompt(context, question):
    """Build the chat-formatted prompt for inference."""
    ctx = truncate_context(context, 1200)
    return (
        f"<|im_start|>system\n{SYSTEM_MSG}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Financial Data:\n{ctx}\n\n"
        f"Question: {question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def build_training_text(context, question, answer):
    """Build full training text (with answer) for SFT."""
    ctx = truncate_context(context, 1200)
    return (
        f"<|im_start|>system\n{SYSTEM_MSG}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Financial Data:\n{ctx}\n\n"
        f"Question: {question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"{answer}<|im_end|>"
    )
