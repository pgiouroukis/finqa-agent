"""
FinQA Data Loading and Formatting.

Functions for loading FinQA dataset and formatting examples for the agent.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any, Sequence


# Default dataset paths
DEFAULT_TRAIN_PATH = "FinQA/dataset/train.json"
DEFAULT_DEV_PATH = "FinQA/dataset/dev.json"
DEFAULT_TEST_PATH = "FinQA/dataset/test.json"


def load_finqa_split(path: str, max_samples: int = 0) -> List[Dict]:
    """
    Load a FinQA dataset split.
    
    Args:
        path: Path to JSON file
        max_samples: If > 0, limit to this many samples
    
    Returns:
        List of FinQA examples
    """
    raw_data = json.loads(Path(path).read_text())
    if max_samples:
        raw_data = raw_data[:max_samples]
    return raw_data


def table_to_lines(table: Sequence[Sequence[str]] | None) -> List[str]:
    """Convert a table to list of formatted row strings."""
    if not table:
        return []
    lines: List[str] = []
    for row in table:
        cleaned = [cell.strip() for cell in row if cell and cell.strip()]
        if cleaned:
            lines.append(" | ".join(cleaned))
    return lines


def build_context(entry: Dict, mode: str = "gold") -> str:
    """
    Build context string from a FinQA example.
    
    Args:
        entry: FinQA example dict
        mode: "gold" (only annotated evidence), "all" (full page), "noisy_gold"
    
    Returns:
        Formatted context string
    """
    qa = entry["qa"]
    
    if mode == "gold":
        gold = qa.get("gold_inds") or {}
        ordered = [gold[key].strip() for key in sorted(gold)]
        if ordered:
            return "\n".join(ordered)
    
    pre = entry.get("pre_text") or []
    post = entry.get("post_text") or []
    table_lines = table_to_lines(entry.get("table"))
    
    parts: List[str] = []
    if pre:
        parts.append("PRE TEXT:\n" + "\n".join(pre))
    if table_lines:
        parts.append("TABLE:\n" + "\n".join(table_lines))
    if post:
        parts.append("POST TEXT:\n" + "\n".join(post))
    
    return "\n\n".join(parts) if parts else "No additional context provided."


def get_all_evidence_pieces(entry: Dict) -> Dict[str, str]:
    """
    Extract all potential evidence pieces from an example.
    
    Returns dict mapping evidence_id -> evidence_text for retriever to score.
    """
    evidence = {}
    
    # Pre-text pieces
    pre_text = entry.get("pre_text") or []
    for i, text in enumerate(pre_text):
        if text.strip() and text.strip() != ".":
            evidence[f"pre_{i}"] = text.strip()
    
    # Post-text pieces
    post_text = entry.get("post_text") or []
    for i, text in enumerate(post_text):
        if text.strip() and text.strip() != ".":
            evidence[f"post_{i}"] = text.strip()
    
    # Table rows
    table = entry.get("table") or []
    for i, row in enumerate(table):
        row_text = " | ".join([cell.strip() for cell in row if cell.strip()])
        if row_text:
            evidence[f"table_{i}"] = row_text
    
    return evidence


def get_gold_evidence_ids(entry: Dict) -> List[str]:
    """
    Get the IDs of gold evidence pieces for evaluation.
    """
    qa = entry.get("qa", {})
    gold_ids = []
    
    # Text evidence (ann_text_rows maps to pre_text + post_text concatenated)
    ann_text_rows = qa.get("ann_text_rows") or []
    pre_len = len(entry.get("pre_text") or [])
    
    for idx in ann_text_rows:
        if idx < pre_len:
            gold_ids.append(f"pre_{idx}")
        else:
            gold_ids.append(f"post_{idx - pre_len}")
    
    # Table evidence
    ann_table_rows = qa.get("ann_table_rows") or []
    for idx in ann_table_rows:
        gold_ids.append(f"table_{idx}")
    
    return gold_ids


def get_question_data(data: List[Dict], query_id: str | int) -> Dict | None:
    """
    Get question data by ID.
    
    Args:
        data: Loaded FinQA data
        query_id: Index or ID string
    
    Returns:
        Question data dict or None
    """
    try:
        idx = int(query_id)
        if 0 <= idx < len(data):
            entry = data[idx]
            return {
                "query_id": query_id,
                "question": entry["qa"]["question"],
                "answer": str(entry["qa"].get("exe_ans", entry["qa"].get("answer", ""))),
                "program": entry["qa"].get("program", ""),
                "gold_inds": entry["qa"].get("gold_inds", {}),
                "table": entry.get("table"),
                "pre_text": entry.get("pre_text"),
                "post_text": entry.get("post_text"),
                "full_entry": entry,
            }
    except (ValueError, IndexError):
        pass
    return None


def format_evidence_for_generator(evidence_pieces: List[str]) -> str:
    """Format retrieved evidence pieces for the generator prompt."""
    if not evidence_pieces:
        return "No evidence retrieved."
    return "\n".join(f"- {piece}" for piece in evidence_pieces)


def build_generator_prompt(question: str, evidence: str) -> str:
    """
    Build prompt for the DSL generator model.
    
    Follows the format used in training (from finetune_gemma.py).
    """
    hint = "Generate the FinQA reasoning program (DSL) that answers the question."
    return f"{hint}\n\nQuestion:\n{question}\n\nContext:\n{evidence}\n\nAnswer:"
