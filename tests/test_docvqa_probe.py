"""Unit tests for the DocVQA reading-probe helpers (no model calls)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evaluation" / "docvqa"))

from build_manifest import clean_question, select_pairs
from run_docvqa_probe import (
    anls_score,
    extract_answer,
    normalize_answer,
    summarize,
)


def test_clean_question_strips_prefix():
    assert clean_question("Answer the question: What is the total?") == "What is the total?"
    assert clean_question("answer the question :  Who signed?") == "Who signed?"
    assert clean_question("Who signed?") == "Who signed?"


def test_select_pairs_is_deterministic_and_one_per_image():
    rows = [{"image_path": f"/x/img_{i % 7}.jpg", "prompt": f"q{i}"} for i in range(40)]
    first = select_pairs(rows, limit=5, seed=42)
    second = select_pairs(rows, limit=5, seed=42)
    assert [r["prompt"] for r in first] == [r["prompt"] for r in second]
    assert len(first) == 5
    assert len({Path(r["image_path"]).name for r in first}) == 5


def test_normalize_answer_removes_case_punctuation_articles():
    assert normalize_answer("The 42.5%") == "425"
    assert normalize_answer("A Total: 1,200") == "total 1200"
    assert normalize_answer("  ") == ""


def test_anls_exact_and_threshold():
    assert anls_score("0.28", "0.28") == 1.0
    # punctuation is stripped: "029" vs "028" is one edit in three
    assert abs(anls_score("0.29", "0.28") - 2 / 3) < 1e-9
    # completely different answers fall below 0.5
    assert anls_score("banana", "0.28") == 0.0
    assert anls_score("", "0.28") == 0.0


def test_extract_answer_handles_common_shapes():
    assert extract_answer("  March 12, 1975\n") == "March 12, 1975"
    assert extract_answer("Answer: $1,200") == "$1,200"
    assert extract_answer('{"answer": "0.28"}') == "0.28"
    assert extract_answer('```json\n{"answer": "0.28"}\n```') == "0.28"
    assert extract_answer("") == ""


def test_summarize_counts_and_means():
    rows = [
        {"anls": 1.0, "exact": True, "latency_ms": 100.0, "error": None},
        {"anls": 0.0, "exact": False, "latency_ms": 300.0, "error": None},
        {"anls": 0.0, "exact": False, "latency_ms": 200.0, "error": "timeout"},
    ]
    out = summarize(rows)
    assert out["n"] == 3
    assert out["anls_mean"] == 1 / 3
    assert out["exact_rate"] == 1 / 3
    assert out["errors"] == 1
    assert out["p50_latency_ms"] == 200.0
