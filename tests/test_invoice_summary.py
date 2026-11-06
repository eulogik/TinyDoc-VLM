"""Unit tests for the invoice score summarizer (no model calls)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evaluation" / "invoice"))

from summarize_scores import summarize


def _doc(doc_id, category, key_hits, fully, line_f1=None):
    fields = {}
    for name, hit in key_hits.items():
        fields[name] = {"correct": hit, "scored": 1}
    return {
        "doc_id": doc_id,
        "category": category,
        "fields": fields,
        "fully_correct": fully,
        "line_f1": line_f1,
    }


def test_summary_key_accuracy_and_categories():
    scores = {
        "meta": {"system": "test", "run_date": "2026-11-06"},
        "summary": {"documents_in_corpus": 4},
        "documents": [
            _doc("a", "clean", {"vendor_name": True, "invoice_number": False}, False, 1.0),
            _doc("b", "clean", {"vendor_name": True, "invoice_number": True}, True, 0.5),
            _doc("c", "photo", {"vendor_name": False, "invoice_number": True}, False, 0.0),
        ],
    }
    rows = [
        {"id": "a", "latency_ms": 100.0, "error": None},
        {"id": "b", "latency_ms": 300.0, "error": "timeout"},
        {"id": "c", "latency_ms": 200.0, "error": None},
    ]
    out = summarize(scores, rows)
    assert out["n_docs"] == 3
    assert out["key_header_correct"] == 4
    assert out["key_header_scored"] == 6
    assert abs(out["key_header_accuracy"] - 4 / 6) < 1e-9
    assert out["fully_correct"] == 1
    assert out["per_category"]["clean"]["docs"] == 2
    assert out["per_category"]["clean"]["key_correct"] == 3
    assert out["per_category"]["clean"]["fully_correct"] == 1
    assert out["per_category"]["photo"]["key_correct"] == 1
    assert out["latency_ms"]["p50"] == 200.0
    assert out["parse_errors"] == 1


def test_summary_without_rows():
    scores = {"documents": [], "meta": {}}
    out = summarize(scores, None)
    assert out["n_docs"] == 0
    assert out["key_header_accuracy"] == 0.0
    assert "latency_ms" not in out
