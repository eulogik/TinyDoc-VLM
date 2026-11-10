"""Unit tests for the invoice score summarizer (no model calls)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evaluation" / "invoice"))

from summarize_scores import bootstrap_ci, clopper_pearson, summarize


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


def test_bootstrap_ci_constant_documents():
    docs = [
        _doc(f"d{i}", "clean", {"vendor_name": True, "invoice_number": True}, True)
        for i in range(10)
    ]
    ci = bootstrap_ci(docs, n_boot=200)
    assert ci["key_header_accuracy"] == [1.0, 1.0]
    assert ci["fully_correct_rate"] == [1.0, 1.0]


def test_bootstrap_ci_contains_point_estimate():
    docs = [
        _doc("a", "clean", {"vendor_name": True, "invoice_number": True}, True),
        _doc("b", "photo", {"vendor_name": False, "invoice_number": True}, False),
        _doc("c", "photo", {"vendor_name": False, "invoice_number": False}, False),
    ]
    ci = bootstrap_ci(docs, n_boot=500)
    low, high = ci["key_header_accuracy"]
    assert low <= 3 / 6 <= high
    low, high = ci["fully_correct_rate"]
    assert low <= 1 / 3 <= high


def test_exact_interval_does_not_collapse_at_all_correct():
    interval = clopper_pearson(10, 10)
    assert interval is not None
    assert 0.68 < interval[0] < 0.70
    assert interval[1] == 1.0


def test_exact_interval_at_zero_and_empty():
    interval = clopper_pearson(0, 10)
    assert interval[0] == 0.0
    assert 0.30 < interval[1] < 0.32
    assert clopper_pearson(0, 0) is None


def test_summary_contains_exact_interval():
    scores = {
        "documents": [
            _doc(f"d{i}", "clean", {"vendor_name": True}, True) for i in range(10)
        ],
        "meta": {},
    }
    out = summarize(scores, None)
    exact = out["ci95_exact"]
    assert exact["fully_correct_rate"][1] == 1.0
    assert exact["fully_correct_rate"][0] < 0.75
    assert "optimistic" in exact["note"]
