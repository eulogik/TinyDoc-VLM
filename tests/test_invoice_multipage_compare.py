"""Unit tests for the page-1 vs all-pages multipage comparison (no model calls)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evaluation" / "invoice"))

from compare_multipage import compare, latency_summary


def _doc(doc_id, correctness, fully_correct, line_f1=1.0):
    return {
        "doc_id": doc_id,
        "fields": {
            name: {"correct": hit} for name, hit in correctness.items()
        },
        "fully_correct": fully_correct,
        "line_f1": line_f1,
    }


def _scores(*docs):
    return {"documents": list(docs)}


def test_multipage_gain_and_regression_are_paired():
    page1 = _scores(
        _doc("multipage-a", {"vendor_name": True, "total": False}, False),
        _doc("multipage-b", {"vendor_name": True, "total": True}, True),
        _doc("clean-c", {"vendor_name": True, "total": True}, True),
    )
    all_pages = _scores(
        _doc("multipage-a", {"vendor_name": True, "total": True}, True),
        _doc("multipage-b", {"vendor_name": False, "total": True}, False),
    )
    report = compare(page1, all_pages)

    assert report["n_docs"] == 2
    assert report["page1"]["key_correct"] == 3
    assert report["all_pages"]["key_correct"] == 3
    assert report["delta"]["key_accuracy"] == 0.0
    assert report["transitions"]["key_improved"] == 1
    assert report["transitions"]["key_regressed"] == 1
    assert report["transitions"]["key_unchanged"] == 0
    assert report["transitions"]["key_sign_p"] == 1.0
    assert report["transitions"]["fully_correct_gained"] == 1
    assert report["transitions"]["fully_correct_lost"] == 1
    assert report["transitions"]["fully_correct_sign_p"] == 1.0
    assert report["per_field"]["total"]["page1_correct"] == 1
    assert report["per_field"]["total"]["all_pages_correct"] == 2
    assert report["per_field"]["vendor_name"]["delta"] == -0.5
    assert "Descriptive only" in report["caveat"]


def test_non_multipage_docs_are_ignored():
    page1 = _scores(_doc("multipage-a", {"total": True}, True))
    all_pages = _scores(
        _doc("multipage-a", {"total": True}, True),
        _doc("clean-b", {"total": True}, True),
    )
    report = compare(page1, all_pages)
    assert report["n_docs"] == 1
    assert report["missing_from_page1_scores"] == []


def test_latency_summary_per_document_and_page():
    rows = [
        {"latency_ms": 10_000, "image_paths": ["a", "b"]},
        {"latency_ms": 30_000, "image_paths": ["a", "b", "c", "d"]},
    ]
    out = latency_summary(rows)
    assert out["n"] == 2
    assert out["p50_s"] == 20.0
    assert out["mean_s"] == 20.0
    assert out["p50_s_per_page"] == 6.25
