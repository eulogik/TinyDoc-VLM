"""Unit tests for the page-wise vs all-pages comparison (no model calls)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evaluation" / "invoice"))

from compare_pagewise import compare_runs


def _doc(doc_id, correctness, fully, line_f1=1.0):
    return {
        "doc_id": doc_id,
        "fields": {name: {"correct": hit} for name, hit in correctness.items()},
        "fully_correct": fully,
        "line_f1": line_f1,
    }


def _scores(*docs):
    return {"documents": list(docs)}


def test_pagewise_gain_is_paired():
    before = _scores(
        _doc("a", {"vendor_name": True, "total": False}, False),
        _doc("b", {"vendor_name": True, "total": True}, True),
    )
    after = _scores(
        _doc("a", {"vendor_name": True, "total": True}, True),
        _doc("b", {"vendor_name": False, "total": True}, False),
    )
    report = compare_runs(before, after)
    assert report["n_docs"] == 2
    assert report["before"]["all_pages"]["key_correct"] == 3
    assert report["after"]["pagewise"]["key_correct"] == 3
    assert report["transitions"]["key_improved"] == 1
    assert report["transitions"]["key_regressed"] == 1
    assert report["transitions"]["fully_correct_gained"] == 1
    assert report["transitions"]["fully_correct_lost"] == 1
    assert report["per_field"]["total"]["all_pages_correct"] == 1
    assert report["per_field"]["total"]["pagewise_correct"] == 2
    assert "Descriptive only" in report["caveat"]


def test_missing_documents_are_reported():
    before = _scores(_doc("a", {"total": True}, True))
    after = _scores(
        _doc("a", {"total": True}, True),
        _doc("b", {"total": True}, True),
    )
    report = compare_runs(before, after)
    assert report["n_docs"] == 1
    assert report["missing_from_before"] == ["b"]
    assert report["missing_from_after"] == []
