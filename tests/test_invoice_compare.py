"""Unit tests for the paired invoice-model comparison (no model calls)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evaluation" / "invoice"))

from compare_models import compare


def _doc(doc_id, category, hits, full):
    return {
        "doc_id": doc_id,
        "category": category,
        "fields": {name: {"correct": hit} for name, hit in hits.items()},
        "fully_correct": full,
    }


def _scores(docs):
    return {"documents": docs}


def test_compare_paired_documents():
    a = _scores(
        [
            _doc("d1", "clean", {"vendor_name": True, "invoice_number": True}, True),
            _doc("d2", "clean", {"vendor_name": True, "invoice_number": True}, True),
            _doc("d3", "photo", {"vendor_name": True, "invoice_number": False}, False),
            _doc("d4", "photo", {"vendor_name": False, "invoice_number": False}, False),
        ]
    )
    b = _scores(
        [
            _doc("d1", "clean", {"vendor_name": True, "invoice_number": True}, True),
            _doc("d2", "clean", {"vendor_name": True, "invoice_number": False}, False),
            _doc("d3", "photo", {"vendor_name": False, "invoice_number": False}, False),
            _doc("d4", "photo", {"vendor_name": False, "invoice_number": False}, False),
        ]
    )
    report = compare(a, b, "3b", "2b", n_boot=300)

    assert report["n_docs"] == 4
    assert report["key"]["a"] == 5 / 8
    assert report["key"]["b"] == 3 / 8
    assert report["key"]["diff"] == -2 / 8
    assert report["key"]["a_better_docs"] == 2
    assert report["key"]["b_better_docs"] == 0
    assert report["key"]["ties"] == 2
    assert report["key"]["ci95"][0] <= -2 / 8 <= report["key"]["ci95"][1]

    assert report["fully_correct"]["a"] == 0.5
    assert report["fully_correct"]["b"] == 0.25
    assert report["fully_correct"]["a_only"] == 1
    assert report["fully_correct"]["b_only"] == 0

    assert report["per_category"]["clean"]["diff"] == -1 / 4
    assert report["per_category"]["photo"]["diff"] == -1 / 4
    assert report["per_field"]["invoice_number"]["diff"] == -0.25


def test_compare_identical_runs():
    docs = [_doc("d1", "clean", {"vendor_name": True}, True)]
    report = compare(_scores(docs), _scores(docs), n_boot=100)
    assert report["key"]["diff"] == 0.0
    assert report["fully_correct"]["diff"] == 0.0
    assert report["key"]["sign_p"] == 1.0
