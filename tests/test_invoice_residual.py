"""Unit tests for the invoice residual report (no model calls)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evaluation" / "invoice"))

from residual_report import analyze


def _doc(doc_id, category, misses):
    fields = {name: {"correct": name not in misses} for name in ("subtotal", "tax", "total", "currency")}
    return {"doc_id": doc_id, "category": category, "fields": fields}


def test_groups_multipage_and_dual_currency():
    docs = [
        _doc("multipage-2p-za-1", "multipage", {"subtotal", "tax", "total"}),
        _doc("currency-dual-usd-zar-1", "currency", {"subtotal", "currency"}),
        _doc("clean-us-clean-simple", "clean", set()),
    ]
    report = analyze(docs, {}, {})
    assert report["groups"]["multipage"]["docs"] == 1
    assert report["groups"]["multipage"]["field_misses"] == {"subtotal": 1, "tax": 1, "total": 1}
    assert report["groups"]["dual_currency"]["field_misses"] == {"currency": 1, "subtotal": 1}
    assert report["groups"]["shipping_discount"]["docs"] == 0
    assert report["ungrouped_field_misses"] == {}


def test_ungrouped_misses_are_counted():
    docs = [_doc("hard-wrong-vat-amount", "hard", {"tax"})]
    report = analyze(docs, {}, {})
    assert report["groups"]["multipage"]["docs"] == 0
    assert report["ungrouped_field_misses"] == {"tax": 1}


def test_naive_consistency_check_flags_adjustment_docs():
    docs = [
        _doc("clean-us-with-shipping", "clean", set()),
        _doc("hard-total-inflated", "hard", set()),
    ]
    preds = {
        "clean-us-with-shipping": {
            "subtotal": 100.0,
            "tax": 10.0,
            "total_amount": 130.0,
            "line_items": [{"amount": 100.0}],
        },
        "hard-total-inflated": {
            "subtotal": 100.0,
            "tax": 10.0,
            "total_amount": 140.0,
            "line_items": [{"amount": 100.0}],
        },
    }
    report = analyze(docs, preds, {"hard-total-inflated": True})
    check = report["naive_consistency_check"]
    assert check["fp"] == 1
    assert check["false_positives"] == ["clean-us-with-shipping"]
    assert check["tp"] == 1
    assert check["fn"] == 0
