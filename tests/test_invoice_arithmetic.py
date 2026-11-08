"""Unit tests for deterministic invoice arithmetic checks (no model calls)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evaluation" / "invoice"))

from arithmetic_checks import check_prediction, flag_predictions


def test_consistent_invoice_has_no_issues():
    pred = {
        "subtotal": 100.0,
        "tax": 20.0,
        "total_amount": 120.0,
        "line_items": [{"description": "a", "amount": 60.0}, {"description": "b", "amount": 40.0}],
    }
    assert check_prediction(pred) == []


def test_tax_total_mismatch_fires():
    pred = {"subtotal": 100.0, "tax": 20.0, "total_amount": 130.0}
    issues = check_prediction(pred)
    assert [issue["check"] for issue in issues] == ["subtotal_tax_total"]
    assert issues[0]["difference"] == -10.0


def test_line_sum_mismatch_fires():
    pred = {
        "subtotal": 100.0,
        "tax": 20.0,
        "total_amount": 120.0,
        "line_items": [{"description": "a", "amount": 60.0}, {"description": "b", "amount": 30.0}],
    }
    issues = check_prediction(pred)
    assert [issue["check"] for issue in issues] == ["line_sum_subtotal"]
    assert issues[0]["difference"] == -10.0


def test_rounding_is_within_tolerance():
    pred = {"subtotal": 25083.76, "tax": 3762.56, "total_amount": 28846.32}
    assert check_prediction(pred) == []


def test_missing_numbers_do_not_fire():
    assert check_prediction({"subtotal": None, "tax": 2.0, "total_amount": 2.0}) == []
    assert check_prediction({"subtotal": "n/a", "tax": "1.0", "total_amount": "1.0"}) == []


def test_credit_note_signs_are_respected():
    pred = {
        "subtotal": -100.0,
        "tax": -20.0,
        "total_amount": -120.0,
        "line_items": [{"description": "refund", "amount": -100.0}],
    }
    assert check_prediction(pred) == []


def test_flag_predictions_copies_and_marks_only_bad_docs():
    flagged, report = flag_predictions(
        {
            "good": {"subtotal": 1.0, "tax": 0.1, "total_amount": 1.1},
            "bad": {"subtotal": 1.0, "tax": 0.1, "total_amount": 2.1},
        }
    )
    assert flagged["good"].get("flagged") is None
    assert flagged["bad"]["flagged"] is True
    assert flagged["bad"]["flag_reasons"] == ["subtotal_tax_total"]
    assert report["flagged"] == ["bad"]
    assert report["n"] == 2
