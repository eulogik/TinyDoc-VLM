"""Unit tests for OCR total parsing and application (no OCR engine calls)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evaluation" / "invoice"))

from totals_ocr import (
    apply_totals,
    parse_amount,
    parse_totals_lines,
    totals_are_consistent,
)


def test_space_thousands_separator():
    assert parse_amount("R25 083.76") == 25083.76
    assert parse_amount("N$61 209.13") == 61209.13
    assert parse_amount("R 1 234") == 1234.0


def test_comma_and_dot_conventions():
    assert parse_amount("25,083.76") == 25083.76
    assert parse_amount("25.083,76") == 25083.76
    assert parse_amount("1,234") == 1234.0
    assert parse_amount("12,5") == 12.5
    assert parse_amount("not a number") is None


def test_parse_totals_skips_tax_rate_and_uses_labels():
    lines = [
        "Code Description Qty Price Total",
        "BRA663 Brake pads 4 576.57 2 306.28",
        "Subtotal R25 083.76",
        "VAT 15% R3 762.56",
        "TOTAL R28 846.32",
        "Page 2 of 2",
    ]
    totals = parse_totals_lines(lines)
    assert totals["subtotal"] == 25083.76
    assert totals["tax"] == 3762.56
    assert totals["total"] == 28846.32
    assert totals_are_consistent(totals)


def test_currency_symbol_is_captured_from_the_subtotal_line():
    totals = parse_totals_lines(
        [
            "Subtotal N$61 209.13",
            "VAT 15% N$9 181.37",
            "TOTAL N$70 390.50",
        ]
    )
    assert totals["currency_symbol"] == "N$"
    assert totals_are_consistent(totals)


def test_ocr_confusable_leading_one_is_recovered():
    lines = [
        "Subtotal RI75 129.13",
        "VAT 15% R26 269.37",
        "TOTAL R201 398.50",
    ]
    totals = parse_totals_lines(lines)
    assert totals["subtotal"] == 175129.13
    assert totals["tax"] == 26269.37
    assert totals["total"] == 201398.50
    assert totals_are_consistent(totals)


def test_german_labels_and_trailing_currency_code():
    totals = parse_totals_lines(
        [
            "Subtotal 144.203,59 EUR",
            "MwSt 19% 27.398,68 EUR",
            "TOTAL 171.602,27 EUR",
        ]
    )
    assert totals["subtotal"] == 144203.59
    assert totals["tax"] == 27398.68
    assert totals["total"] == 171602.27
    assert totals["currency_symbol"] == "EUR"
    assert totals_are_consistent(totals)


def test_currency_code_before_amount_is_captured():
    totals = parse_totals_lines(
        [
            "Subtotal NAD 188 667.92",
            "VAT 15% NAD 28 300.19",
            "TOTAL NAD 216 968.11",
        ]
    )
    assert totals["currency_symbol"] == "NAD"
    assert totals_are_consistent(totals)


def test_parenthesised_negative_amounts():
    assert parse_amount("(1,234.00)") == -1234.0


def test_non_numeric_current_value_is_replaced():
    pred = {"subtotal": "n/a", "tax": None, "total_amount": 132.0}
    totals = {"subtotal": 120.0, "tax": 12.0, "total": 132.0, "consistent": True}
    updated, corrections = apply_totals(pred, totals)
    assert updated["subtotal"] == 120.0
    assert updated["tax"] == 12.0
    assert updated["total_amount"] == 132.0
    assert corrections == ["totals_from_ocr:subtotal+tax"]


def test_totals_are_consistent_rejects_bad_arithmetic_and_absurd_tax():
    assert not totals_are_consistent({"subtotal": 100.0, "tax": 20.0, "total": 130.0})
    assert not totals_are_consistent({"subtotal": 100.0, "tax": 80.0, "total": 180.0})
    assert not totals_are_consistent({"subtotal": 100.0, "tax": None, "total": 120.0})


def test_apply_totals_replaces_digit_truncated_amounts():
    pred = {
        "subtotal": 2508.36,
        "tax": 762.56,
        "total_amount": 846.32,
        "line_items": [{"description": "a", "amount": 10.0}],
    }
    totals = {
        "subtotal": 25083.76,
        "tax": 3762.56,
        "total": 28846.32,
        "consistent": True,
    }
    updated, corrections = apply_totals(pred, totals)
    assert updated["subtotal"] == 25083.76
    assert updated["tax"] == 3762.56
    assert updated["total_amount"] == 28846.32
    assert corrections == ["totals_from_ocr:subtotal+tax+total"]


def test_apply_totals_replaces_a_cent_level_model_misread():
    pred = {"subtotal": 61209.42, "tax": 9181.37, "total_amount": 70390.5}
    totals = {
        "subtotal": 61209.13,
        "tax": 9181.37,
        "total": 70390.5,
        "consistent": True,
    }
    updated, corrections = apply_totals(pred, totals)
    assert updated["subtotal"] == 61209.13
    assert corrections == ["totals_from_ocr:subtotal"]


def test_apply_totals_leaves_consistent_values_alone():
    pred = {"subtotal": 100.0, "tax": 20.0, "total_amount": 120.0}
    totals = {"subtotal": 100.0, "tax": 20.0, "total": 120.0, "consistent": True}
    updated, corrections = apply_totals(pred, totals)
    assert updated == pred
    assert corrections == []


def test_apply_totals_is_a_noop_when_ocr_is_inconsistent():
    pred = {"subtotal": 1.0, "tax": 0.1, "total_amount": 1.1}
    totals = {"subtotal": 100.0, "tax": 20.0, "total": 130.0, "consistent": False}
    updated, corrections = apply_totals(pred, totals)
    assert updated == pred
    assert corrections == []
