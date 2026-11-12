"""Unit tests for page-by-page invoice extraction and merge (no model calls)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evaluation" / "invoice"))

from run_invoice_pagewise import (
    is_summary_row,
    merge_is_sane,
    merge_pages,
    page_prompt,
    reconcile_totals,
    symbol_currency,
)


def _page(**kwargs):
    base = {
        "vendor_name": "",
        "invoice_number": "",
        "invoice_date": None,
        "due_date": None,
        "currency": None,
        "document_type": "invoice",
        "line_items": [],
        "subtotal": None,
        "tax": None,
        "total": None,
    }
    base.update(kwargs)
    return base


def test_merge_uses_first_header_and_final_totals():
    pages = [
        _page(
            vendor_name="Acme Ltd",
            invoice_number="INV-1",
            invoice_date="2026-02-22",
            due_date="2026-03-24",
            currency="USD",
            line_items=[{"description": "Paper", "amount": 10.0}],
        ),
        _page(
            vendor_name="Acme Ltd",
            invoice_number="INV-1",
            line_items=[
                {"description": "Ink", "amount": 20.0},
                {"description": "Carried forward", "amount": 30.0},
            ],
            subtotal=30.0,
            tax=3.0,
            total=33.0,
        ),
    ]
    merged = merge_pages(pages, "doc-us-1")
    assert merged["vendor_name"] == "Acme Ltd"
    assert merged["invoice_number"] == "INV-1"
    assert merged["invoice_date"] == "2026-02-22"
    assert merged["due_date"] == "2026-03-24"
    assert merged["currency"] == "USD"
    assert [line["description"] for line in merged["line_items"]] == ["Paper", "Ink"]
    assert merged["subtotal"] == 30.0
    assert merged["tax"] == 3.0
    assert merged["total_amount"] == 33.0


def test_missing_final_totals_fall_back_to_line_sum_and_arithmetic():
    pages = [
        _page(
            vendor_name="Acme Ltd",
            line_items=[{"description": "A", "amount": 12.5}],
        ),
        _page(line_items=[{"description": "B", "amount": 7.5}], tax=2.0),
    ]
    merged = merge_pages(pages, "doc")
    assert merged["subtotal"] == 20.0
    assert merged["tax"] == 2.0
    assert merged["total_amount"] == 22.0


def test_tax_is_derived_when_only_total_is_printed():
    pages = [
        _page(
            vendor_name="Acme Ltd",
            line_items=[{"description": "A", "amount": 100.0}],
            total=120.0,
        )
    ]
    merged = merge_pages(pages, "doc")
    assert merged["subtotal"] == 100.0
    assert merged["tax"] == 20.0
    assert merged["total_amount"] == 120.0


def test_summary_rows_are_not_line_items():
    assert is_summary_row("Carried forward")
    assert is_summary_row("Brought forward")
    assert is_summary_row("TOTAL")
    assert is_summary_row("VAT 15%") is True
    assert is_summary_row("Sales Tax (7%)") is True
    assert is_summary_row("Tax preparation service") is False
    assert is_summary_row("Page 2 of 8") is True
    assert is_summary_row("Copy paper A4") is False


def test_sanity_checks_do_not_use_gold():
    page = _page(vendor_name="Acme", total=10.0)
    merged = merge_pages([page], "doc")
    assert merge_is_sane(merged, [page]) == []
    empty = merge_pages([], "doc")
    assert "no page parsed" in merge_is_sane(empty, [])
    missing = merge_pages([_page(vendor_name="Acme")], "doc")
    assert "no total" in merge_is_sane(missing, [_page(vendor_name="Acme")])


def test_page_prompt_pins_the_fixed_rules():
    prompt = page_prompt(2, 8)
    assert "page 2 of 8" in prompt
    assert "line total" in prompt
    assert "never the unit price" in prompt
    assert "carried forward" in prompt
    assert "Never repeat items from another page" in prompt
    assert "JSON only" in prompt


def test_truncated_total_is_repaired_when_lines_match_subtotal():
    page = _page(
        vendor_name="Acme",
        line_items=[{"description": "A", "amount": 100.0}],
        subtotal=100.0,
        tax=20.0,
        total=12.0,
    )
    merged = merge_pages([page], "doc")
    assert merged["total_amount"] == 120.0
    assert merged["corrections"] == ["total_from_subtotal_tax"]


def test_total_is_not_repaired_when_lines_do_not_match_subtotal():
    page = _page(
        vendor_name="Acme",
        line_items=[{"description": "A", "amount": 90.0}],
        subtotal=100.0,
        tax=20.0,
        total=130.0,
    )
    merged = merge_pages([page], "doc")
    assert merged["total_amount"] == 130.0
    assert "corrections" not in merged


def test_reconcile_derives_tax_when_only_total_is_printed():
    merged, corrections = reconcile_totals(
        {
            "subtotal": 100.0,
            "tax": None,
            "total_amount": 120.0,
            "line_items": [{"description": "A", "amount": 100.0}],
        }
    )
    assert merged["tax"] == 20.0
    assert corrections == ["tax_from_total_subtotal"]


def test_subtotal_is_repaired_from_total_tax_when_lines_corroborate():
    page = _page(
        line_items=[{"description": "A", "amount": 100.0}],
        subtotal=100.29,
        tax=12.0,
        total=112.0,
    )
    merged = merge_pages([page], "doc")
    assert merged["subtotal"] == 100.0
    assert merged["total_amount"] == 112.0
    assert "subtotal_from_total_tax" in merged["corrections"]


def test_small_subtotal_residual_is_resolved_toward_total_minus_tax():
    page = _page(
        line_items=[{"description": "A", "amount": 16593.4}],
        subtotal=61209.42,
        tax=9181.37,
        total=70390.5,
    )
    merged = merge_pages([page], "doc")
    assert merged["subtotal"] == 61209.13
    assert merged["total_amount"] == 70390.5
    assert "subtotal_from_total_tax_residual" in merged["corrections"]


def test_large_subtotal_residual_is_not_hidden():
    page = _page(
        line_items=[{"description": "A", "amount": 100.0}],
        subtotal=2508.36,
        tax=762.56,
        total=846.32,
    )
    merged = merge_pages([page], "doc")
    assert merged["subtotal"] == 2508.36
    assert merged["total_amount"] == 846.32
    assert "corrections" not in merged


def test_tax_is_repaired_when_lines_corroborate_subtotal():
    page = _page(
        line_items=[{"description": "A", "amount": 175129.13}],
        subtotal=175129.13,
        tax=20000.0,
        total=201398.5,
    )
    merged = merge_pages([page], "doc")
    assert merged["tax"] == 26269.37
    assert merged["total_amount"] == 201398.5
    assert "tax_from_total_subtotal" in merged["corrections"]


def test_truncated_tax_is_repaired_without_line_support():
    page = _page(
        line_items=[{"description": "A", "amount": 174359.13}],
        subtotal=175129.13,
        tax=26.69,
        total=201398.5,
    )
    merged = merge_pages([page], "doc")
    assert merged["tax"] == 26269.37
    assert merged["total_amount"] == 201398.5
    assert "tax_from_total_subtotal_truncated" in merged["corrections"]


def test_normal_tax_is_not_rewritten():
    page = _page(
        line_items=[{"description": "A", "amount": 100.0}],
        subtotal=100.0,
        tax=20.0,
        total=120.0,
    )
    merged = merge_pages([page], "doc")
    assert merged["tax"] == 20.0
    assert "corrections" not in merged


def test_exact_large_totals_are_not_marked_as_corrections():
    page = _page(
        line_items=[{"description": "A", "amount": 91660.06}],
        subtotal=91660.06,
        tax=13749.01,
        total=105409.07,
    )
    merged = merge_pages([page], "doc")
    assert "corrections" not in merged


def test_currency_symbol_prompt_is_versioned():
    v1 = page_prompt(1, 2, version=1)
    v2 = page_prompt(1, 2, version=2)
    assert "currency_symbol" not in v1
    assert "N$ = NAD" not in v1
    assert "currency_symbol" in v2
    assert "N$ = NAD" in v2


def test_printed_symbol_overrides_a_wrong_currency_code():
    page = _page(
        vendor_name="Acme",
        currency="ZAR",
        currency_symbol="N$",
        line_items=[{"description": "A", "amount": 10.0}],
        subtotal=10.0,
        tax=0.0,
        total=10.0,
    )
    merged = merge_pages([page], "doc")
    assert merged["currency"] == "NAD"


def test_currency_prefers_the_final_page_symbol():
    pages = [
        _page(currency="ZAR", currency_symbol="R"),
        _page(currency="USD", currency_symbol="$"),
        _page(currency="NAD", currency_symbol="N$"),
    ]
    merged = merge_pages(pages, "doc")
    assert merged["currency"] == "NAD"


def test_symbol_map_is_exact():
    assert symbol_currency("N$") == "NAD"
    assert symbol_currency("R") == "ZAR"
    assert symbol_currency("$") == "USD"
    assert symbol_currency("€") == "EUR"
    assert symbol_currency("K") == "ZMW"
    assert symbol_currency("EUR") == "EUR"
    assert symbol_currency("NAD") == "NAD"
    assert symbol_currency("not-a-symbol") is None


def test_printed_total_is_preserved_on_tax_inclusive_receipt():
    # A Portuguese receipt prints a gross total and the VAT amount, not a net
    # subtotal. The old merge rewrote 32.00 as 32.00 + 5.98.
    page = _page(
        line_items=[{"description": "A", "amount": 32.0}],
        subtotal=32.0,
        tax=5.98,
        total=32.0,
    )
    merged = merge_pages([page], "doc")
    assert merged["total_amount"] == 32.0
    assert merged["tax"] == 5.98
    assert "corrections" not in merged or "total_from_subtotal_tax" not in merged["corrections"]


def test_derived_subtotal_does_not_rewrite_printed_tax():
    # The receipt prints total 10.00 and VAT 0.65 but no net subtotal. The line
    # sum is not a printed subtotal and must not drive a tax overwrite.
    page = _page(
        line_items=[{"description": "A", "amount": 6.98}],
        subtotal=None,
        tax=0.65,
        total=10.0,
    )
    merged = merge_pages([page], "doc")
    assert merged["tax"] == 0.65
    assert merged["total_amount"] == 10.0


def test_small_total_below_subtotal_is_not_treated_as_truncation():
    # A 0.10 discount/rounding is not a truncated total (the repair threshold is
    # the larger of 1.00 and 5% of the printed subtotal).
    page = _page(
        line_items=[{"description": "A", "amount": 13.3}],
        subtotal=13.3,
        tax=1.52,
        total=13.2,
    )
    merged = merge_pages([page], "doc")
    assert merged["total_amount"] == 13.2
