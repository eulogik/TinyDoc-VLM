"""Unit tests for page-by-page invoice extraction and merge (no model calls)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evaluation" / "invoice"))

from run_invoice_pagewise import is_summary_row, merge_is_sane, merge_pages


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
    assert is_summary_row("VAT 15%") is False  # not an exact summary label
    assert is_summary_row("Sales Tax") is True
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
