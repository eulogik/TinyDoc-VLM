"""Unit tests for rebuilding page-wise predictions from saved raws (no model calls)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evaluation" / "invoice"))

from rebuild_pagewise import rebuild_row


def _page(page, payload, error=None):
    return {
        "page": page,
        "raw": json.dumps(payload),
        "error": error,
    }


def test_rebuild_repairs_truncated_total_from_saved_raws():
    row = {
        "id": "doc-1",
        "pages": [
            _page(
                1,
                {
                    "vendor_name": "Acme",
                    "invoice_number": "A-1",
                    "line_items": [{"description": "Paper", "amount": 100.0}],
                    "subtotal": None,
                    "tax": None,
                    "total": None,
                },
            ),
            _page(
                2,
                {
                    "line_items": [{"description": "Ink", "amount": 20.0}],
                    "subtotal": 120.0,
                    "tax": 12.0,
                    "total": 13.2,
                },
            ),
        ],
    }
    rebuilt = rebuild_row(row)
    assert rebuilt["error"] is None
    assert rebuilt["pred"]["subtotal"] == 120.0
    assert rebuilt["pred"]["total_amount"] == 132.0
    assert rebuilt["pred"]["corrections"] == ["total_from_subtotal_tax"]


def test_rebuild_keeps_page_errors_visible():
    row = {
        "id": "doc-2",
        "pages": [
            _page(1, {"vendor_name": "Acme", "line_items": [], "total": 10.0}),
            _page(2, {}, error="no JSON object"),
        ],
    }
    rebuilt = rebuild_row(row)
    assert "page 2: no JSON object" in rebuilt["error"]
