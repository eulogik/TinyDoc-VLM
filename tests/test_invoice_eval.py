"""Unit tests for the invoice-eval normalization (no model calls)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evaluation" / "invoice"))
sys.path.insert(0, str(ROOT / "sdk"))

from run_invoice_eval import (
    _currency,
    _date,
    _doc_type,
    _load_resume_rows,
    _num,
    _read_image,
    _row_image_paths,
    manifest_is_multipage,
    multipage_prompt,
    normalize_prediction,
)
from tinydoc.pipeline import _extract_json


def test_num_parses_dot_decimal():
    assert _num("6702.90") == 6702.9
    assert _num(12) == 12.0
    assert _num("$1,234.56") == 1234.56


def test_num_parses_eu_decimal():
    assert _num("1.234,56") == 1234.56
    assert _num("1 234,56") == 1234.56
    assert _num("1,234") == 1234.0


def test_num_parses_negative_and_invalid():
    assert _num("(1,234.56)") == -1234.56
    assert _num("R -1 234,56") == -1234.56
    assert _num("") is None
    assert _num("n/a") is None
    assert _num(None) is None
    assert _num(True) is None


def test_doc_type_mapping():
    assert _doc_type("Credit Note") == "credit_note"
    assert _doc_type("credit-note") == "credit_note"
    assert _doc_type("statement") == "other"
    assert _doc_type("INVOICE") == "invoice"
    assert _doc_type(None) == "invoice"


def test_currency_normalization():
    assert _currency("NAD") == "NAD"
    assert _currency("N$") == "NAD"
    assert _currency("R") == "ZAR"
    assert _currency("€") == "EUR"
    assert _currency("£") == "GBP"
    assert _currency("K") == "ZMW"
    assert _currency("US$") == "USD"
    assert _currency("$") == "USD"
    assert _currency("R (ZAR)") == "ZAR"
    assert _currency("") is None
    assert _currency(None) is None


def test_date_normalization_resolves_ambiguity():
    assert _date("2024-03-12") == "2024-03-12"
    assert _date("30/04/2026") == "2026-04-30"
    assert _date("04/30/2026") == "2026-04-30"
    # Ambiguous: US documents/currency are month-first.
    assert _date("03/12/2024", "clean-us-clean-simple", "USD") == "2024-03-12"
    assert _date("03/12/2024", "clean-de-grid", "EUR") == "2024-12-03"
    assert _date("30.04.2026", "clean-de-grid", "EUR") == "2026-04-30"
    assert _date("", "x", "USD") is None
    assert _date(None, "x", "USD") is None


def test_normalize_prediction_aliases():
    pred = normalize_prediction(
        {
            "vendor": "Acme",
            "invoice_no": "INV-1",
            "invoice_date": "2026-04-30",
            "currency": "eur",
            "subtotal": "5.632,69",
            "tax": "1.070,21",
            "total": "6.702,90",
            "document_type": "Credit Note",
            "line_items": [{"description": "Bolt", "total": "1,50"}],
            "flagged": 1,
            "references": ["TR1"],
        }
    )
    assert pred["vendor_name"] == "Acme"
    assert pred["invoice_number"] == "INV-1"
    assert pred["currency"] == "EUR"
    assert pred["subtotal"] == 5632.69
    assert pred["tax"] == 1070.21
    assert pred["total_amount"] == 6702.9
    assert pred["document_type"] == "credit_note"
    assert pred["line_items"] == [{"description": "Bolt", "amount": 1.5}]
    assert pred["flagged"] is True
    assert pred["references"] == ["TR1"]


def test_extract_json_from_fenced_response():
    raw = 'Here is the result:\n```json\n{"vendor_name": "Acme", "total": 12.5}\n```'
    parsed = _extract_json(raw)
    assert parsed == {"vendor_name": "Acme", "total": 12.5}
    assert _extract_json("no json") is None


def test_load_resume_rows_drops_failures(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(
        '{"id": "a", "error": null, "latency_ms": 10}\n'
        '{"id": "b", "error": "URLError: refused", "latency_ms": 3}\n'
        '{"id": "c", "error": null, "latency_ms": 20}\n'
    )
    done, failed = _load_resume_rows(path)
    assert [r["id"] for r in done] == ["a", "c"]
    assert failed == 1
    assert _load_resume_rows(tmp_path / "missing.jsonl") == ([], 0)


def test_read_image_retries_transient_permission_error(tmp_path, monkeypatch):
    path = tmp_path / "doc.jpg"
    path.write_bytes(b"image")
    calls = {"n": 0}
    real = type(path).read_bytes

    def flaky(self):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError("Operation not permitted")
        return real(self)

    monkeypatch.setattr(type(path), "read_bytes", flaky)
    assert _read_image(path, attempts=3, delay=0) == b"image"
    assert calls["n"] == 3


def test_row_image_paths_prefers_list():
    assert _row_image_paths({"id": "a", "image_paths": ["p1.png", "p2.png"]}) == ["p1.png", "p2.png"]
    assert _row_image_paths({"id": "b", "image_path": "one.png"}) == ["one.png"]
    try:
        _row_image_paths({"id": "c"})
    except ValueError as exc:
        assert "no image path" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_multipage_prompt_mentions_all_pages():
    prompt = multipage_prompt()
    assert "every page" in prompt
    assert "document-wide totals" in prompt
    assert "Read the invoice image" not in prompt


def test_manifest_is_multipage():
    assert manifest_is_multipage([{"id": "a", "image_paths": ["p1.png"]}]) is False
    assert manifest_is_multipage([{"id": "a", "image_paths": ["p1.png", "p2.png"]}]) is True
    assert manifest_is_multipage([{"id": "a", "image_path": "one.png"}]) is False
