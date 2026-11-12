"""Tests for the real Portuguese invoice/receipt manifest builder (no model calls)."""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "evaluation" / "invoice"
if str(EVAL) not in sys.path:
    sys.path.insert(0, str(EVAL))

from build_pt_manifest import build_row, deduplicate, parse_amount, parse_date
from prepare_pt_probe import prepare_image


def _image(root: Path, name: str = "doc.jpg") -> Path:
    path = root / name
    Image.new("RGB", (60, 80), "white").save(path)
    return path


def test_parse_amount_accepts_portuguese_and_dot_decimals():
    assert parse_amount("1.234,56") == 1234.56
    assert parse_amount("13,00") == 13.0
    assert parse_amount("6.950") == 6.95
    assert parse_amount("4,750") == 4.75
    assert parse_amount("0.404") == 0.404
    assert parse_amount(" 2.40 ") == 2.4
    assert parse_amount("") is None
    assert parse_amount(None) is None


def test_parse_date_is_day_first_unless_year_is_first():
    assert parse_date("23.01.19") == "2019-01-23"
    assert parse_date("04-01-2021") == "2021-01-04"
    assert parse_date("2019/03/26") == "2019-03-26"
    assert parse_date("2019-01-23") == "2019-01-23"
    assert parse_date("19/07/2019") == "2019-07-19"
    assert parse_date("not a date") is None


def test_build_row_maps_fields_and_keeps_private_fields_out(tmp_path):
    image = _image(tmp_path)
    raw = {
        "_id": "20210318_172420",
        "company": "carvalho, batista, lda",
        "date": "2019-01-30",
        "address": "rua privada 1",
        "total": "179,17",
        "invoice_number": "FT 2019A17/679",
        "nif_buyer": "510776914",
        "nif_seller": "500537380",
        "iva_amount": "33,50",
    }
    row = build_row(raw, image, "company", tmp_path)
    expected = row["expected"]
    assert expected["vendor_name"] == "carvalho, batista, lda"
    assert expected["invoice_number"] == "FT 2019A17/679"
    assert expected["invoice_date"] == "2019-01-30"
    assert expected["total"] == 179.17
    assert expected["tax"] == 33.5
    assert expected["currency"] == "EUR"
    assert expected["scored"] == [
        "vendor_name",
        "invoice_number",
        "invoice_date",
        "currency",
        "tax",
        "total",
    ]
    assert row["annotation_issues"] == []
    assert "nif_buyer" not in row["annotation_raw"]
    assert "nif_seller" not in row["annotation_raw"]
    assert "address" not in row["annotation_raw"]


def test_build_row_excludes_implausible_tax_and_missing_fields(tmp_path):
    image = _image(tmp_path)
    raw = {
        "_id": "x",
        "company": "",
        "date": "not a date",
        "total": "6,00",
        "invoice_number": "",
        "iva_amount": "4,88",
    }
    row = build_row(raw, image, "company", tmp_path)
    expected = row["expected"]
    assert "vendor_name" not in expected["scored"]
    assert "invoice_number" not in expected["scored"]
    assert "invoice_date" not in expected["scored"]
    assert "tax" not in expected["scored"]
    assert expected["scored"] == ["currency", "total"]
    assert "implausible_tax" in row["annotation_issues"]
    assert "missing_vendor" in row["annotation_issues"]
    assert "missing_invoice_number" in row["annotation_issues"]
    assert "unparseable_date" in row["annotation_issues"]


def _row(doc_id: str, scored: list[str], total: float = 10.0, vendor: str = "loja") -> dict:
    return {
        "id": doc_id,
        "expected": {
            "vendor_name": vendor,
            "invoice_number": "FT 1",
            "invoice_date": "2019-01-01",
            "total": total,
            "scored": scored,
        },
        "_bytes": 100,
    }


def test_prepare_image_applies_product_cap_and_records_it(tmp_path):
    source = tmp_path / "source.jpg"
    Image.new("RGB", (200, 300), "white").save(source, quality=95)
    destination = tmp_path / "prepared.jpg"
    info = prepare_image(source, destination, max_px=100, quality=90)
    assert info["source_width"] == 200
    assert info["source_height"] == 300
    assert info["width"] == 67
    assert info["height"] == 100
    assert info["resized"] is True
    with Image.open(destination) as image:
        assert image.size == (67, 100)


def test_prepare_image_copies_when_already_small(tmp_path):
    source = tmp_path / "source.jpg"
    Image.new("RGB", (60, 80), "white").save(source, quality=95)
    destination = tmp_path / "prepared.jpg"
    info = prepare_image(source, destination, max_px=1400, quality=90)
    assert info["resized"] is False
    assert destination.read_bytes() == source.read_bytes()


def test_deduplicate_keeps_richest_copy_and_reports_drop():
    rows = [
        _row("ptreal-company-a", ["currency", "total"]),
        _row("ptreal-company-b", ["vendor_name", "invoice_number", "invoice_date", "currency", "total"]),
        _row("ptreal-company-c", ["currency", "total"], total=11.0),
    ]
    kept, dropped = deduplicate(rows)
    assert [row["id"] for row in kept] == ["ptreal-company-b", "ptreal-company-c"]
    assert dropped == [
        {
            "id": "ptreal-company-a",
            "duplicate_of": "ptreal-company-b",
            "business_key": ["loja", "ft 1", "2019-01-01", 10.0],
        }
    ]
