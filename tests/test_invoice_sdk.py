"""Tests for the multipage invoice SDK path and parity with the evaluation run.

No model calls: a fake engine returns saved/canned page JSON, and the merge
parity test replays the committed page-wise raw rows through both the SDK and
the evaluation merge. If those diverge, the published 10-document result no
longer describes the shipped product path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SDK = ROOT / "sdk"
EVAL = ROOT / "evaluation" / "invoice"
for path in (SDK, EVAL):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tinydoc.invoice import (
    InvoicePipeline,
    check_prediction,
    invoice_schema_is_valid,
    merge_pages,
    page_prompt,
    reconcile_totals,
)
from tinydoc.pipeline import BaseEngine, _extract_json


class FakeEngine(BaseEngine):
    name = "fake"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate(self, prompt, image_path, num_predict=2048, **kwargs):
        self.calls.append((prompt, image_path, num_predict))
        index = len(self.calls) - 1
        raw = self.responses[index] if index < len(self.responses) else "{}"
        return {"raw": raw, "done_reason": "stop", "eval_count": 1}

    def extract_fields(self, image_path):  # pragma: no cover - not used
        return {}


def _png(path: Path, color=(255, 255, 255)) -> str:
    Image.new("RGB", (60, 80), color).save(path)
    return str(path)


def test_prompt_parity_with_evaluation_module():
    from run_invoice_pagewise import page_prompt as eval_prompt

    for page, total in ((1, 1), (2, 5), (8, 8)):
        assert eval_prompt(page, total) == page_prompt(page, total)
        assert eval_prompt(page, total, 2) == page_prompt(page, total, 2)


def test_sdk_merge_reproduces_the_measured_pagewise_predictions():
    results = ROOT / "evaluation" / "invoice" / "results"
    rows_path = results / "rows_invoice_qwen3-vl_2b_mp_pagewise.jsonl"
    if not rows_path.exists():  # pragma: no cover - committed in the repo
        pytest.skip("page-wise rows artifact not present")

    from run_invoice_pagewise import merge_pages as eval_merge

    rows = [
        json.loads(line)
        for line in rows_path.read_text().splitlines()
        if line.strip()
    ]
    assert rows
    for row in rows:
        pages = [
            parsed
            for page in row["pages"]
            if (parsed := _extract_json(page.get("raw") or "")) is not None
        ]
        sdk = merge_pages(pages, row["id"])
        evaluation = eval_merge(pages, row["id"])
        assert sdk == evaluation, row["id"]


def test_single_page_pipeline_merges_and_validates(tmp_path):
    image = _png(tmp_path / "invoice.png")
    engine = FakeEngine(
        [
            json.dumps(
                {
                    "vendor_name": "Acme Supply",
                    "invoice_number": "INV-001",
                    "invoice_date": "2026-11-01",
                    "due_date": None,
                    "currency": "USD",
                    "document_type": "invoice",
                    "line_items": [
                        {"description": "Paper", "amount": 100.0},
                        {"description": "Tax", "amount": 15.0},
                    ],
                    "subtotal": 100.0,
                    "tax": 15.0,
                    "total": 115.0,
                }
            )
        ]
    )
    result = InvoicePipeline(engine).extract(
        image, with_ocr_totals=False, with_evidence=False
    )
    assert result.page_count == 1
    assert result.fields["vendor_name"] == "Acme Supply"
    assert result.fields["subtotal"] == 100.0
    assert result.fields["total_amount"] == 115.0
    assert result.line_items == [
        {"description": "Paper", "amount": 100.0},
    ]
    assert result.schema_valid is True
    assert result.review_flags == []
    assert result.corrections == []
    assert result.pages[0]["done_reason"] == "stop"
    assert engine.calls[0][2] == 2048
    assert "page 1 of 1" in engine.calls[0][0]


def test_multipage_merge_uses_first_header_and_final_totals(tmp_path):
    page1 = _png(tmp_path / "p1.png")
    page2 = _png(tmp_path / "p2.png")
    engine = FakeEngine(
        [
            json.dumps(
                {
                    "vendor_name": "Acme Supply",
                    "invoice_number": "INV-002",
                    "invoice_date": "2026-11-02",
                    "currency": "ZAR",
                    "document_type": "invoice",
                    "line_items": [
                        {"description": "Paper", "amount": 100.0},
                        {"description": "Carried forward", "amount": 100.0},
                    ],
                    "subtotal": None,
                    "tax": None,
                    "total": None,
                }
            ),
            json.dumps(
                {
                    "vendor_name": "",
                    "invoice_number": "",
                    "invoice_date": None,
                    "currency": "ZAR",
                    "document_type": "invoice",
                    "line_items": [{"description": "Ink", "amount": 50.0}],
                    "subtotal": 150.0,
                    "tax": 22.5,
                    "total": 172.5,
                }
            ),
        ]
    )
    result = InvoicePipeline(engine).extract(
        page1,
        images=[page1, page2],
        with_ocr_totals=False,
        with_evidence=False,
    )
    assert result.page_count == 2
    assert result.fields["vendor_name"] == "Acme Supply"
    assert result.fields["subtotal"] == 150.0
    assert result.fields["total_amount"] == 172.5
    assert [item["description"] for item in result.line_items] == ["Paper", "Ink"]
    assert result.review_flags == []
    assert "page 2 of 2" in engine.calls[1][0]


def test_ocr_totals_are_applied_with_a_recorded_correction(tmp_path, monkeypatch):
    image = _png(tmp_path / "invoice.png")
    engine = FakeEngine(
        [
            json.dumps(
                {
                    "vendor_name": "Acme",
                    "invoice_number": "INV-003",
                    "currency": "NAD",
                    "document_type": "invoice",
                    "line_items": [{"description": "Paper", "amount": 100.0}],
                    "subtotal": 2508.36,
                    "tax": 762.56,
                    "total": 846.32,
                }
            )
        ]
    )

    monkeypatch.setattr(
        "tinydoc.invoice.read_totals",
        lambda path: {
            "subtotal": 25083.76,
            "tax": 3762.56,
            "total": 28846.32,
            "currency_symbol": "N$",
            "consistent": True,
            "evidence": {
                "subtotal": "Subtotal N$25 083.76",
                "tax": "VAT N$3 762.56",
                "total": "Total N$28 846.32",
            },
        },
    )
    result = InvoicePipeline(engine).extract(
        image, with_ocr_totals=True, with_evidence=True
    )
    assert result.fields["subtotal"] == 25083.76
    assert result.fields["tax"] == 3762.56
    assert result.fields["total_amount"] == 28846.32
    assert result.fields["currency"] == "NAD"
    assert "totals_from_ocr:subtotal+tax+total" in result.corrections
    assert result.ocr_totals["consistent"] is True
    assert result.evidence["total"]["kind"] == "ocr_line"


def test_pdf_pages_are_rendered(tmp_path):
    pytest.importorskip("pypdfium2")
    first = Image.new("RGB", (200, 300), "white")
    second = Image.new("RGB", (200, 300), "grey")
    pdf_path = tmp_path / "two-pages.pdf"
    first.save(pdf_path, save_all=True, append_images=[second])

    engine = FakeEngine(
        [
            json.dumps({"vendor_name": "A", "line_items": [], "total": 10.0}),
            json.dumps({"vendor_name": "A", "line_items": [], "total": 10.0}),
        ]
    )
    result = InvoicePipeline(engine).extract(
        pdf_path, with_ocr_totals=False, with_evidence=False
    )
    assert result.page_count == 2
    assert "page 2 of 2" in engine.calls[1][0]


def test_engine_without_generate_fails_loudly(tmp_path):
    image = _png(tmp_path / "invoice.png")

    class NoGenerate(BaseEngine):
        name = "ocr_regex"

        def extract_fields(self, image_path):
            return {}

    with pytest.raises(RuntimeError, match="cannot run invoice page prompts"):
        InvoicePipeline(NoGenerate()).extract(
            image, with_ocr_totals=False, with_evidence=False
        )


def test_reconcile_and_flags_are_internal_only():
    merged = {
        "subtotal": 100.0,
        "tax": 15.0,
        "total_amount": 116.0,
        "line_items": [{"description": "Paper", "amount": 100.0}],
    }
    repaired, corrections = reconcile_totals(dict(merged))
    assert repaired["tax"] == 16.0
    assert "tax_from_total_subtotal" in corrections

    issues = check_prediction(merged)
    assert [issue["check"] for issue in issues] == ["subtotal_tax_total"]


def test_invoice_schema_rejects_wrong_types():
    good = {
        "vendor_name": "A",
        "invoice_number": "1",
        "invoice_date": "2026-11-01",
        "due_date": None,
        "currency": "USD",
        "subtotal": 1.0,
        "tax": 0.0,
        "total_amount": 1.0,
        "document_type": "invoice",
        "line_items": [{"description": "x", "amount": 1.0}],
    }
    assert invoice_schema_is_valid(good) is True
    bad = dict(good)
    bad["line_items"] = [{"description": "x", "amount": "one"}]
    assert invoice_schema_is_valid(bad) is False


def test_evaluation_ocr_module_is_the_sdk_module():
    import tinydoc.ocr_totals as sdk_ocr
    import totals_ocr

    assert totals_ocr.read_totals is sdk_ocr.read_totals
    assert totals_ocr.parse_totals_lines is sdk_ocr.parse_totals_lines
    assert totals_ocr.apply_totals is sdk_ocr.apply_totals
