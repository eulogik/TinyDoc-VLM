"""Smoke test for the optional local UI (skipped when Gradio is absent)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SDK = ROOT / "sdk"
if str(SDK) not in sys.path:
    sys.path.insert(0, str(SDK))

gradio = pytest.importorskip("gradio")


def test_build_app_constructs_blocks():
    from tinydoc.ui import build_app

    app = build_app()
    assert isinstance(app, gradio.Blocks)


def test_ui_documents_local_and_measured_boundaries():
    from tinydoc import ui

    assert "runs on **your machine**" in ui.HEADER
    assert "pipeline result, not a pure VLM" in ui.HEADER
    assert "degraded photo/scan" in ui.HEADER
    assert "Real supplier invoices are unmeasured" in ui.HEADER
    assert "hosted" in ui.FOOTER


def test_invoice_flag_markdown_includes_ocr_state_and_corrections():
    from types import SimpleNamespace

    from tinydoc.ui import _invoice_flags

    result = SimpleNamespace(
        ocr_totals={"consistent": True},
        review_flags=["line_sum_subtotal"],
        arithmetic_issues=[
            {
                "check": "line_sum_subtotal",
                "expected": 100.0,
                "computed": 90.0,
                "difference": -10.0,
            }
        ],
        corrections=["totals_from_ocr:total"],
    )
    text = _invoice_flags(result)
    assert "OCR printed totals: **consistent**" in text
    assert "line_sum_subtotal" in text
    assert "totals_from_ocr:total" in text
