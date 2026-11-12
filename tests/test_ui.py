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
    assert "Real supplier invoices are unmeasured" in ui.HEADER
    assert "hosted" in ui.FOOTER
