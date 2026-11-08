"""Unit tests for merging a targeted multipage retry (no model calls)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evaluation" / "invoice"))

from merge_multipage_runs import merge


def _manifest(*ids):
    return [{"id": doc_id, "image_paths": ["p1.png", "p2.png"]} for doc_id in ids]


def _row(doc_id, error=None, reason="stop"):
    return {"id": doc_id, "error": error, "done_reason": reason, "image_paths": ["p1.png", "p2.png"]}


def _preds(*ids):
    return {"system": "eval", "run_date": "2026-11-08", "documents": {i: {"total_amount": 1} for i in ids}}


def test_failed_row_is_replaced_by_successful_retry():
    merged = merge(
        _manifest("a", "b"),
        [_row("a"), _row("b", error="length")],
        [_row("b")],
        _preds("a", "b"),
        _preds("b"),
    )
    assert merged["report"]["ok"] is True
    assert merged["report"]["replaced"] == ["b"]
    assert merged["rows"][1]["error"] is None
    assert merged["predictions"]["page_mode"] == "all_pages"
    assert "merged" in merged["predictions"]["notes"]


def test_remaining_error_blocks_the_merge():
    merged = merge(
        _manifest("a", "b"),
        [_row("a"), _row("b", error="length")],
        [_row("b", error="still bad")],
        _preds("a", "b"),
        _preds("b"),
    )
    assert merged["report"]["ok"] is False
    assert merged["report"]["remaining_errors"] == ["b"]
    assert merged["report"]["retry_errors"] == ["b"]


def test_missing_prediction_is_reported():
    merged = merge(
        _manifest("a", "b"),
        [_row("a"), _row("b")],
        [],
        _preds("a"),
        _preds(),
    )
    assert merged["report"]["ok"] is False
    assert merged["report"]["missing_predictions"] == ["b"]
