"""Unit tests for the multipage row audit (no model calls)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evaluation" / "invoice"))

from audit_multipage_rows import audit_rows


def _manifest(*pages):
    return [{"id": f"doc-{i}", "image_paths": [f"p{j}.png" for j in range(n)]} for i, n in enumerate(pages)]


def _row(doc_id, pages, prompt=3000, generated=800, reason="stop", error=None):
    return {
        "id": doc_id,
        "image_paths": [f"p{j}.png" for j in range(pages)],
        "prompt_eval_count": prompt,
        "eval_count": generated,
        "done_reason": reason,
        "error": error,
    }


def test_clean_multipage_run_passes():
    report = audit_rows(_manifest(2, 3), [_row("doc-0", 2), _row("doc-1", 3)])
    assert report["ok"] is True
    assert report["pages"] == 5
    assert report["issues"] == []


def test_missing_pages_context_cut_and_length_are_flagged():
    report = audit_rows(
        _manifest(2, 3),
        [
            _row("doc-0", 1),
            _row("doc-1", 3, prompt=12000, generated=5000, reason="length"),
        ],
    )
    joined = " ".join(report["issues"])
    assert report["ok"] is False
    assert "sent 1 images" in joined
    assert "done_reason='length'" in joined
    assert "> context" in joined


def test_missing_and_extra_rows_are_flagged():
    report = audit_rows(_manifest(2, 2), [_row("doc-0", 2), _row("doc-9", 2)])
    joined = " ".join(report["issues"])
    assert "doc-1: missing row" in joined
    assert "doc-9: row not in manifest" in joined


def test_per_row_context_overrides_default():
    rows = [_row("doc-0", 2, prompt=12000, generated=6000, reason="stop")]
    for row in rows:
        row["num_ctx"] = 24576
        row["num_predict"] = 8192
    report = audit_rows(_manifest(2), rows, num_ctx=8192, num_predict=4096)
    assert report["ok"] is True
    assert report["num_ctx_values"] == [24576]


def test_visible_failed_rows_can_be_allowed():
    rows = [
        _row("doc-0", 2),
        _row("doc-1", 2, prompt=12000, generated=8192, reason="length", error="no JSON"),
    ]
    blocked = audit_rows(_manifest(2, 2), rows, num_ctx=16384, num_predict=8192)
    allowed = audit_rows(
        _manifest(2, 2), rows, num_ctx=16384, num_predict=8192, allowed_failures=1
    )
    assert blocked["ok"] is False
    assert blocked["failed_rows"][0]["id"] == "doc-1"
    assert blocked["issues"] == []
    assert allowed["ok"] is True
