"""Unit tests for the page-wise row audit (no model calls)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evaluation" / "invoice"))

from audit_pagewise_rows import audit_rows


def _manifest(*pages):
    return [
        {"id": f"doc-{i}", "image_paths": [f"p{j}.png" for j in range(n)]}
        for i, n in enumerate(pages)
    ]


def _row(doc_id, pages, prompt=1000, generated=500, reason="stop", error=None):
    return {
        "id": doc_id,
        "error": error,
        "pred": {"line_items": [{"description": "a", "amount": 1.0}]},
        "pages": [
            {
                "page": j + 1,
                "prompt_eval_count": prompt,
                "eval_count": generated,
                "done_reason": reason,
                "error": None,
            }
            for j in range(pages)
        ],
    }


def test_clean_pagewise_run_passes():
    report = audit_rows(_manifest(2, 3), [_row("doc-0", 2), _row("doc-1", 3)])
    assert report["ok"] is True
    assert report["pages"] == 5
    assert report["issues"] == []


def test_missing_page_and_length_cut_are_flagged():
    report = audit_rows(
        _manifest(2, 3),
        [_row("doc-0", 1), _row("doc-1", 3, prompt=7000, generated=2048, reason="length")],
        num_ctx=8192,
        num_predict=2048,
    )
    joined = " ".join(report["issues"])
    assert report["ok"] is False
    assert "1 page results" in joined
    assert "done_reason='length'" in joined
    assert ">= budget" in joined


def test_visible_failed_row_can_be_allowed():
    rows = [
        _row("doc-0", 2),
        _row("doc-1", 2, error="page 2: no JSON object"),
    ]
    blocked = audit_rows(_manifest(2, 2), rows)
    allowed = audit_rows(_manifest(2, 2), rows, allowed_failures=1)
    assert blocked["ok"] is False
    assert blocked["failed_rows"][0]["id"] == "doc-1"
    assert allowed["ok"] is True
