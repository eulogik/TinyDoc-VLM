"""Unit tests for independently recomputing DocVQA probe scores (no model calls)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evaluation" / "docvqa"))

import summarize_probe
from run_docvqa_probe import anls_score, normalize_answer


def _row(answer, target, latency=100.0, error=None):
    return {
        "answer": answer,
        "target": target,
        "anls": anls_score(answer, target),
        "exact": normalize_answer(answer) == normalize_answer(target),
        "latency_ms": latency,
        "error": error,
    }


def test_recompute_matches_stored_rows():
    rows = [
        _row("0.28", "0.28", 100.0),
        _row("March 12, 1975", "March 12, 1975", 300.0),
        _row("banana", "0.28", 200.0, error="timeout"),
    ]
    out = summarize_probe.summarize_rows(rows)
    assert out["n"] == 3
    assert out["anls_mean"] == 2 / 3
    assert out["exact_rate"] == 2 / 3
    assert out["errors"] == 1
    assert out["p50_latency_ms"] == 200.0
    assert out["stored_anls_mismatches"] == 0


def test_edited_anls_is_detected(tmp_path, monkeypatch):
    row = _row("0.28", "0.28")
    row["anls"] = 0.0
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text(json.dumps(row) + "\n")
    out_path = tmp_path / "summary.json"

    monkeypatch.setattr(
        sys,
        "argv",
        ["summarize_probe.py", "--rows", str(rows_path), "--out", str(out_path)],
    )
    assert summarize_probe.main() == 1
    assert not out_path.exists()

    report = summarize_probe.summarize_rows([row])
    assert report["stored_anls_mismatches"] == 1
    assert report["max_stored_anls_drift"] == 1.0
