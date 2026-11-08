#!/usr/bin/env python3
"""Recompute DocVQA reading-probe scores from per-question rows.

The runner writes both rows and a scores JSON. This script rebuilds ANLS, exact
match, error count, and latency percentiles from the rows alone and checks the
stored per-row ANLS, so a hand-edited score cannot survive the audit.

Usage:
    python evaluation/docvqa/summarize_probe.py \
        --rows evaluation/docvqa/results/rows_docvqa_qwen3-vl_2b_instruct.jsonl \
        --out evaluation/docvqa/results/summary_docvqa_qwen3-vl_2b_instruct.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_docvqa_probe import anls_score, normalize_answer

TOL = 1e-9


def summarize_rows(rows: list[dict]) -> dict:
    answers = [str(row.get("answer", "")) for row in rows]
    targets = [str(row.get("target", "")) for row in rows]
    anls = [anls_score(answer, target) for answer, target in zip(answers, targets)]
    exact = [
        normalize_answer(answer) == normalize_answer(target)
        for answer, target in zip(answers, targets)
    ]
    latencies = [float(row.get("latency_ms", 0.0)) for row in rows]

    stored = [float(row.get("anls", 0.0)) for row in rows]
    drift = [abs(a - b) for a, b in zip(anls, stored) if abs(a - b) > TOL]

    return {
        "n": len(rows),
        "anls_mean": statistics.mean(anls) if anls else 0.0,
        "exact_rate": statistics.mean(exact) if exact else 0.0,
        "errors": sum(1 for row in rows if row.get("error")),
        "p50_latency_ms": statistics.median(latencies) if latencies else 0.0,
        "mean_latency_ms": statistics.mean(latencies) if latencies else 0.0,
        "stored_anls_mismatches": len(drift),
        "max_stored_anls_drift": max(drift) if drift else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows_path = Path(args.rows)
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    if not rows:
        print(f"no rows in {rows_path}", file=sys.stderr)
        return 2
    summary = summarize_rows(rows)
    if summary["stored_anls_mismatches"]:
        print(
            f"ERROR: {summary['stored_anls_mismatches']} rows have edited ANLS "
            f"(max drift {summary['max_stored_anls_drift']:.3g})",
            file=sys.stderr,
        )
        return 1

    summary.update(
        {
            "rows": str(rows_path),
            "set": "docvqa_public_mirror",
            "single_reference": True,
            "sampling": "one question per image, seed 42",
            "caveat": (
                "public DocVQA questions; may be in pretraining; reading probe, not extraction; "
                "single-reference ANLS, not comparable to leaderboard numbers"
            ),
        }
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
