#!/usr/bin/env python3
"""Compact summary of an invoice-benchmark scores file (no model calls).

Reads `scores_invoice_<tag>.json` from the benchmark scorer plus the runner's
rows JSONL (for latency/parse errors) and writes one small JSON with overall,
per-field, and per-category key-header accuracy.

Usage:
    python evaluation/invoice/summarize_scores.py \
        --scores evaluation/invoice/results/scores_invoice_qwen2.5vl_3b.json \
        --rows evaluation/invoice/results/rows_invoice_qwen2.5vl_3b.jsonl \
        --out evaluation/invoice/results/summary_invoice_qwen2.5vl_3b.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

KEY_FIELDS = (
    "vendor_name",
    "invoice_number",
    "invoice_date",
    "due_date",
    "currency",
    "subtotal",
    "tax",
    "total",
    "document_type",
)


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = min(max(round(q * (len(ordered) - 1)), 0), len(ordered) - 1)
    return ordered[index]


def bootstrap_ci(docs: list[dict], n_boot: int = 2000, seed: int = 42) -> dict:
    """Document-level bootstrap for key-header accuracy and fully-correct rate."""
    per_doc = []
    for doc in docs:
        correct = scored = 0
        for name in KEY_FIELDS:
            result = (doc.get("fields") or {}).get(name)
            if result is None:
                continue
            scored += 1
            correct += int(bool(result.get("correct")))
        per_doc.append((correct, scored, int(bool(doc.get("fully_correct")))))
    if not per_doc:
        return {"key_header_accuracy": [0.0, 0.0], "fully_correct_rate": [0.0, 0.0]}

    rng = random.Random(seed)
    key_samples: list[float] = []
    full_samples: list[float] = []
    for _ in range(n_boot):
        key_correct = key_scored = fully = 0
        for _ in per_doc:
            correct, scored, is_full = per_doc[rng.randrange(len(per_doc))]
            key_correct += correct
            key_scored += scored
            fully += is_full
        key_samples.append(key_correct / key_scored if key_scored else 0.0)
        full_samples.append(fully / len(per_doc))
    return {
        "key_header_accuracy": [
            _percentile(key_samples, 0.025),
            _percentile(key_samples, 0.975),
        ],
        "fully_correct_rate": [
            _percentile(full_samples, 0.025),
            _percentile(full_samples, 0.975),
        ],
    }


def summarize(scores: dict, rows: list[dict] | None = None) -> dict:
    docs = scores.get("documents") or []
    per_category: dict[str, dict] = defaultdict(
        lambda: {"docs": 0, "key_correct": 0, "key_scored": 0, "fully_correct": 0, "line_f1": []}
    )
    per_field: dict[str, dict] = defaultdict(lambda: {"correct": 0, "scored": 0})
    key_correct = key_scored = fully_correct = 0

    for doc in docs:
        category = doc.get("category") or "unknown"
        cat = per_category[category]
        cat["docs"] += 1
        if doc.get("fully_correct"):
            fully_correct += 1
            cat["fully_correct"] += 1
        if doc.get("line_f1") is not None:
            cat["line_f1"].append(float(doc["line_f1"]))
        for name, result in (doc.get("fields") or {}).items():
            if name not in KEY_FIELDS:
                continue
            per_field[name]["scored"] += 1
            cat["key_scored"] += 1
            key_scored += 1
            if result.get("correct"):
                per_field[name]["correct"] += 1
                cat["key_correct"] += 1
                key_correct += 1

    for cat in per_category.values():
        f1s = cat.pop("line_f1")
        cat["line_f1_mean"] = sum(f1s) / len(f1s) if f1s else None

    latency = []
    errors = 0
    if rows:
        for row in rows:
            latency.append(float(row.get("latency_ms", 0.0)))
            if row.get("error"):
                errors += 1

    outgoing = {
        "system": (scores.get("meta") or {}).get("system"),
        "run_date": (scores.get("meta") or {}).get("run_date"),
        "n_docs": len(docs),
        "n_corpus": (scores.get("summary") or {}).get("documents_in_corpus"),
        "key_header_accuracy": key_correct / key_scored if key_scored else 0.0,
        "key_header_correct": key_correct,
        "key_header_scored": key_scored,
        "fully_correct": fully_correct,
        "fully_correct_rate": fully_correct / len(docs) if docs else 0.0,
        "ci95": bootstrap_ci(docs),
        "per_field": dict(sorted(per_field.items())),
        "per_category": dict(sorted(per_category.items())),
        "flags": (scores.get("summary") or {}).get("flags"),
        "line_items": (scores.get("summary") or {}).get("line_items"),
    }
    if latency:
        outgoing["latency_ms"] = {
            "mean": statistics.mean(latency),
            "p50": statistics.median(latency),
        }
        outgoing["parse_errors"] = errors
    return outgoing


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--rows", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    scores = json.loads(Path(args.scores).read_text())
    rows = None
    if args.rows:
        with Path(args.rows).open() as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    summary = summarize(scores, rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"summary -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
