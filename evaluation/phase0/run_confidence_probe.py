#!/usr/bin/env python3
"""Confidence calibration probe for existing pipeline predictions (no model calls).

Reads per-document prediction JSONL files produced by
`evaluation/phase0/run_pipeline_eval.py` and reports:

  - ECE / Brier for field-level confidence vs field hit
  - AURC and risk-coverage (lower AURC is better)
  - top-band precision at 50% / 70% coverage
  - doc-level exact-4/4 calibration

If a row lacks `field_confidences` (older artifacts), the document-level
confidence is used for every field; the report says so explicitly.

Usage:
    python evaluation/phase0/run_confidence_probe.py \
        --preds evaluation/phase0/results/preds_pipeline_ollama_qwen2.5vl_3b_clean_ctx8k.jsonl \
        --out evaluation/phase0/results/confidence_probe_qwen2.5vl_3b_clean.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Sequence
from pathlib import Path

FIELDS = ("company", "date", "address", "total")


def ece(confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 10) -> float:
    """Expected calibration error with equal-width confidence bins."""
    if not confidences:
        return 0.0
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for c, y in zip(confidences, correct):
        idx = min(max(int(float(c) * n_bins), 0), n_bins - 1)
        bins[idx].append((float(c), bool(y)))
    total = len(confidences)
    err = 0.0
    for bucket in bins:
        if not bucket:
            continue
        conf = sum(c for c, _ in bucket) / len(bucket)
        acc = sum(y for _, y in bucket) / len(bucket)
        err += (len(bucket) / total) * abs(acc - conf)
    return err


def brier(confidences: Sequence[float], correct: Sequence[bool]) -> float:
    if not confidences:
        return 0.0
    return sum((float(c) - float(bool(y))) ** 2 for c, y in zip(confidences, correct)) / len(
        confidences
    )


def aurc(confidences: Sequence[float], correct: Sequence[bool]) -> float:
    """Area under the risk-coverage curve when accepting high confidence first."""
    if not confidences:
        return 0.0
    pairs = sorted(zip(confidences, correct), key=lambda p: p[0], reverse=True)
    risks = [0.0]
    covs = [0.0]
    hits = 0
    for k, (_, y) in enumerate(pairs, 1):
        hits += int(bool(y))
        risks.append(1.0 - hits / k)
        covs.append(k / len(pairs))
    area = 0.0
    for i in range(1, len(covs)):
        area += (covs[i] - covs[i - 1]) * (risks[i] + risks[i - 1]) / 2
    return area


def top_band_precision(
    confidences: Sequence[float], correct: Sequence[bool], coverage: float = 0.5
) -> float:
    if not confidences:
        return 0.0
    k = max(1, math.ceil(len(confidences) * coverage))
    pairs = sorted(zip(confidences, correct), key=lambda p: p[0], reverse=True)[:k]
    return sum(int(bool(y)) for _, y in pairs) / k


def summarize(confidences: Sequence[float], correct: Sequence[bool]) -> dict:
    return {
        "n": len(confidences),
        "mean_confidence": statistics.mean(confidences) if confidences else 0.0,
        "accuracy": sum(int(bool(y)) for y in correct) / len(correct) if correct else 0.0,
        "ece_10bin": ece(confidences, correct),
        "brier": brier(confidences, correct),
        "aurc": aurc(confidences, correct),
        "top50_precision": top_band_precision(confidences, correct, 0.5),
        "top70_precision": top_band_precision(confidences, correct, 0.7),
    }


def load_rows(paths: Sequence[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        with path.open() as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def probe(rows: Sequence[dict]) -> dict:
    field_conf: list[float] = []
    field_ok: list[bool] = []
    doc_conf: list[float] = []
    doc_ok: list[bool] = []
    used_field_conf = 0

    for row in rows:
        score = row.get("score") or {}
        fields = score.get("fields") or {}
        per_field = row.get("field_confidences")
        if isinstance(per_field, dict):
            used_field_conf += 1
        for field in FIELDS:
            hit = bool((fields.get(field) or {}).get("hit"))
            conf = (
                float(per_field.get(field, row.get("confidence", 0.0)))
                if isinstance(per_field, dict)
                else float(row.get("confidence", 0.0))
            )
            field_conf.append(conf)
            field_ok.append(hit)
        all_hit = all(bool((fields.get(f) or {}).get("hit")) for f in FIELDS)
        doc_conf.append(float(row.get("confidence", 0.0)))
        doc_ok.append(all_hit)

    return {
        "n_docs": len(rows),
        "n_field_rows": len(field_conf),
        "used_field_confidences": used_field_conf == len(rows) and len(rows) > 0,
        "field_level": summarize(field_conf, field_ok),
        "doc_exact4": summarize(doc_conf, doc_ok),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    paths = [Path(p) for p in args.preds]
    rows = load_rows(paths)
    if not rows:
        print("no prediction rows")
        return 2
    report = {
        "preds": [str(p) for p in paths],
        "n_docs": len(rows),
        **probe(rows),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
