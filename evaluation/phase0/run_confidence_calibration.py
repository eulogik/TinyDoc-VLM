#!/usr/bin/env python3
"""Grouped cross-validated calibration for confidence scores (no model calls).

Reads prediction JSONL with per-field confidences, fits isotonic regression on
training folds, and reports out-of-fold ECE / Brier / AURC for the raw and
calibrated scores. Folds are assigned by document, never by field, so fields
from the same document cannot leak across the split.

This is a measurement harness, not a production calibrator yet. The fitted map
is saved only when `--save-map` is passed, and it must be trained on a dedicated
calibration split before being wired into the SDK.

Usage:
    python evaluation/phase0/run_confidence_calibration.py \
        --preds evaluation/phase0/results/preds_pipeline_ollama_qwen2.5vl_3b_clean_ctx8k_evidence.jsonl \
        --out evaluation/phase0/results/calibration_qwen2.5vl_3b_clean.json
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import statistics
from collections.abc import Sequence
from pathlib import Path

from run_confidence_probe import FIELDS, aurc, brier, ece, top_band_precision

Knots = list[tuple[float, float]]


def grouped_folds(ids: Sequence[str], k: int = 5) -> list[int]:
    """Deterministic document-level folds, balanced round-robin."""
    unique = sorted(set(ids), key=lambda s: hashlib.sha256(s.encode()).hexdigest())
    fold_of = {doc_id: i % k for i, doc_id in enumerate(unique)}
    return [fold_of[doc_id] for doc_id in ids]


def fit_isotonic(
    xs: Sequence[float], ys: Sequence[bool], weights: Sequence[float] | None = None
) -> Knots:
    """Weighted pool-adjacent-violators; returns (x, y) knots, monotone in y."""
    if len(xs) != len(ys):
        raise ValueError("xs and ys differ in length")
    if not xs:
        return []
    ws = list(weights) if weights is not None else [1.0] * len(xs)

    agg: dict[float, list[float]] = {}
    for x, y, w in zip(xs, ys, ws):
        bucket = agg.setdefault(float(x), [0.0, 0.0])
        bucket[0] += w
        bucket[1] += w * float(bool(y))
    points = [(x, w, sy / w) for x, (w, sy) in sorted(agg.items()) if w > 0]

    blocks: list[list[float]] = []  # [weight, weighted_x, mean_y]
    for x, w, mean_y in points:
        blocks.append([w, w * x, mean_y])
        while len(blocks) > 1 and blocks[-2][2] > blocks[-1][2]:
            w2, wx2, m2 = blocks.pop()
            w1, wx1, m1 = blocks.pop()
            ww = w1 + w2
            blocks.append([ww, (wx1 + wx2), (m1 * w1 + m2 * w2) / ww])

    return [(wx / w, mean_y) for w, wx, mean_y in blocks]


def predict_isotonic(knots: Knots, x: float) -> float:
    """Piecewise-linear interpolation between monotone knots, clamped at ends."""
    if not knots:
        return float(x)
    xs = [k[0] for k in knots]
    if x <= xs[0]:
        return knots[0][1]
    if x >= xs[-1]:
        return knots[-1][1]
    i = bisect.bisect_right(xs, x) - 1
    x0, y0 = knots[i]
    x1, y1 = knots[i + 1]
    if x1 <= x0:
        return y1
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def cross_val_predict(
    xs: Sequence[float],
    ys: Sequence[bool],
    ids: Sequence[str],
    k: int = 5,
) -> tuple[list[float], int]:
    folds = grouped_folds(ids, k)
    out = [float(x) for x in xs]
    for fold in range(k):
        train = [i for i, f in enumerate(folds) if f != fold]
        test = [i for i, f in enumerate(folds) if f == fold]
        if not train or not test:
            continue
        knots = fit_isotonic([xs[i] for i in train], [ys[i] for i in train])
        for i in test:
            out[i] = predict_isotonic(knots, xs[i])
    return out, k


def summarize(conf: Sequence[float], correct: Sequence[bool]) -> dict:
    return {
        "n": len(conf),
        "mean_confidence": statistics.mean(conf) if conf else 0.0,
        "accuracy": sum(int(bool(y)) for y in correct) / len(correct) if correct else 0.0,
        "ece_10bin": ece(conf, correct),
        "brier": brier(conf, correct),
        "aurc": aurc(conf, correct),
        "top50_precision": top_band_precision(conf, correct, 0.5),
        "top70_precision": top_band_precision(conf, correct, 0.7),
    }


def calibration_report(rows: Sequence[dict], k: int = 5) -> dict:
    field_conf: list[float] = []
    field_ok: list[bool] = []
    field_ids: list[str] = []
    per_field_conf: dict[str, list[float]] = {name: [] for name in FIELDS}
    per_field_ok: dict[str, list[bool]] = {name: [] for name in FIELDS}
    per_field_ids: dict[str, list[str]] = {name: [] for name in FIELDS}
    doc_conf: list[float] = []
    doc_ok: list[bool] = []
    doc_ids: list[str] = []

    for row in rows:
        score = row.get("score") or {}
        fields = score.get("fields") or {}
        per_field = row.get("field_confidences")
        doc_id = str(row.get("id", row.get("image_path", "")))
        hits = []
        for name in FIELDS:
            hit = bool((fields.get(name) or {}).get("hit"))
            conf = (
                float(per_field.get(name, row.get("confidence", 0.0)))
                if isinstance(per_field, dict)
                else float(row.get("confidence", 0.0))
            )
            field_conf.append(conf)
            field_ok.append(hit)
            field_ids.append(doc_id)
            per_field_conf[name].append(conf)
            per_field_ok[name].append(hit)
            per_field_ids[name].append(doc_id)
            hits.append(hit)
        doc_conf.append(float(row.get("confidence", 0.0)))
        doc_ok.append(all(hits))
        doc_ids.append(doc_id)

    field_cal, folds = cross_val_predict(field_conf, field_ok, field_ids, k)
    doc_cal, _ = cross_val_predict(doc_conf, doc_ok, doc_ids, k)

    by_field = {}
    for name in FIELDS:
        calibrated, _ = cross_val_predict(
            per_field_conf[name], per_field_ok[name], per_field_ids[name], k
        )
        by_field[name] = {
            "raw": summarize(per_field_conf[name], per_field_ok[name]),
            "isotonic_oof": summarize(calibrated, per_field_ok[name]),
        }

    return {
        "n_docs": len(rows),
        "n_field_rows": len(field_conf),
        "n_docs_with_field_confidences": sum(
            1 for r in rows if isinstance(r.get("field_confidences"), dict)
        ),
        "folds": folds,
        "constant_confidence": len({round(c, 6) for c in field_conf}) == 1,
        "fields": {
            "raw": summarize(field_conf, field_ok),
            "isotonic_oof": summarize(field_cal, field_ok),
        },
        "by_field": by_field,
        "doc_exact4": {
            "raw": summarize(doc_conf, doc_ok),
            "isotonic_oof": summarize(doc_cal, doc_ok),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--save-map", default=None)
    args = ap.parse_args()

    rows: list[dict] = []
    for path in args.preds:
        with Path(path).open() as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    if not rows:
        print("no prediction rows")
        return 2
    if not any(isinstance(r.get("field_confidences"), dict) for r in rows):
        print("no field_confidences in input; rerun with evidence attached first")
        return 2

    report = {
        "preds": [str(p) for p in args.preds],
        **calibration_report(rows, args.folds),
    }

    if args.save_map:
        xs: list[float] = []
        ys: list[bool] = []
        for row in rows:
            per_field = row.get("field_confidences") or {}
            fields = (row.get("score") or {}).get("fields") or {}
            for name in FIELDS:
                xs.append(float(per_field.get(name, row.get("confidence", 0.0))))
                ys.append(bool((fields.get(name) or {}).get("hit")))
        knots = fit_isotonic(xs, ys)
        map_path = Path(args.save_map)
        map_path.parent.mkdir(parents=True, exist_ok=True)
        map_path.write_text(
            json.dumps(
                {
                    "method": "isotonic",
                    "fitted_on": [str(p) for p in args.preds],
                    "n": len(xs),
                    "knots": knots,
                    "warning": "fit on all rows; validate on a held-out split before production",
                },
                indent=2,
            )
            + "\n"
        )
        report["saved_map"] = str(map_path)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
