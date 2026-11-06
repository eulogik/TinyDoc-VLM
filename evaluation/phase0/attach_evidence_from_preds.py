#!/usr/bin/env python3
"""Attach OCR-span evidence to existing predictions (no VLM calls).

The phase-0 clean-100 runs were produced with evidence off. That makes the
document confidence a constant (0.4025) and discards the per-field evidence
signal. This utility replays only the OCR/evidence stage on the saved fields,
using the same `locate_field_evidence` and `_field_confidence` code as the
shipped pipeline, and writes an evidence-on prediction file.

No model inference. It does run RapidOCR over the eval images, which is still
batch OCR: get approval before running it over 10+ documents.

Usage:
    python evaluation/phase0/attach_evidence_from_preds.py \
        --preds evaluation/phase0/results/preds_pipeline_ollama_qwen2.5vl_3b_clean_ctx8k.jsonl \
        --out evaluation/phase0/results/preds_pipeline_ollama_qwen2.5vl_3b_clean_ctx8k_evidence.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "sdk"))

from tinydoc.evidence import _ocr_words, locate_field_evidence
from tinydoc.pipeline import FIELDS, _field_confidence, schema_is_valid

Words = list[dict]


def ocr_words(image_path: str) -> tuple[Words, str]:
    """RapidOCR words with an explicit Tesseract fallback label."""
    words = _ocr_words(image_path, engine="rapidocr")
    if words:
        return words, "rapidocr"
    return _ocr_words(image_path, engine="tesseract"), "tesseract"


def enrich_row(row: dict) -> dict:
    fields = {k: str((row.get("pred") or {}).get(k, "") or "") for k in FIELDS}
    schema_ok = bool(row.get("schema_valid_pipeline", schema_is_valid(fields)))

    evidence_error = None
    try:
        words, ocr_engine = ocr_words(str(row["image_path"]))
    except Exception as exc:  # noqa: BLE001 - report, do not lose the row
        words, ocr_engine = [], "none"
        evidence_error = f"{type(exc).__name__}: {exc}"

    field_conf = {}
    field_ev = {}
    confs = []
    present_with_evidence = 0
    for name in FIELDS:
        value = fields[name]
        ev = locate_field_evidence(value, str(row["image_path"]), words=words)
        conf = _field_confidence(name, value, ev, schema_ok)
        field_conf[name] = conf
        field_ev[name] = ev.to_dict() if ev else None
        if value:
            confs.append(conf)
            if ev is not None:
                present_with_evidence += 1

    coverage = present_with_evidence / len(FIELDS)
    mean_conf = sum(confs) / len(confs) if confs else 0.0
    overall = mean_conf * (0.7 + 0.3 * coverage)
    if not schema_ok:
        overall *= 0.75

    out = dict(row)
    out["confidence"] = round(float(min(max(overall, 0.0), 1.0)), 4)
    out["field_confidences"] = field_conf
    out["field_evidence"] = field_ev
    out["evidence_coverage"] = round(coverage, 4)
    out["evidence_engine"] = ocr_engine
    if evidence_error:
        out["evidence_error"] = evidence_error
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    src = Path(args.preds)
    with src.open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print(f"no rows in {src}", file=sys.stderr)
        return 2

    out_rows = []
    for i, row in enumerate(rows, 1):
        out_rows.append(enrich_row(row))
        if i % 10 == 0:
            print(
                f"[{i}/{len(rows)}] mean_conf="
                f"{statistics.mean(r['confidence'] for r in out_rows):.4f} "
                f"mean_cov={statistics.mean(r['evidence_coverage'] for r in out_rows):.3f}",
                flush=True,
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for row in out_rows:
            handle.write(json.dumps(row) + "\n")

    errors = sum(1 for r in out_rows if r.get("evidence_error"))
    engine_counts: dict[str, int] = {}
    for row in out_rows:
        engine_counts[row["evidence_engine"]] = engine_counts.get(row["evidence_engine"], 0) + 1
    print(
        json.dumps(
            {
                "preds": str(src),
                "out": str(out),
                "n": len(out_rows),
                "mean_confidence": statistics.mean(r["confidence"] for r in out_rows),
                "mean_evidence_coverage": statistics.mean(
                    r["evidence_coverage"] for r in out_rows
                ),
                "evidence_errors": errors,
                "ocr_engines": engine_counts,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
