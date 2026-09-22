#!/usr/bin/env python3
"""Run Phase-0 baselines on the 100-doc SROIE eval set.

Usage:
  python evaluation/phase0/run_baseline.py --engine ocr_regex
  python evaluation/phase0/run_baseline.py --engine smolvlm2 --limit 20
  python evaluation/phase0/run_baseline.py --engine ollama --limit 20
  python evaluation/phase0/run_baseline.py --engine ppdocbee --limit 20
  python evaluation/phase0/run_baseline.py --engine all --limit 20

Writes:
  results/preds_{engine}.jsonl
  results/scores_{engine}.json
  results/baseline_table.md   (merged, engine scores present so far)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).absolute().parent))

from metrics import macro_prf, score_example, normalize_gold_target  # noqa: E402
from engines import build_engine  # noqa: E402

# absolute parent walk — evaluation/ is a symlink; resolve() would leave the repo
ROOT = Path(__file__).absolute().parent.parent.parent
EVAL_PATH = Path(__file__).absolute().parent / "results" / "sroie_eval.json"
RESULTS = Path(__file__).absolute().parent / "results"


def run_engine(
    engine_name: str,
    limit: int | None,
    model_path: str | None = None,
    eval_path: Path | None = None,
    out_suffix: str = "",
):
    path = eval_path or EVAL_PATH
    eval_set = json.loads(path.read_text())
    if limit:
        eval_set = eval_set[:limit]
    kwargs = {}
    if model_path and engine_name == "smolvlm2":
        kwargs["model_path"] = model_path
    print(f"engine={engine_name} n={len(eval_set)} eval={path}")
    engine = build_engine(engine_name, **kwargs)

    preds = []
    scores = []
    for i, ex in enumerate(eval_set):
        try:
            out = engine.extract(ex["image_path"])
        except Exception as e:
            traceback.print_exc()
            out = {
                "engine": engine_name,
                "fields": {k: "" for k in ("company", "date", "address", "total")},
                "raw": f"ERROR: {e}",
                "latency_ms": 0,
                "schema_ready": False,
            }
        gold = ex["gold"]
        sc = score_example(out["fields"], gold)
        rec = {
            "id": ex["id"],
            "image_path": ex["image_path"],
            "pred": out["fields"],
            "gold": gold,
            "latency_ms": out.get("latency_ms", 0),
            "raw": out.get("raw", "")[:2000],
            "score": sc,
        }
        preds.append(rec)
        scores.append(sc)
        if (i + 1) % 5 == 0 or i == 0 or (i + 1) == len(eval_set):
            agg = macro_prf(scores)
            print(
                f"  [{i+1}/{len(eval_set)}] f1={agg['field_f1']:.3f} "
                f"anls={agg['anls_mean']:.3f} schema={agg['schema_valid_rate']:.3f}"
            )

    agg = macro_prf(scores)
    agg["engine"] = engine_name
    agg["eval_path"] = str(path)
    agg["n_examples"] = len(scores)
    agg["limit"] = limit or len(eval_set)
    agg["avg_latency_ms"] = sum(p["latency_ms"] for p in preds) / max(len(preds), 1)
    if out_suffix:
        agg["split"] = out_suffix.lstrip("_")

    RESULTS.mkdir(parents=True, exist_ok=True)
    safe = engine_name.replace(":", "_")
    pred_path = RESULTS / f"preds_{safe}{out_suffix}.jsonl"
    with open(pred_path, "w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    score_path = RESULTS / f"scores_{safe}{out_suffix}.json"
    score_path.write_text(json.dumps(agg, indent=2))
    print(json.dumps(agg, indent=2))
    return agg


def merge_table():
    rows = []
    for p in sorted(RESULTS.glob("scores_*.json")):
        # skip split-suffixed scores (e.g. scores_ollama_unseen.json) so the
        # main table stays n=100 holdout only; unseen is reported separately.
        # skip pipeline end-to-end scores (scores_pipeline_*.json) — product path,
        # reported in Phase-1 docs, not Phase-0 engine baselines.
        if p.stem.endswith("_unseen") or p.stem.endswith("_seen"):
            continue
        if p.stem.startswith("scores_pipeline_") or p.stem.startswith("pipeline_"):
            continue
        d = json.loads(p.read_text())
        if d.get("split"):
            continue
        if str(d.get("engine", "")).startswith("receipt_pipeline"):
            continue
        rows.append(d)
    if not rows:
        return
    lines = [
        "# Phase 0 — SROIE receipt field extraction baselines",
        "",
        f"Eval: 100 held-out SROIE images (`results/sroie_eval.json`), fields: company/date/address/total.",
        "",
        "| Engine | n | Field P | Field R | Field F1 | ANLS | Schema-valid | Exact-4/4 | Avg latency ms |",
        "|--------|---|---------|---------|----------|------|--------------|-----------|----------------|",
    ]
    rows.sort(key=lambda r: -r.get("field_f1", 0))
    for r in rows:
        lines.append(
            f"| {r.get('engine')} | {r.get('n_examples')} "
            f"| {r.get('field_precision', 0):.3f} | {r.get('field_recall', 0):.3f} "
            f"| **{r.get('field_f1', 0):.3f}** | {r.get('anls_mean', 0):.3f} "
            f"| {r.get('schema_valid_rate', 0):.3f} | {r.get('exact_json_all_fields', 0):.3f} "
            f"| {r.get('avg_latency_ms', 0):.0f} |"
        )
    lines += [
        "",
        "Notes:",
        "- `field_f1` is micro over (example × field) binary matches with normalize + type-aware rules (money/date).",
        "- company/address match is **punctuation-insensitive** (alnum tokens; same rule as evidence.py) — 2026-09-22 metric fix for OCR comma/period variance. Re-scored from existing preds (`rescore_preds.py`).",
        "- `ocr_regex` is a floor (weak OCR heuristics), not a competitive parser.",
        "- Layout-family split: `results/layout_families.json` + `scores_*_unseen.json` (excluded from this table).",
        "- Ollama engine: unconstrained JSON + salvage (no `format=json` — see engines.py).",
        "- Marked numbers are measured on this machine (MPS/CPU), not copied from papers.",
    ]
    (RESULTS / "baseline_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--engine",
        required=True,
        help="ocr_regex | smolvlm2 | ppdocbee | ollama | ollama:<model> | all",
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model-path", default=None, help="override smolvlm2 path")
    ap.add_argument("--table-only", action="store_true")
    ap.add_argument(
        "--eval-path",
        default=None,
        help="override eval JSON (default results/sroie_eval.json)",
    )
    ap.add_argument(
        "--out-suffix",
        default="",
        help="suffix for preds_/scores_ files, e.g. _unseen",
    )
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.table_only:
        merge_table()
        return

    eval_path = Path(args.eval_path) if args.eval_path else None
    engines = (
        ["ocr_regex", "smolvlm2", "ollama", "ppdocbee"]
        if args.engine == "all"
        else [args.engine]
    )
    for e in engines:
        try:
            run_engine(
                e,
                args.limit,
                model_path=args.model_path,
                eval_path=eval_path,
                out_suffix=args.out_suffix,
            )
        except Exception as ex:
            print(f"ENGINE {e} FAILED: {ex}")
            traceback.print_exc()
    if not args.out_suffix:
        merge_table()


if __name__ == "__main__":
    main()
