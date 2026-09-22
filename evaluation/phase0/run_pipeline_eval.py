#!/usr/bin/env python3
"""Authoritative end-to-end eval: ReceiptPipeline (SDK) on SROIE holdout.

Uses sdk/tinydoc ReceiptPipeline (sanitize + schema + confidence path)
with the same metrics as Phase-0 baselines.

Usage:
  python evaluation/phase0/run_pipeline_eval.py --engine ollama
  python evaluation/phase0/run_pipeline_eval.py --engine ollama --limit 10
  python evaluation/phase0/run_pipeline_eval.py --engine ollama --eval-path ... --out-suffix _unseen

Writes (NOT merged into Phase-0 baseline_table — separate product metrics):
  results/preds_pipeline_{engine}{suffix}.jsonl
  results/scores_pipeline_{engine}{suffix}.json
  results/pipeline_address_failures{suffix}.json   (address misses only)
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).absolute().parent))

from metrics import macro_prf, score_example, FIELDS  # noqa: E402

# evaluation/ is a symlink — never resolve(); absolute walk only
ROOT = Path(__file__).absolute().parent.parent.parent
SDK = ROOT / "sdk"
if str(SDK) not in sys.path:
    sys.path.insert(0, str(SDK))

from tinydoc.pipeline import ReceiptPipeline  # noqa: E402

RESULTS = Path(__file__).absolute().parent / "results"
DEFAULT_EVAL = RESULTS / "sroie_eval.json"


def run_pipeline(
    engine_name: str,
    limit: int | None,
    eval_path: Path,
    out_suffix: str,
    with_evidence: bool,
    model: str | None,
) -> dict:
    eval_set = json.loads(eval_path.read_text())
    if limit:
        eval_set = eval_set[:limit]
    kwargs = {}
    if model:
        kwargs["model"] = model
    # engine_name: ollama | ocr_regex | smolvlm2 | auto | ollama:<model>
    pipe = ReceiptPipeline(engine_name, **kwargs)
    eng_label = pipe.engine.name.replace(":", "_")
    print(
        f"pipeline engine={pipe.engine.name} n={len(eval_set)} "
        f"evidence={with_evidence} eval={eval_path}"
    )

    preds: list[dict] = []
    scores: list[dict] = []
    for i, ex in enumerate(eval_set):
        try:
            doc = pipe.extract(ex["image_path"], with_evidence=with_evidence)
            fields = doc.fields
            raw = doc.raw
            lat = doc.latency_ms
            schema_ok = doc.schema_valid
            conf = doc.confidence
            cov = doc.evidence_coverage
            error = None
        except Exception as e:
            traceback.print_exc()
            fields = {k: "" for k in FIELDS}
            raw = f"ERROR: {e}"
            lat = 0.0
            schema_ok = False
            conf = 0.0
            cov = 0.0
            error = str(e)

        gold = ex["gold"]
        sc = score_example(fields, gold)
        # score_example computes schema from pred; pipeline path is authoritative
        # when they disagree (should not — same jsonschema rules).
        rec = {
            "id": ex["id"],
            "image_path": ex["image_path"],
            "pred": fields,
            "gold": gold,
            "latency_ms": lat,
            "schema_valid_pipeline": schema_ok,
            "schema_valid_metrics": sc["schema_valid"],
            "confidence": conf,
            "evidence_coverage": cov,
            "raw": raw[:2000],
            "error": error,
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
    agg["engine"] = f"receipt_pipeline:{pipe.engine.name}"
    agg["eval_path"] = str(eval_path)
    agg["n_examples"] = len(scores)
    agg["limit"] = limit or len(eval_set)
    agg["avg_latency_ms"] = sum(p["latency_ms"] for p in preds) / max(len(preds), 1)
    agg["with_evidence"] = with_evidence
    agg["pipeline_schema_valid_rate"] = sum(
        1 for p in preds if p["schema_valid_pipeline"]
    ) / max(len(preds), 1)
    agg["mean_confidence"] = sum(p["confidence"] for p in preds) / max(len(preds), 1)
    agg["mean_evidence_coverage"] = sum(
        p["evidence_coverage"] for p in preds
    ) / max(len(preds), 1)
    if out_suffix:
        agg["split"] = out_suffix.lstrip("_")

    # per-field hit rates
    per = {}
    for f in FIELDS:
        hits = sum(1 for s in scores if s["fields"][f]["hit"])
        anls_v = sum(s["fields"][f]["anls"] for s in scores) / max(len(scores), 1)
        per[f] = {"hit_rate": hits / max(len(scores), 1), "anls_mean": anls_v}
    agg["per_field"] = per

    RESULTS.mkdir(parents=True, exist_ok=True)
    # evidence-on writes *_evidence.* so it never clobbers the evidence-off baseline
    ev_tag = "_evidence" if with_evidence else ""
    stem = f"pipeline_{eng_label}{out_suffix}{ev_tag}"
    with (RESULTS / f"preds_{stem}.jsonl").open("w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    (RESULTS / f"scores_{stem}.json").write_text(json.dumps(agg, indent=2))

    # address (and any-field) failure dump for analysis — no gold leakage into pipeline
    fails = []
    for p in preds:
        sf = p["score"]["fields"]
        if not sf["address"]["hit"]:
            fails.append(
                {
                    "id": p["id"],
                    "image_path": p["image_path"],
                    "pred": sf["address"]["pred"],
                    "gold": sf["address"]["gold"],
                    "anls": sf["address"]["anls"],
                    "pred_len": len(str(sf["address"]["pred"] or "")),
                    "gold_len": len(str(sf["address"]["gold"] or "")),
                }
            )
    fail_path = RESULTS / f"pipeline_address_failures{out_suffix}{ev_tag}.json"
    fail_path.write_text(json.dumps(fails, indent=2))

    print(json.dumps(agg, indent=2))
    print(f"address_misses={len(fails)} → {fail_path}")
    return agg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", default="ollama", help="ReceiptPipeline engine id")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--eval-path", default=None)
    ap.add_argument("--out-suffix", default="")
    ap.add_argument(
        "--with-evidence",
        action="store_true",
        help="attach Tesseract OCR evidence (default off — same fields, slower)",
    )
    ap.add_argument("--model", default=None, help="ollama model id override")
    args = ap.parse_args()

    eval_path = Path(args.eval_path) if args.eval_path else DEFAULT_EVAL
    if not eval_path.exists():
        print(f"eval not found: {eval_path}", file=sys.stderr)
        raise SystemExit(2)

    run_pipeline(
        engine_name=args.engine,
        limit=args.limit,
        eval_path=eval_path,
        out_suffix=args.out_suffix,
        with_evidence=args.with_evidence,
        model=args.model,
    )


if __name__ == "__main__":
    main()
