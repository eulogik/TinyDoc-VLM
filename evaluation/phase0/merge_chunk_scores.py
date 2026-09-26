"""Merge chunked 3B baseline runs into one clean-100 scores file.

Chunks (15-20 docs each with full ollama unload between) are an operational
necessity on the 16GB host: sustained qwen2.5vl:3b vision inference decays
available memory until the watchdog fires. Every chunk uses the identical
engine, prompt, temperature, and scorer — merging is concatenation of per-doc
records in doc order with macro recomputed by the same metrics.py.
Usage:
  python evaluation/phase0/merge_chunk_scores.py \
      --chunks _clean_c0 _clean_c1 _clean_c2 _clean_c3 _clean_c4 \
      --out-suffix _clean
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "evaluation" / "phase0"))

from metrics import FIELDS, macro_prf

RESULTS = ROOT / "evaluation/phase0/results"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", nargs="+", required=True)
    ap.add_argument("--out-suffix", default="_clean")
    ap.add_argument("--engine-label", default="ollama_qwen2.5vl_3b")
    args = ap.parse_args()

    preds, scores, errs, prov = [], [], 0, []
    for suf in args.chunks:
        pj = RESULTS / f"preds_pipeline_{args.engine_label}{suf}.jsonl"
        sj = RESULTS / f"scores_pipeline_{args.engine_label}{suf}.json"
        chunk_preds = [json.loads(l) for l in pj.read_text().splitlines() if l.strip()]
        chunk_scores = json.loads(sj.read_text())
        preds.extend(chunk_preds)
        scores.extend(r["score"] for r in chunk_preds)
        errs += sum(1 for r in chunk_preds if r.get("error"))
        prov.append({"suffix": suf, "n": len(chunk_preds),
                     "errors": sum(1 for r in chunk_preds if r.get("error")),
                     "field_f1": chunk_scores.get("field_f1")})
    assert len(preds) == sum(p["n"] for p in prov), "chunk size mismatch"
    ids = [p["id"] for p in preds]
    assert len(set(ids)) == len(ids), "duplicate doc ids across chunks"

    agg = macro_prf(scores)
    out = {
        "engine": "receipt_pipeline:ollama:qwen2.5vl:3b",
        "n_examples": len(preds),
        "field_f1": agg["field_f1"],
        "field_precision": agg["field_precision"],
        "field_recall": agg["field_recall"],
        "schema_valid_rate": agg["schema_valid_rate"],
        "pipeline_schema_valid_rate": agg["schema_valid_rate"],
        "anls_mean": agg["anls_mean"],
        "exact_json_all_fields": agg["exact_json_all_fields"],
        "avg_latency_ms": sum(p["latency_ms"] for p in preds) / max(len(preds), 1),
        "per_field_hit_rate": {
            f: sum(1 for s in scores if s["fields"][f]["hit"]) / len(scores) for f in FIELDS
        },
        "errors": errs,
        "error_ids": [p["id"] for p in preds if p.get("error")],
        "merged_from_chunks": prov,
        "merge_method": ("concatenated per-doc preds in doc order; macro recomputed "
                         "with evaluation/phase0/metrics.py macro_prf (identical scorer); "
                         "identical engine/prompt/temperature in every chunk"),
    }
    (RESULTS / f"scores_pipeline_{args.engine_label}{args.out_suffix}.json").write_text(
        json.dumps(out, indent=2))
    (RESULTS / f"preds_pipeline_{args.engine_label}{args.out_suffix}.jsonl").write_text(
        "\n".join(json.dumps(p) for p in preds) + "\n")
    print(json.dumps({k: out[k] for k in ("n_examples", "field_f1", "errors")}, indent=2))
    print("chunks:", json.dumps(prov))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
