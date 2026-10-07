"""Product-path measurement + parity: ReceiptPipeline(hybrid_2p2b) on clean-100.

Two outputs in one run (~15 min, MLX weights required, NOT in CI):
 1. results/preds_product_hybrid_clean.jsonl + scores_product_hybrid_clean.json
    — the SHIPPED path's own clean-100 measurement (same scorer family as
    every other engine). THIS is the shippable number, whatever it is.
 2. results/product_parity_hybrid.json — per-doc field equality vs the
    phase-0 harness preds (committed F1 0.5985). Explains any delta
    (known systematic source: ReceiptPipeline.sanitize_fields, which the
    harness bypasses: repetition-collapse + whitespace + max-len truncation).
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "sdk"))
sys.path.insert(0, str(ROOT / "evaluation" / "phase0"))

from metrics import FIELDS, macro_prf, score_example
from tinydoc.pipeline import ReceiptPipeline

RESULTS = ROOT / "evaluation/phase0/results"


def main() -> int:
    ref = {r["id"]: r for r in
           map(json.loads, (RESULTS / "preds_hybrid-2.2b-tess_clean.jsonl")
               .read_text().splitlines()) if True}
    items = json.loads((RESULTS / "sroie_eval_clean.json").read_text())
    pipe = ReceiptPipeline("hybrid_2p2b")
    match = mismatch = 0
    mismatches = []
    lat, preds, scores = [], [], []
    for n, item in enumerate(items):
        t0 = time.time()
        doc = pipe.extract(item["image_path"], with_evidence=False)
        lat.append((time.time() - t0) * 1000)
        got = {k: doc.fields.get(k, "") for k in FIELDS}
        gold = {f: item["gold"].get(f, "") for f in FIELDS}
        sc = score_example(got, gold)
        preds.append({"id": item["id"], "image_path": item["image_path"],
                      "pred": got, "gold": gold, "latency_ms": lat[-1],
                      "error": "", "score": sc})
        scores.append(sc)
        want = ref[item["id"]]["pred"]
        if got == want:
            match += 1
        else:
            mismatch += 1
            if len(mismatches) < 5:
                mismatches.append({"id": item["id"], "product": got,
                                   "harness": want})
        if (n + 1) % 20 == 0 or n + 1 == len(items):
            agg = macro_prf(scores)
            print(f"  [{n + 1}/{len(items)}] f1={agg['field_f1']:.3f} "
                  f"match={match}/{n + 1}", flush=True)
    agg = macro_prf(scores)
    out_scores = {
        "engine": "receipt_pipeline:hybrid_2p2b",
        "n_examples": len(preds),
        "field_f1": agg["field_f1"],
        "field_precision": agg["field_precision"],
        "field_recall": agg["field_recall"],
        "schema_valid_rate": agg["schema_valid_rate"],
        "pipeline_schema_valid_rate": agg["schema_valid_rate"],
        "anls_mean": agg["anls_mean"],
        "exact_json_all_fields": agg["exact_json_all_fields"],
        "avg_latency_ms": sum(lat) / max(len(lat), 1),
        "per_field_hit_rate": {
            f: sum(1 for s in scores if s["fields"][f]["hit"]) / len(scores)
            for f in FIELDS
        },
        "errors": 0,
    }
    (RESULTS / "preds_product_hybrid_clean.jsonl").write_text(
        "\n".join(json.dumps(p) for p in preds) + "\n")
    (RESULTS / "scores_product_hybrid_clean.json").write_text(
        json.dumps(out_scores, indent=2))
    out = {"n": len(items), "match": match, "mismatch": mismatch,
           "match_rate": match / len(items),
           "product_field_f1": agg["field_f1"],
           "latency_ms_mean": round(statistics.mean(lat)),
           "latency_ms_median": round(statistics.median(lat)),
           "latency_ms_p90": round(sorted(lat)[int(len(lat) * 0.9)]),
           "mismatch_examples": mismatches}
    (RESULTS / "product_parity_hybrid.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
