#!/usr/bin/env python3
"""Re-score existing preds_*.jsonl with current metrics (no model re-run).

Updates scores_*.json for Phase-0 engines and scores_pipeline_*.json +
pipeline_address_failures.json for the product path. Rebuilds baseline_table.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).absolute().parent))

from metrics import macro_prf, score_example, FIELDS  # noqa: E402
from run_baseline import merge_table  # noqa: E402

RESULTS = Path(__file__).absolute().parent / "results"


def rescore_engine(stem: str) -> dict:
    pred_path = RESULTS / f"preds_{stem}.jsonl"
    if not pred_path.exists():
        print(f"skip missing {pred_path.name}")
        return {}
    rows = [json.loads(l) for l in pred_path.open()]
    scores = [score_example(r["pred"], r["gold"]) for r in rows]
    # attach updated score into pred rows for address failure dumps
    for r, sc in zip(rows, scores):
        r["score"] = sc
    agg = macro_prf(scores)
    # preserve prior metadata from old score file if present
    old_path = RESULTS / f"scores_{stem}.json"
    old = json.loads(old_path.read_text()) if old_path.exists() else {}
    for k in (
        "engine",
        "eval_path",
        "limit",
        "avg_latency_ms",
        "split",
        "with_evidence",
        "pipeline_schema_valid_rate",
        "mean_confidence",
        "mean_evidence_coverage",
    ):
        if k in old:
            agg[k] = old[k]
    if "engine" not in agg:
        # derive from stem: ollama, ocr_regex, smolvlm2, pipeline_*
        agg["engine"] = stem
    agg["n_examples"] = len(scores)
    agg["rescored"] = True
    per = {}
    for f in FIELDS:
        hits = sum(1 for s in scores if s["fields"][f]["hit"])
        anls_v = sum(s["fields"][f]["anls"] for s in scores) / max(len(scores), 1)
        per[f] = {"hit_rate": hits / max(len(scores), 1), "anls_mean": anls_v}
    if old.get("per_field") is not None or stem.startswith("pipeline_"):
        agg["per_field"] = per
    old_path.write_text(json.dumps(agg, indent=2))
    # rewrite preds with updated scores (same fields; score block refreshed)
    with pred_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    # address failures for pipeline preds
    if stem.startswith("pipeline_"):
        fails = []
        for r in rows:
            sf = r["score"]["fields"]
            if not sf["address"]["hit"]:
                fails.append(
                    {
                        "id": r["id"],
                        "image_path": r["image_path"],
                        "pred": sf["address"]["pred"],
                        "gold": sf["address"]["gold"],
                        "anls": sf["address"]["anls"],
                        "pred_len": len(str(sf["address"]["pred"] or "")),
                        "gold_len": len(str(sf["address"]["gold"] or "")),
                    }
                )
        suffix = ""
        # pipeline_stem is pipeline_ollama_qwen2_5vl_3b or with _unseen
        base = stem[len("pipeline_") :] if stem.startswith("pipeline_") else stem
        if base.endswith("_unseen"):
            suffix = "_unseen"
        elif base.endswith("_seen"):
            suffix = "_seen"
        out = RESULTS / f"pipeline_address_failures{suffix}.json"
        out.write_text(json.dumps(fails, indent=2))
        print(f"  address_misses={len(fails)} → {out.name}")
    print(
        f"  {stem}: F1={agg['field_f1']:.3f} ANLS={agg['anls_mean']:.3f} "
        f"schema={agg['schema_valid_rate']:.3f} n={agg['n_examples']}"
    )
    return agg


def main() -> None:
    stems = []
    for p in sorted(RESULTS.glob("preds_*.jsonl")):
        stems.append(p.stem[len("preds_") :])
    print("rescoring:", stems)
    for s in stems:
        rescore_engine(s)
    merge_table()
    print("DONE")


if __name__ == "__main__":
    main()
