#!/usr/bin/env python3
"""Compare two prompts on the same frozen manifest (paired field/doc counts).

Usage:
    python evaluation/invoice/compare_pt_prompts.py \
        --manifest evaluation/invoice/manifest_pt_pilot30_1400.jsonl \
        --a-preds evaluation/invoice/results/preds_invoice_pt_pilot30_v1.json \
        --b-preds evaluation/invoice/results/preds_invoice_pt_pilot30_v3_full.json \
        --a-name v1 --b-name v3 \
        --out evaluation/invoice/results/compare_pt_pilot30_v1_vs_v3.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pt_score import score_manifest


def _load(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    docs = raw.get("documents", raw) if isinstance(raw, dict) else raw
    if isinstance(docs, list):
        return {str(doc["doc_id"]): doc for doc in docs}
    return {str(key): value for key, value in docs.items() if value is not None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--a-preds", required=True, type=Path)
    ap.add_argument("--b-preds", required=True, type=Path)
    ap.add_argument("--a-name", default="A")
    ap.add_argument("--b-name", default="B")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    a = _load(args.a_preds)
    b = _load(args.b_preds)
    sa, suma = score_manifest(rows, a, system=args.a_name)
    sb, sumb = score_manifest(rows, b, system=args.b_name)

    per_field = {}
    for name in sorted(set(suma["per_field"]) | set(sumb["per_field"])):
        fa = suma["per_field"].get(name, {"correct": 0, "scored": 0})
        fb = sumb["per_field"].get(name, {"correct": 0, "scored": 0})
        per_field[name] = {
            "a_correct": fa["correct"],
            "b_correct": fb["correct"],
            "scored": fa["scored"],
            "delta": fb["correct"] - fa["correct"],
        }

    docs = {doc["doc_id"]: doc for doc in sa["documents"]}
    table = {"both": 0, "a_only": 0, "b_only": 0, "neither": 0}
    field_table = {"both": 0, "a_only": 0, "b_only": 0, "neither": 0}
    for doc_b in sb["documents"]:
        doc_a = docs[doc_b["doc_id"]]
        a_full = bool(doc_a["fully_correct"])
        b_full = bool(doc_b["fully_correct"])
        key = "both" if a_full and b_full else "a_only" if a_full else "b_only" if b_full else "neither"
        table[key] += 1
        for name, item_b in doc_b["fields"].items():
            item_a = doc_a["fields"].get(name)
            if item_a is None:
                continue
            a_ok, b_ok = bool(item_a["correct"]), bool(item_b["correct"])
            fkey = "both" if a_ok and b_ok else "a_only" if a_ok else "b_only" if b_ok else "neither"
            field_table[fkey] += 1

    payload = {
        "manifest": str(args.manifest),
        "a_name": args.a_name,
        "b_name": args.b_name,
        "documents": len(rows),
        "a": {
            "field_correct": suma["field_correct"],
            "field_scored": suma["field_scored"],
            "field_accuracy": suma["field_accuracy"],
            "fully_correct": suma["fully_correct"],
            "fully_correct_six": suma["fully_correct_six"],
            "six_scored": suma["six_scored"],
        },
        "b": {
            "field_correct": sumb["field_correct"],
            "field_scored": sumb["field_scored"],
            "field_accuracy": sumb["field_accuracy"],
            "fully_correct": sumb["fully_correct"],
            "fully_correct_six": sumb["fully_correct_six"],
            "six_scored": sumb["six_scored"],
        },
        "delta": {
            "field_accuracy": sumb["field_accuracy"] - suma["field_accuracy"],
            "fully_correct": sumb["fully_correct"] - suma["fully_correct"],
        },
        "per_field": per_field,
        "field_pairs": field_table,
        "doc_pairs": table,
        "note": "Paired on the same documents; fields still cluster within documents.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "documents": len(rows),
                "a_field": f"{suma['field_correct']}/{suma['field_scored']}",
                "b_field": f"{sumb['field_correct']}/{sumb['field_scored']}",
                "delta_pp": round(100 * payload["delta"]["field_accuracy"], 2),
                "a_fully": f"{suma['fully_correct']}/{len(rows)}",
                "b_fully": f"{sumb['fully_correct']}/{len(rows)}",
                "field_pairs": field_table,
                "doc_pairs": table,
                "out": str(args.out),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
