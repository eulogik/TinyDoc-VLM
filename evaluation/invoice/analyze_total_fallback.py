#!/usr/bin/env python3
"""Replay the deterministic single-amount total fallback on saved predictions.

Some Portuguese receipts (toll booths, card slips) print one amount and no
``Total`` line. The reader may put that amount in ``tax`` ("Taxa") or
``subtotal`` ("Compra"); :func:`tinydoc.invoice.apply_single_amount_total_fallback`
reclassifies it as the total only when exactly one amount exists. This script
measures that rule without new model calls:

- start from the saved first-pass prediction;
- apply the rule (if the final page did not fail);
- score before and after with the same scorer and manifest.

The output is a paired replay. A run with the shipped rule is still required
before treating the lift as held out.

Usage:
    python evaluation/invoice/analyze_total_fallback.py \
        --manifest evaluation/invoice/manifest_pt_retry100_1400.jsonl \
        --preds evaluation/invoice/results/preds_invoice_pt_retry100_1400_retry2400_firstpass_full.json \
        --out evaluation/invoice/results/total_fallback_replay_pt_retry100.json \
        --label "fresh seed44"
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "sdk"))

from pt_score import score_manifest
from tinydoc.invoice import apply_single_amount_total_fallback


def _summary(manifest: list[dict], predictions: dict[str, dict], name: str) -> dict:
    _, summary = score_manifest(manifest, predictions, system=name)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    manifest = [
        json.loads(line)
        for line in Path(args.manifest).read_text().splitlines()
        if line.strip()
    ]
    first = json.loads(Path(args.preds).read_text())["documents"]

    after: dict[str, dict] = {}
    changes: list[dict] = []
    for row in manifest:
        doc_id = row["id"]
        before = first.get(doc_id)
        if before is None:
            continue
        pred = copy.deepcopy(before)
        info = apply_single_amount_total_fallback(pred)
        after[doc_id] = pred
        if info:
            # The rule must never touch a field other than total / tax.
            changed = {
                key
                for key in set(before) | set(pred)
                if before.get(key) != pred.get(key)
            }
            unexpected = changed - {"total_amount", "tax", "subtotal"}
            if unexpected:
                raise SystemExit(f"{doc_id}: rule changed unexpected fields {unexpected}")
            changes.append({"doc_id": doc_id, **info})

    before_summary = _summary(manifest, first, "first pass (1400 px)")
    after_summary = _summary(
        manifest, after, "first pass + single-amount total fallback"
    )
    output = {
        "label": args.label,
        "documents": len(after),
        "manifest_documents": len(manifest),
        "triggered": len(changes),
        "changes": changes,
        "baseline": before_summary,
        "fallback": after_summary,
        "per_field": {
            name: {
                "before_correct": before_summary["per_field"][name]["correct"],
                "after_correct": after_summary["per_field"][name]["correct"],
                "scored": before_summary["per_field"][name]["scored"],
                "delta": after_summary["per_field"][name]["correct"]
                - before_summary["per_field"][name]["correct"],
            }
            for name in before_summary["per_field"]
        },
        "fully_correct_delta": after_summary["fully_correct"]
        - before_summary["fully_correct"],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
