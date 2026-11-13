#!/usr/bin/env python3
"""Replay the targeted high-resolution tax retry on saved predictions.

The 2400 px pilot showed that the extra resolution fixes exactly the
tax-inclusive receipts where the first pass reports the gross amount, the VAT
rate, or a truncated digit as ``tax``. This script evaluates the product rule
without new model calls:

- the trigger is :func:`tinydoc.invoice.tax_retry_reason` on the 1400 px
  prediction;
- the candidate is the saved 2400 px prediction's ``tax``;
- :func:`tinydoc.invoice.retry_tax_is_plausible` decides whether to keep it;
- everything except ``tax`` stays at 1400 px.

The output is a paired replay, not a new measurement. A run with the shipped
retry path is still required before treating the lift as held out.

Usage:
    python evaluation/invoice/analyze_tax_retry.py \
        --manifest evaluation/invoice/manifest_pt_holdout50_2400.jsonl \
        --first-pass evaluation/invoice/results/preds_invoice_pt_holdout50_1400.json \
        --retry evaluation/invoice/results/preds_invoice_pt_holdout50_2400_v1_full.json \
        --out evaluation/invoice/results/retry_replay_pt_holdout50.json
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

from pt_score import score_document, score_manifest
from tinydoc.invoice import retry_tax_is_plausible, tax_retry_reason


def _summary(manifest: list[dict], predictions: dict[str, dict], name: str) -> dict:
    _, summary = score_manifest(manifest, predictions, system=name)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--first-pass", required=True)
    ap.add_argument("--retry", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    manifest = [
        json.loads(line)
        for line in Path(args.manifest).read_text().splitlines()
        if line.strip()
    ]
    first = json.loads(Path(args.first_pass).read_text())["documents"]
    retry = json.loads(Path(args.retry).read_text())["documents"]

    # The counterfactual product output: first pass everywhere, a plausible
    # retry tax only where the structural trigger fired.
    combined: dict[str, dict] = {}
    changes: list[dict] = []
    triggered = 0
    applied = 0
    for row in manifest:
        doc_id = row["id"]
        before = first.get(doc_id)
        if before is None:
            continue
        after = copy.deepcopy(before)
        reason = tax_retry_reason(before)
        if reason:
            triggered += 1
            candidate = (retry.get(doc_id) or {}).get("tax")
            accepted = retry_tax_is_plausible(before, candidate)
            if accepted:
                after["tax"] = round(float(candidate), 2)
                applied += 1
            if reason or accepted:
                changes.append(
                    {
                        "doc_id": doc_id,
                        "trigger": reason,
                        "tax_before": before.get("tax"),
                        "tax_candidate": candidate,
                        "tax_after": after.get("tax"),
                        "expected": row["expected"].get("tax"),
                        "accepted": accepted,
                        "correct_before": score_document(
                            row["expected"], before
                        )["fields"].get("tax", {}).get("correct"),
                        "correct_after": score_document(
                            row["expected"], after
                        )["fields"].get("tax", {}).get("correct"),
                    }
                )
        combined[doc_id] = after

    base_summary = _summary(manifest, first, "first pass (1400 px)")
    retry_summary = _summary(manifest, combined, "targeted tax retry (2400 px)")
    base_fields = {k: v for k, v in base_summary["per_field"].items()}
    retry_fields = {k: v for k, v in retry_summary["per_field"].items()}
    per_field = {}
    for name in base_fields:
        base = base_fields[name]
        after = retry_fields.get(name, {"correct": 0, "scored": base["scored"]})
        per_field[name] = {
            "before_correct": base["correct"],
            "after_correct": after["correct"],
            "scored": base["scored"],
            "delta": after["correct"] - base["correct"],
        }

    gained = sum(
        1 for item in changes if item["correct_after"] and not item["correct_before"]
    )
    lost = sum(
        1 for item in changes if item["correct_before"] and not item["correct_after"]
    )
    result = {
        "label": args.label,
        "documents": len(combined),
        "triggered": triggered,
        "applied": applied,
        "baseline": {
            "field_correct": base_summary["field_correct"],
            "field_scored": base_summary["field_scored"],
            "field_accuracy": base_summary["field_accuracy"],
            "fully_correct": base_summary["fully_correct"],
            "fully_correct_six": base_summary["fully_correct_six"],
            "six_scored": base_summary["six_scored"],
        },
        "retry": {
            "field_correct": retry_summary["field_correct"],
            "field_scored": retry_summary["field_scored"],
            "field_accuracy": retry_summary["field_accuracy"],
            "fully_correct": retry_summary["fully_correct"],
            "fully_correct_six": retry_summary["fully_correct_six"],
            "six_scored": retry_summary["six_scored"],
        },
        "per_field": per_field,
        "fields_gained": gained,
        "fields_lost": lost,
        "changes": changes,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in result if k != "changes"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
