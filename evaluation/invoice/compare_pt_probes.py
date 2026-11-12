#!/usr/bin/env python3
"""Compare the development and holdout real-Portuguese probe summaries.

The two samples are disjoint, so this is an unpaired comparison. It reports the
point estimates, the difference, and both exact intervals; it does not claim a
paired win or a significance test on clustered fields.

Usage:
    python evaluation/invoice/compare_pt_probes.py \
        --dev evaluation/invoice/results/summary_invoice_pt_probe100_1400_mergefix5.json \
        --holdout evaluation/invoice/results/summary_invoice_pt_holdout100_1400_mergefix.json \
        --out evaluation/invoice/results/compare_pt_dev_vs_holdout.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _summary(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "n_docs": data["n_docs"],
        "field_accuracy": data["field_accuracy"],
        "field_correct": data["field_correct"],
        "field_scored": data["field_scored"],
        "fully_correct": data["fully_correct"],
        "fully_correct_rate": data["fully_correct_rate"],
        "fully_correct_six": data["fully_correct_six"],
        "six_scored": data["six_scored"],
        "fully_correct_six_rate": data["fully_correct_six_rate"],
        "ci95_exact": data["ci95_exact"],
        "per_field": data["per_field"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dev", required=True, type=Path)
    ap.add_argument("--holdout", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    dev = _summary(args.dev)
    holdout = _summary(args.holdout)
    per_field = {}
    for name in sorted(set(dev["per_field"]) | set(holdout["per_field"])):
        a = dev["per_field"].get(name) or {"correct": 0, "scored": 0, "accuracy": 0.0}
        b = holdout["per_field"].get(name) or {"correct": 0, "scored": 0, "accuracy": 0.0}
        per_field[name] = {
            "dev_correct": a["correct"],
            "dev_scored": a["scored"],
            "dev_accuracy": a["accuracy"],
            "holdout_correct": b["correct"],
            "holdout_scored": b["scored"],
            "holdout_accuracy": b["accuracy"],
            "delta_accuracy": b["accuracy"] - a["accuracy"],
        }

    payload = {
        "dev": dev,
        "holdout": holdout,
        "delta": {
            "field_accuracy": holdout["field_accuracy"] - dev["field_accuracy"],
            "fully_correct_rate": holdout["fully_correct_rate"] - dev["fully_correct_rate"],
            "fully_correct_six_rate": (
                (holdout["fully_correct_six_rate"] or 0.0)
                - (dev["fully_correct_six_rate"] or 0.0)
            ),
            "field_accuracy_pp": 100 * (holdout["field_accuracy"] - dev["field_accuracy"]),
            "fully_correct_pp": 100 * (holdout["fully_correct_rate"] - dev["fully_correct_rate"]),
        },
        "per_field": per_field,
        "note": (
            "Disjoint development and holdout samples; unpaired comparison. "
            "Field-level intervals are optimistic because fields cluster within documents."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "dev_field_accuracy": round(dev["field_accuracy"], 4),
                "holdout_field_accuracy": round(holdout["field_accuracy"], 4),
                "delta_pp": round(payload["delta"]["field_accuracy_pp"], 2),
                "dev_fully_correct": f"{dev['fully_correct']}/{dev['n_docs']}",
                "holdout_fully_correct": f"{holdout['fully_correct']}/{holdout['n_docs']}",
                "delta_pp_fully": round(payload["delta"]["fully_correct_pp"], 2),
                "out": str(args.out),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
