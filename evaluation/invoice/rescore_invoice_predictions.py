#!/usr/bin/env python3
"""Re-normalize saved invoice predictions without model calls.

The first run scored currency symbols (`N$`, `R`, `€`) as non-ISO strings. This
re-parses each raw response, applies the current `normalize_prediction`, and
writes new rows/predictions so the benchmark scorer can be re-run. No inference.

Usage:
    python evaluation/invoice/rescore_invoice_predictions.py \
        --rows evaluation/invoice/results/rows_invoice_qwen2.5vl_3b.jsonl \
        --preds evaluation/invoice/results/preds_invoice_qwen2.5vl_3b.json \
        --out-preds evaluation/invoice/results/preds_invoice_qwen2.5vl_3b_v2.json \
        --out-rows evaluation/invoice/results/rows_invoice_qwen2.5vl_3b_v2.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "sdk"))

from run_invoice_eval import normalize_prediction
from tinydoc.pipeline import _extract_json


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out-preds", required=True)
    ap.add_argument("--out-rows", required=True)
    args = ap.parse_args()

    with Path(args.rows).open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    old = json.loads(Path(args.preds).read_text())

    predictions = {}
    changed = 0
    for row in rows:
        parsed = _extract_json(row.get("raw") or "") or row.get("pred") or {}
        normalized = normalize_prediction(parsed, str(row["id"]))
        if normalized.get("currency") != (row.get("pred") or {}).get("currency"):
            changed += 1
        row["pred"] = normalized
        predictions[row["id"]] = normalized

    out_preds = Path(args.out_preds)
    out_preds.parent.mkdir(parents=True, exist_ok=True)
    out_preds.write_text(
        json.dumps(
            {
                "system": old.get("system", "TinyDoc engine eval"),
                "run_date": old.get("run_date", time.strftime("%Y-%m-%d")),
                "notes": "Re-normalized currency symbols to ISO; no new inference. Page 1 only.",
                "documents": predictions,
            },
            indent=2,
        )
        + "\n"
    )

    out_rows = Path(args.out_rows)
    with out_rows.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    print(
        json.dumps(
            {
                "n": len(rows),
                "currency_changed": changed,
                "out_preds": str(out_preds),
                "out_rows": str(out_rows),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
