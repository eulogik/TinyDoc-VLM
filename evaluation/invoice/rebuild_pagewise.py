#!/usr/bin/env python3
"""Rebuild page-wise predictions from saved page raws, with no model calls.

The runner saves every page's raw response. If the deterministic merge changes
(for example, to add an internal-consistency total repair), re-parse those raws
and rebuild the predictions instead of re-running inference. This is also the
audit path: the saved raws and the merged prediction cannot drift apart.

Usage:
    python evaluation/invoice/rebuild_pagewise.py \
        --rows evaluation/invoice/results/rows_invoice_qwen3-vl_2b_mp_pagewise.jsonl \
        --out-rows evaluation/invoice/results/rows_invoice_qwen3-vl_2b_mp_pagewise_v2.jsonl \
        --out-preds evaluation/invoice/results/preds_invoice_qwen3-vl_2b_mp_pagewise_v2.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "sdk"))

from arithmetic_checks import check_prediction
from run_invoice_eval import _extract_json
from run_invoice_pagewise import merge_is_sane, merge_pages


def rebuild_row(row: dict) -> dict:
    rebuilt = dict(row)
    page_preds: list[dict] = []
    page_errors: list[str] = []
    for page in row.get("pages") or []:
        number = page.get("page")
        if page.get("error"):
            page_errors.append(f"page {number}: {page['error']}")
            continue
        parsed = _extract_json(page.get("raw") or "")
        if parsed is None:
            page_errors.append(f"page {number}: no JSON object")
            continue
        page_preds.append(parsed)

    merged = merge_pages(page_preds, row.get("id", ""))
    sanity = merge_is_sane(merged, page_preds)
    sanity.extend(page_errors)
    rebuilt["pred"] = merged
    rebuilt["review_flags"] = [
        issue["check"] for issue in check_prediction(merged)
    ]
    rebuilt["error"] = "; ".join(sanity) if sanity else None
    return rebuilt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out-rows", required=True)
    ap.add_argument("--out-preds", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--run-date", default=None)
    ap.add_argument("--out-report", default=None)
    args = ap.parse_args()

    rows = [
        json.loads(line)
        for line in Path(args.rows).read_text().splitlines()
        if line.strip()
    ]
    rebuilt = [rebuild_row(row) for row in rows]

    out_rows = Path(args.out_rows)
    out_rows.parent.mkdir(parents=True, exist_ok=True)
    with out_rows.open("w") as handle:
        for row in rebuilt:
            handle.write(json.dumps(row) + "\n")

    documents = {
        row["id"]: row["pred"] for row in rebuilt if not row.get("error")
    }
    model = args.model or (rebuilt[0].get("model") if rebuilt else None) or "unknown"
    run_date = (
        args.run_date
        or (rebuilt[0].get("run_date") if rebuilt else None)
        or time.strftime("%Y-%m-%d")
    )
    preds = {
        "system": f"TinyDoc engine eval ({model})",
        "run_date": run_date,
        "page_mode": "pagewise",
        "notes": (
            "Rebuilt from saved page raws with the current deterministic merge; "
            "no model calls; synthetic benchmark; see evaluation/invoice/README.md"
        ),
        "documents": documents,
    }
    out_preds = Path(args.out_preds)
    out_preds.parent.mkdir(parents=True, exist_ok=True)
    out_preds.write_text(json.dumps(preds, indent=2) + "\n")

    report = {
        "rows": len(rebuilt),
        "parsed": len(documents),
        "errors": sum(1 for row in rebuilt if row.get("error")),
        "review_flags": sum(1 for row in rebuilt if row.get("review_flags")),
        "corrections": sum(
            1
            for row in rebuilt
            if (row.get("pred") or {}).get("corrections")
        ),
        "out_rows": str(out_rows),
        "out_preds": str(out_preds),
    }
    if args.out_report:
        Path(args.out_report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
