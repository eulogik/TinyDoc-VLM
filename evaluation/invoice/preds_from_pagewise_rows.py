#!/usr/bin/env python3
"""Build full prediction files from page-wise rows, without dropping sanity flags.

`run_invoice_pagewise.py` excludes a document from its `preds` file whenever the
merge sanity check fires (for example, a receipt with no vendor and no invoice
number). That is useful for the synthetic multipage audit but wrong for a
field-level real-document probe: a parsed page with a correct total should still
score that total. This script includes every document with at least one parsed
page and writes two files:

- merged predictions: the product merge (`row["pred"]`)
- raw predictions: the first parsed page's JSON, normalized, before merge

Documents with no parsed page stay out, so the scorer counts every field wrong.

Usage:
    python evaluation/invoice/preds_from_pagewise_rows.py \
        --rows evaluation/invoice/results/rows_invoice_pt_smoke5_1400.jsonl \
        --out evaluation/invoice/results/preds_invoice_pt_smoke5_1400.json \
        --raw-out evaluation/invoice/results/preds_invoice_pt_smoke5_1400_raw.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "sdk"))

from run_invoice_eval import _extract_json, normalize_prediction
from tinydoc.pipeline import _extract_json as _extract_json_sdk


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--raw-out", required=True, type=Path)
    ap.add_argument(
        "--pred-key",
        default="pred",
        help="Row key to write as the merged prediction (use pred_first_pass for a retry comparison)",
    )
    args = ap.parse_args()

    rows = [json.loads(line) for line in args.rows.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"no rows in {args.rows}")

    merged_docs: dict[str, dict] = {}
    raw_docs: dict[str, dict] = {}
    no_page = []
    for row in rows:
        parsed_pages = [page for page in row.get("pages") or [] if page.get("error") is None]
        if not parsed_pages:
            no_page.append(row["id"])
            continue
        doc_id = str(row["id"])
        merged = row.get(args.pred_key)
        if merged:
            merged_docs[doc_id] = merged
        first = parsed_pages[0]
        parsed = _extract_json(first.get("raw") or "") or _extract_json_sdk(first.get("raw") or "")
        if parsed is not None:
            raw_docs[doc_id] = normalize_prediction(parsed, doc_id)

    model = rows[0].get("model")
    prompt_version = rows[0].get("prompt_version")
    run_date = rows[0].get("run_date")
    meta = {
        "system": f"TinyDoc engine eval ({model})",
        "run_date": run_date,
        "page_mode": "pagewise",
        "prompt_version": prompt_version,
        "notes": (
            "Built from saved page-wise rows without dropping merge-sanity flags; "
            "see evaluation/invoice/PT_REAL.md"
        ),
    }
    for path, documents in ((args.out, merged_docs), (args.raw_out, raw_docs)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**meta, "documents": documents}, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "documents": len(rows),
                "merged": len(merged_docs),
                "raw": len(raw_docs),
                "no_parsed_page": len(no_page),
                "no_parsed_page_ids": no_page,
                "out": str(args.out),
                "raw_out": str(args.raw_out),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
