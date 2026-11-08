#!/usr/bin/env python3
"""Audit multipage invoice rows for silent context or output cuts.

A multipage result only counts if every page was sent, the model stopped on its
own (`done_reason=stop`), and prompt + output fit the context window. This
script checks those conditions against the manifest and exits non-zero on a
violation. It makes no model calls.

Usage:
    python evaluation/invoice/audit_multipage_rows.py \
        --manifest evaluation/invoice/manifest_multipage.jsonl \
        --rows evaluation/invoice/results/rows_invoice_qwen3-vl_2b_mp.jsonl \
        --num-ctx 16384 --num-predict 5120
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def audit_rows(
    manifest: list[dict],
    rows: list[dict],
    num_ctx: int = 16384,
    num_predict: int = 5120,
    allowed_failures: int = 0,
) -> dict:
    expected_pages = {item["id"]: len(item.get("image_paths") or []) for item in manifest}
    issues: list[str] = []
    failed_rows: list[dict] = []
    row_by_id = {row.get("id"): row for row in rows}

    for doc_id, pages in expected_pages.items():
        row = row_by_id.get(doc_id)
        if row is None:
            issues.append(f"{doc_id}: missing row")
            continue
        if pages <= 1:
            issues.append(f"{doc_id}: not a multipage manifest row ({pages} page)")
        sent = len(row.get("image_paths") or [])
        if sent != pages:
            issues.append(f"{doc_id}: sent {sent} images, manifest has {pages}")
        if row.get("error"):
            failed_rows.append(
                {
                    "id": doc_id,
                    "error": row["error"],
                    "done_reason": row.get("done_reason"),
                    "eval_count": row.get("eval_count"),
                }
            )
            continue
        if row.get("done_reason") != "stop":
            issues.append(f"{doc_id}: done_reason={row.get('done_reason')!r}, expected 'stop'")
        row_ctx = row.get("num_ctx") or num_ctx
        row_budget = row.get("num_predict") or num_predict
        prompt = row.get("prompt_eval_count")
        generated = row.get("eval_count")
        if not isinstance(prompt, int):
            issues.append(f"{doc_id}: missing prompt_eval_count")
        elif prompt >= row_ctx:
            issues.append(f"{doc_id}: prompt {prompt} >= context {row_ctx}")
        if not isinstance(generated, int):
            issues.append(f"{doc_id}: missing eval_count")
        elif generated >= row_budget:
            issues.append(f"{doc_id}: output {generated} >= budget {row_budget}")
        if isinstance(prompt, int) and isinstance(generated, int) and prompt + generated > row_ctx:
            issues.append(
                f"{doc_id}: prompt {prompt} + output {generated} > context {row_ctx}"
            )

    extra = sorted(set(row_by_id) - set(expected_pages))
    for doc_id in extra:
        issues.append(f"{doc_id}: row not in manifest")

    return {
        "n_docs": len(rows),
        "n_expected": len(expected_pages),
        "pages": sum(expected_pages.values()),
        "num_ctx_default": num_ctx,
        "num_predict_default": num_predict,
        "num_ctx_values": sorted(
            {row.get("num_ctx") for row in rows if row.get("num_ctx") is not None}
        ),
        "num_predict_values": sorted(
            {row.get("num_predict") for row in rows if row.get("num_predict") is not None}
        ),
        "failed_rows": failed_rows,
        "allowed_failures": allowed_failures,
        "ok": not issues and len(failed_rows) <= allowed_failures,
        "issues": issues,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--num-ctx", type=int, default=16384)
    ap.add_argument("--num-predict", type=int, default=5120)
    ap.add_argument(
        "--allow-failed",
        type=int,
        default=0,
        help="visible failed rows that are counted in the denominator (not silent cuts)",
    )
    args = ap.parse_args()

    manifest = [
        json.loads(line)
        for line in Path(args.manifest).read_text().splitlines()
        if line.strip()
    ]
    rows = [json.loads(line) for line in Path(args.rows).read_text().splitlines() if line.strip()]
    report = audit_rows(
        manifest, rows, args.num_ctx, args.num_predict, args.allow_failed
    )
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
