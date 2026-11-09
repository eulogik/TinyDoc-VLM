#!/usr/bin/env python3
"""Audit page-wise invoice rows for silent page cuts or missing pages.

A page-wise document only counts if every page was sent, every page parsed and
stopped on its own (`done_reason=stop`), and no page hit the output budget or the
context window. This script checks those conditions against the manifest and
exits non-zero on a violation. It makes no model calls.

Usage:
    python evaluation/invoice/audit_pagewise_rows.py \
        --manifest evaluation/invoice/manifest_multipage.jsonl \
        --rows evaluation/invoice/results/rows_invoice_qwen3-vl_2b_mp_pagewise.jsonl \
        --num-ctx 8192 --num-predict 2048
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def audit_rows(
    manifest: list[dict],
    rows: list[dict],
    num_ctx: int = 8192,
    num_predict: int = 2048,
    allowed_failures: int = 0,
) -> dict:
    expected_pages = {
        item["id"]: len(item.get("image_paths") or []) for item in manifest
    }
    issues: list[str] = []
    failed_rows: list[dict] = []
    row_by_id = {row.get("id"): row for row in rows}

    for doc_id, pages in expected_pages.items():
        row = row_by_id.get(doc_id)
        if row is None:
            issues.append(f"{doc_id}: missing row")
            continue
        sent = len(row.get("pages") or [])
        if sent != pages:
            issues.append(f"{doc_id}: {sent} page results, manifest has {pages}")
        if row.get("error"):
            failed_rows.append(
                {
                    "id": doc_id,
                    "error": row["error"],
                    "pages": sent,
                    "expected_pages": pages,
                }
            )
            continue
        pred = row.get("pred") or {}
        if not pred:
            issues.append(f"{doc_id}: no merged prediction")
        if len(pred.get("line_items") or []) > 250:
            issues.append(f"{doc_id}: runaway line-item count")
        for page in row.get("pages") or []:
            number = page.get("page")
            label = f"{doc_id} p{number}"
            if page.get("error"):
                issues.append(f"{label}: {page['error']}")
                continue
            if page.get("done_reason") != "stop":
                issues.append(
                    f"{label}: done_reason={page.get('done_reason')!r}, expected 'stop'"
                )
            prompt = page.get("prompt_eval_count")
            generated = page.get("eval_count")
            row_ctx = page.get("num_ctx") or num_ctx
            budget = page.get("num_predict") or num_predict
            if not isinstance(prompt, int):
                issues.append(f"{label}: missing prompt_eval_count")
            if not isinstance(generated, int):
                issues.append(f"{label}: missing eval_count")
            elif generated >= budget:
                issues.append(f"{label}: output {generated} >= budget {budget}")
            if isinstance(prompt, int) and isinstance(generated, int):
                if prompt >= row_ctx:
                    issues.append(f"{label}: prompt {prompt} >= context {row_ctx}")
                if prompt + generated > row_ctx:
                    issues.append(
                        f"{label}: prompt {prompt} + output {generated} > context {row_ctx}"
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
        "failed_rows": failed_rows,
        "allowed_failures": allowed_failures,
        "ok": not issues and len(failed_rows) <= allowed_failures,
        "issues": issues,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--num-predict", type=int, default=2048)
    ap.add_argument("--allow-failed", type=int, default=0)
    args = ap.parse_args()

    manifest = [
        json.loads(line)
        for line in Path(args.manifest).read_text().splitlines()
        if line.strip()
    ]
    rows = [
        json.loads(line)
        for line in Path(args.rows).read_text().splitlines()
        if line.strip()
    ]
    report = audit_rows(
        manifest, rows, args.num_ctx, args.num_predict, args.allow_failed
    )
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
