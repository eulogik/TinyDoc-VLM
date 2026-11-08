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
) -> dict:
    expected_pages = {item["id"]: len(item.get("image_paths") or []) for item in manifest}
    issues: list[str] = []
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
            issues.append(f"{doc_id}: error {row['error']}")
        if row.get("done_reason") != "stop":
            issues.append(f"{doc_id}: done_reason={row.get('done_reason')!r}, expected 'stop'")
        prompt = row.get("prompt_eval_count")
        generated = row.get("eval_count")
        if not isinstance(prompt, int):
            issues.append(f"{doc_id}: missing prompt_eval_count")
        elif prompt >= num_ctx:
            issues.append(f"{doc_id}: prompt {prompt} >= context {num_ctx}")
        if not isinstance(generated, int):
            issues.append(f"{doc_id}: missing eval_count")
        elif generated >= num_predict:
            issues.append(f"{doc_id}: output {generated} >= budget {num_predict}")
        if isinstance(prompt, int) and isinstance(generated, int) and prompt + generated > num_ctx:
            issues.append(
                f"{doc_id}: prompt {prompt} + output {generated} > context {num_ctx}"
            )

    extra = sorted(set(row_by_id) - set(expected_pages))
    for doc_id in extra:
        issues.append(f"{doc_id}: row not in manifest")

    return {
        "n_docs": len(rows),
        "n_expected": len(expected_pages),
        "pages": sum(expected_pages.values()),
        "num_ctx": num_ctx,
        "num_predict": num_predict,
        "ok": not issues,
        "issues": issues,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--num-ctx", type=int, default=16384)
    ap.add_argument("--num-predict", type=int, default=5120)
    args = ap.parse_args()

    manifest = [
        json.loads(line)
        for line in Path(args.manifest).read_text().splitlines()
        if line.strip()
    ]
    rows = [json.loads(line) for line in Path(args.rows).read_text().splitlines() if line.strip()]
    report = audit_rows(manifest, rows, args.num_ctx, args.num_predict)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
