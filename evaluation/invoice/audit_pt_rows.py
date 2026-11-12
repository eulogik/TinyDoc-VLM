#!/usr/bin/env python3
"""Audit page-wise rows for the real Portuguese probe (no model calls).

Hard failures are missing documents, missing pages, page-level errors, empty
raw output, a non-`stop` completion, or an output-budget hit. Merge sanity flags
("no vendor or invoice number") are recorded but do not hide a parsed page; the
field scorer counts the missing field.

Usage:
    python evaluation/invoice/audit_pt_rows.py \
        --manifest evaluation/invoice/manifest_pt_real_probe100_1400.jsonl \
        --rows evaluation/invoice/results/rows_invoice_pt_probe100_1400.jsonl \
        --out evaluation/invoice/results/audit_invoice_pt_probe100_1400.json \
        --allow-failed 0
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--rows", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--allow-failed", type=int, default=0)
    args = ap.parse_args()

    manifest = [
        json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    rows = [json.loads(line) for line in args.rows.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {str(row["id"]): row for row in rows}

    failures: list[dict] = []
    missing: list[str] = []
    sanity: list[dict] = []
    reasons = Counter()
    latencies: list[float] = []
    prompt_tokens: list[int] = []
    output_tokens: list[int] = []
    truncated: list[dict] = []

    for doc in manifest:
        doc_id = str(doc["id"])
        row = by_id.get(doc_id)
        if row is None:
            missing.append(doc_id)
            continue
        pages = row.get("pages") or []
        expected_pages = len(row.get("image_paths") or [doc.get("image_path")])
        if len(pages) != expected_pages:
            failures.append(
                {
                    "doc_id": doc_id,
                    "check": "page_count",
                    "detail": f"{len(pages)} rows for {expected_pages} page(s)",
                }
            )
        for page in pages:
            number = page.get("page")
            reason = page.get("done_reason")
            reasons[str(reason)] += 1
            prompt_count = page.get("prompt_eval_count")
            eval_count = page.get("eval_count")
            if isinstance(prompt_count, int):
                prompt_tokens.append(prompt_count)
            if isinstance(eval_count, int):
                output_tokens.append(eval_count)
            cap = int(row.get("num_predict") or 0)
            if isinstance(eval_count, int) and cap > 0 and eval_count >= cap:
                truncated.append(
                    {"doc_id": doc_id, "page": number, "eval_count": eval_count, "done_reason": reason}
                )
            if page.get("error"):
                detail = page["error"]
                if reason:
                    detail = f"{detail}; done_reason={reason}"
                failures.append(
                    {"doc_id": doc_id, "page": number, "check": "page_error", "detail": detail}
                )
                continue
            raw = page.get("raw") or ""
            if not raw.strip():
                failures.append(
                    {"doc_id": doc_id, "page": number, "check": "empty_raw", "detail": ""}
                )
            if reason != "stop":
                failures.append(
                    {"doc_id": doc_id, "page": number, "check": "done_reason", "detail": str(reason)}
                )
        if row.get("error"):
            sanity.append({"doc_id": doc_id, "error": row["error"]})
        if row.get("latency_ms"):
            latencies.append(float(row["latency_ms"]))

    payload = {
        "manifest": str(args.manifest),
        "rows_path": str(args.rows),
        "documents": len(manifest),
        "rows": len(rows),
        "missing": missing,
        "failed": len(failures),
        "allowed_failed": args.allow_failed,
        "failures": failures,
        "merge_sanity": sanity,
        "done_reasons": dict(sorted(reasons.items())),
        "truncated_pages": truncated,
        "latency_ms": {
            "p50": statistics.median(latencies) if latencies else None,
            "p90": sorted(latencies)[round(0.9 * (len(latencies) - 1))] if latencies else None,
            "n": len(latencies),
        },
        "tokens": {
            "prompt_p50": statistics.median(prompt_tokens) if prompt_tokens else None,
            "prompt_max": max(prompt_tokens) if prompt_tokens else None,
            "output_p50": statistics.median(output_tokens) if output_tokens else None,
            "output_max": max(output_tokens) if output_tokens else None,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "documents": payload["documents"],
                "rows": payload["rows"],
                "missing": len(missing),
                "failed": len(failures),
                "allowed_failed": args.allow_failed,
                "merge_sanity": len(sanity),
                "done_reasons": payload["done_reasons"],
                "latency_p50_ms": payload["latency_ms"]["p50"],
                "task": str(args.out),
            },
            indent=2,
        )
    )
    return 0 if len(failures) <= args.allow_failed and not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
