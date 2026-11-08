#!/usr/bin/env python3
"""Merge a targeted retry into the canonical multipage rows and predictions.

Long invoices can fail on one budget while shorter ones succeed. Rather than
re-running every document, retry only the failed rows with a larger context /
output budget and splice the successful retry into the canonical artifacts.

The merge refuses to write a result if any document still has an error, if a
retry row is itself an error, or if the manifest ids do not all appear.

Usage:
    python evaluation/invoice/merge_multipage_runs.py \
        --manifest evaluation/invoice/manifest_multipage.jsonl \
        --main-rows evaluation/invoice/results/archive/rows_..._partial.jsonl \
        --retry-rows evaluation/invoice/results/rows_..._8p.jsonl \
        --main-preds evaluation/invoice/results/archive/preds_..._partial.json \
        --retry-preds evaluation/invoice/results/preds_..._8p.json \
        --out-rows evaluation/invoice/results/rows_..._mp.jsonl \
        --out-preds evaluation/invoice/results/preds_..._mp.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def merge(
    manifest: list[dict],
    main_rows: list[dict],
    retry_rows: list[dict],
    main_preds: dict,
    retry_preds: dict,
) -> dict:
    expected = {item["id"]: len(item.get("image_paths") or []) for item in manifest}
    retry_by_id = {row["id"]: row for row in retry_rows}
    main_by_id = {row["id"]: row for row in main_rows}

    merged: list[dict] = []
    replaced: list[str] = []
    for doc_id in expected:
        primary = main_by_id.get(doc_id)
        retry = retry_by_id.get(doc_id)
        if primary is None and retry is None:
            merged.append({"id": doc_id, "error": "missing from both runs"})
            continue
        if primary is not None and not primary.get("error"):
            merged.append(primary)
        elif retry is not None and not retry.get("error"):
            merged.append(retry)
            replaced.append(doc_id)
        else:
            merged.append(retry or primary)  # keep the error visible

    remaining = [row["id"] for row in merged if row.get("error")]
    retry_errors = [row["id"] for row in retry_rows if row.get("error")]
    unknown = sorted(
        (set(retry_by_id) | set(main_by_id)) - set(expected)
    )

    documents = dict(main_preds.get("documents") or {})
    documents.update(retry_preds.get("documents") or {})
    missing_preds = sorted(set(expected) - set(documents))

    return {
        "rows": merged,
        "predictions": {
            "system": main_preds.get("system", "TinyDoc engine eval"),
            "run_date": main_preds.get("run_date"),
            "page_mode": "all_pages",
            "notes": (
                "All pages; synthetic benchmark; merged main + large-context retry "
                "for failed rows; see evaluation/invoice/README.md"
            ),
            "documents": {doc_id: documents[doc_id] for doc_id in expected if doc_id in documents},
        },
        "report": {
            "n_expected": len(expected),
            "n_rows": len(merged),
            "replaced": replaced,
            "remaining_errors": remaining,
            "retry_errors": retry_errors,
            "unknown_ids": unknown,
            "missing_predictions": missing_preds,
            "ok": not (remaining or retry_errors or unknown or missing_preds),
            "done_reasons": {
                row["id"]: row.get("done_reason") for row in merged if not row.get("error")
            },
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--main-rows", required=True)
    ap.add_argument("--retry-rows", required=True)
    ap.add_argument("--main-preds", required=True)
    ap.add_argument("--retry-preds", required=True)
    ap.add_argument("--out-rows", required=True)
    ap.add_argument("--out-preds", required=True)
    ap.add_argument("--out-report", default=None)
    args = ap.parse_args()

    def read_jsonl(path: str) -> list[dict]:
        return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

    manifest = read_jsonl(args.manifest)
    main_rows = read_jsonl(args.main_rows)
    retry_rows = read_jsonl(args.retry_rows)
    main_preds = json.loads(Path(args.main_preds).read_text())
    retry_preds = json.loads(Path(args.retry_preds).read_text())

    merged = merge(manifest, main_rows, retry_rows, main_preds, retry_preds)
    print(json.dumps(merged["report"], indent=2))
    if not merged["report"]["ok"]:
        return 1

    rows_path = Path(args.out_rows)
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    with rows_path.open("w") as handle:
        for row in merged["rows"]:
            handle.write(json.dumps(row) + "\n")

    preds_path = Path(args.out_preds)
    preds_path.write_text(json.dumps(merged["predictions"], indent=2) + "\n")

    if args.out_report:
        report_path = Path(args.out_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(merged["report"], indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
