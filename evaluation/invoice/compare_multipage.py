#!/usr/bin/env python3
"""Compare page-1-only and all-pages invoice scores on multipage documents.

Both inputs are scorer outputs from the Invoice Extraction Benchmark v1. The
page-1 run covers all 181 documents; the all-pages run covers only the 10
`multipage-*` documents, scored with `--allow-missing`. This script keeps the
overlap and reports the paired change.

**Descriptive only.** Ten documents are too few for a significance claim; this
answers "did reading every page fix the multipage failures?" and nothing more.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

KEY_FIELDS = (
    "vendor_name",
    "invoice_number",
    "invoice_date",
    "due_date",
    "currency",
    "subtotal",
    "tax",
    "total",
    "document_type",
)


def _key_counts(document: dict[str, Any]) -> tuple[int, int]:
    fields = document.get("fields") or {}
    correct = sum(1 for name in KEY_FIELDS if (fields.get(name) or {}).get("correct") is True)
    scored = sum(1 for name in KEY_FIELDS if name in fields)
    return correct, scored


def _field_counts(documents: Sequence[dict[str, Any]], name: str) -> tuple[int, int]:
    correct = scored = 0
    for doc in documents:
        result = (doc.get("fields") or {}).get(name)
        if result is None:
            continue
        scored += 1
        correct += 1 if result.get("correct") is True else 0
    return correct, scored


def _tally(documents: Sequence[dict[str, Any]]) -> dict[str, Any]:
    key_correct = sum(_key_counts(doc)[0] for doc in documents)
    key_scored = sum(_key_counts(doc)[1] for doc in documents)
    fully = sum(1 for doc in documents if doc.get("fully_correct") is True)
    line_f1 = [doc["line_f1"] for doc in documents if isinstance(doc.get("line_f1"), (int, float))]
    return {
        "documents": len(documents),
        "key_correct": key_correct,
        "key_scored": key_scored,
        "key_accuracy": key_correct / key_scored if key_scored else 0.0,
        "fully_correct": fully,
        "fully_correct_rate": fully / len(documents) if documents else 0.0,
        "mean_line_f1": statistics.mean(line_f1) if line_f1 else None,
    }


def latency_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    pairs = [
        (float(row["latency_ms"]) / 1000.0, len(row.get("image_paths") or []) or 1)
        for row in rows
        if isinstance(row.get("latency_ms"), (int, float))
    ]
    if not pairs:
        return {"n": 0, "p50_s": None, "mean_s": None, "p50_s_per_page": None}
    per_doc = [seconds for seconds, _ in pairs]
    per_page = [seconds / pages for seconds, pages in pairs]
    return {
        "n": len(pairs),
        "p50_s": statistics.median(per_doc),
        "mean_s": statistics.mean(per_doc),
        "p50_s_per_page": statistics.median(per_page),
    }


def compare(
    page1: dict[str, Any],
    multipage: dict[str, Any],
    prefix: str = "multipage-",
    rows: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    base = {doc["doc_id"]: doc for doc in page1.get("documents", [])}
    selected = [
        doc for doc in multipage.get("documents", []) if doc.get("doc_id", "").startswith(prefix)
    ]
    paired = [doc for doc in selected if doc["doc_id"] in base]
    missing = [doc["doc_id"] for doc in selected if doc["doc_id"] not in base]

    before = [_tally([base[doc["doc_id"]]]) for doc in paired]
    after = [_tally([doc]) for doc in paired]
    page1_docs = [base[doc["doc_id"]] for doc in paired]

    per_field: dict[str, dict[str, Any]] = {}
    for name in KEY_FIELDS:
        a_correct, a_scored = _field_counts(page1_docs, name)
        b_correct, b_scored = _field_counts(paired, name)
        per_field[name] = {
            "page1": a_correct / a_scored if a_scored else 0.0,
            "all_pages": b_correct / b_scored if b_scored else 0.0,
            "delta": (b_correct / b_scored if b_scored else 0.0)
            - (a_correct / a_scored if a_scored else 0.0),
            "page1_correct": a_correct,
            "all_pages_correct": b_correct,
        }

    rows = []
    improved = regressed = same = 0
    for doc, b, a in zip(paired, before, after):
        key_delta = a["key_correct"] - b["key_correct"]
        fully_before = base[doc["doc_id"]].get("fully_correct") is True
        fully_after = doc.get("fully_correct") is True
        if key_delta > 0:
            improved += 1
        elif key_delta < 0:
            regressed += 1
        else:
            same += 1
        rows.append(
            {
                "doc_id": doc["doc_id"],
                "page1_key_correct": b["key_correct"],
                "all_pages_key_correct": a["key_correct"],
                "key_delta": key_delta,
                "page1_fully_correct": fully_before,
                "all_pages_fully_correct": fully_after,
                "page1_line_f1": base[doc["doc_id"]].get("line_f1"),
                "all_pages_line_f1": doc.get("line_f1"),
            }
        )

    overall_before = _tally(page1_docs)
    overall_after = _tally(paired)
    return {
        "n_docs": len(paired),
        "missing_from_page1_scores": missing,
        "page1": overall_before,
        "all_pages": overall_after,
        "delta": {
            "key_accuracy": overall_after["key_accuracy"] - overall_before["key_accuracy"],
            "fully_correct_rate": overall_after["fully_correct_rate"]
            - overall_before["fully_correct_rate"],
            "mean_line_f1": (
                None
                if overall_before["mean_line_f1"] is None or overall_after["mean_line_f1"] is None
                else overall_after["mean_line_f1"] - overall_before["mean_line_f1"]
            ),
        },
        "transitions": {
            "key_improved": improved,
            "key_regressed": regressed,
            "key_unchanged": same,
            "fully_correct_gained": sum(
                1 for b, a in zip(before, after) if a["fully_correct"] and not b["fully_correct"]
            ),
            "fully_correct_lost": sum(
                1 for b, a in zip(before, after) if b["fully_correct"] and not a["fully_correct"]
            ),
        },
        "per_field": per_field,
        "latency": latency_summary(rows) if rows is not None else None,
        "documents": rows,
        "caveat": (
            "Descriptive only: 10 synthetic multipage documents, page 1 vs all pages. "
            "No significance claim, no real-invoice transfer."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--page1", required=True, help="scores JSON from the page-1 run")
    ap.add_argument("--multipage", required=True, help="scores JSON from the all-pages run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="multipage-")
    ap.add_argument("--rows", default=None, help="per-document JSONL for latency")
    args = ap.parse_args()

    page1 = json.loads(Path(args.page1).read_text())
    multipage = json.loads(Path(args.multipage).read_text())
    rows = (
        [json.loads(line) for line in Path(args.rows).read_text().splitlines() if line.strip()]
        if args.rows
        else None
    )
    report = compare(page1, multipage, args.prefix, rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("n_docs", "page1", "all_pages", "delta")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
