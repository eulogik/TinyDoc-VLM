#!/usr/bin/env python3
"""Compare two invoice-scoring runs over the same documents.

The page-wise multipage method and the all-pages request use the same 10
documents, so this reports paired per-document changes: key-header accuracy,
fully-correct documents, line-item F1, latency, and transitions. It is
descriptive only; n=10 does not support a significance claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from compare_multipage import (
    KEY_FIELDS,
    _field_counts,
    _tally,
    latency_summary,
    sign_test,
)


def _load_rows(path: str | None) -> list[dict]:
    if not path:
        return []
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def compare_runs(
    before: dict,
    after: dict,
    before_rows: list[dict] | None = None,
    after_rows: list[dict] | None = None,
    before_label: str = "all_pages",
    after_label: str = "pagewise",
) -> dict:
    before_docs = {doc["doc_id"]: doc for doc in before.get("documents", [])}
    after_docs = {doc["doc_id"]: doc for doc in after.get("documents", [])}
    ids = sorted(set(before_docs) & set(after_docs))
    missing_before = sorted(set(after_docs) - set(before_docs))
    missing_after = sorted(set(before_docs) - set(after_docs))

    before_rows_by_id = {row["id"]: row for row in (before_rows or [])}
    after_rows_by_id = {row["id"]: row for row in (after_rows or [])}

    per_field = {}
    for name in KEY_FIELDS:
        b_correct, b_scored = _field_counts([before_docs[i] for i in ids], name)
        a_correct, a_scored = _field_counts([after_docs[i] for i in ids], name)
        per_field[name] = {
            before_label: b_correct / b_scored if b_scored else 0.0,
            after_label: a_correct / a_scored if a_scored else 0.0,
            "delta": (
                (a_correct / a_scored if a_scored else 0.0)
                - (b_correct / b_scored if b_scored else 0.0)
            ),
            f"{before_label}_correct": b_correct,
            f"{after_label}_correct": a_correct,
        }

    improved = regressed = same = 0
    gained = lost = 0
    documents = []
    for doc_id in ids:
        b = before_docs[doc_id]
        a = after_docs[doc_id]
        b_key = _tally([b])["key_correct"]
        a_key = _tally([a])["key_correct"]
        if a_key > b_key:
            improved += 1
        elif a_key < b_key:
            regressed += 1
        else:
            same += 1
        b_full = b.get("fully_correct") is True
        a_full = a.get("fully_correct") is True
        if a_full and not b_full:
            gained += 1
        elif b_full and not a_full:
            lost += 1
        documents.append(
            {
                "doc_id": doc_id,
                f"{before_label}_key_correct": b_key,
                f"{after_label}_key_correct": a_key,
                "key_delta": a_key - b_key,
                f"{before_label}_fully_correct": b_full,
                f"{after_label}_fully_correct": a_full,
                f"{before_label}_line_f1": b.get("line_f1"),
                f"{after_label}_line_f1": a.get("line_f1"),
                f"{before_label}_latency_ms": (before_rows_by_id.get(doc_id) or {}).get("latency_ms"),
                f"{after_label}_latency_ms": (after_rows_by_id.get(doc_id) or {}).get("latency_ms"),
            }
        )

    overall_before = _tally([before_docs[i] for i in ids])
    overall_after = _tally([after_docs[i] for i in ids])
    return {
        "n_docs": len(ids),
        "missing_from_before": missing_before,
        "missing_from_after": missing_after,
        "before": {before_label: overall_before},
        "after": {after_label: overall_after},
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
            "key_sign_p": sign_test(improved, regressed),
            "fully_correct_gained": gained,
            "fully_correct_lost": lost,
            "fully_correct_sign_p": sign_test(gained, lost),
        },
        "per_field": per_field,
        "latency": {
            before_label: latency_summary(before_rows or []),
            after_label: latency_summary(after_rows or []),
        },
        "documents": documents,
        "caveat": (
            "Descriptive only: same documents, two extraction methods, n<=10. "
            "No significance claim and no real-invoice transfer."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--before", required=True, help="scores JSON for the baseline run")
    ap.add_argument("--after", required=True, help="scores JSON for the comparison run")
    ap.add_argument("--before-rows", default=None)
    ap.add_argument("--after-rows", default=None)
    ap.add_argument("--before-label", default="all_pages")
    ap.add_argument("--after-label", default="pagewise")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    report = compare_runs(
        json.loads(Path(args.before).read_text()),
        json.loads(Path(args.after).read_text()),
        _load_rows(args.before_rows),
        _load_rows(args.after_rows),
        args.before_label,
        args.after_label,
    )
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("n_docs", "before", "after", "delta", "transitions")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
