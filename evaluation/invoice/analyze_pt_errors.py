#!/usr/bin/env python3
"""Categorize field errors in the real Portuguese probe (no model calls).

Reads the scorer output and the page-wise rows, then labels each wrong field so
the next iteration targets the actual failure modes: dot-decimal misreads,
tax-inclusive double counting, digit errors in long invoice numbers, brand vs
legal-entity vendor names, and wrong currency.

Usage:
    python evaluation/invoice/analyze_pt_errors.py \
        --scores evaluation/invoice/results/scores_invoice_pt_probe100_1400_full.json \
        --rows evaluation/invoice/results/rows_invoice_pt_probe100_1400.jsonl \
        --out evaluation/invoice/results/residuals_pt_probe100_1400.json \
        --markdown evaluation/invoice/results/residuals_pt_probe100_1400.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pt_score import norm_number, norm_vendor


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _close(a, b, tol=0.0101):
    return a is not None and b is not None and abs(a - b) <= tol


def _ratio(a, b):
    if a is None or b is None or a == 0:
        return None
    return b / a


def _edit_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def categorize(doc: dict, row: dict) -> list[dict]:
    fields = doc.get("fields") or {}
    pred = row.get("pred") or {}
    out = []

    total = fields.get("total")
    if total and not total.get("correct"):
        expected = _num(total.get("expected"))
        got = _num(total.get("got"))
        subtotal = _num(pred.get("subtotal"))
        tax = _num(pred.get("tax"))
        lines = [_num(item.get("amount")) for item in pred.get("line_items") or []]
        if got is None:
            kind = "total_missing"
        elif expected and _close(got, expected * 1000, tol=max(0.5, abs(expected) * 0.01)):
            kind = "total_dot_decimal_x1000"
        elif _close(got, (subtotal or 0) + (tax or 0)) and _close(subtotal, expected):
            kind = "total_double_counted_tax"
        elif any(_close(got, line) for line in lines if line is not None):
            kind = "total_selected_line_amount"
        else:
            kind = "total_other"
        out.append({"field": "total", "kind": kind, **total})

    tax = fields.get("tax")
    if tax and not tax.get("correct"):
        expected = _num(tax.get("expected"))
        got = _num(tax.get("got"))
        if got is None:
            kind = "tax_missing"
        elif expected and _close(got, expected * 1000, tol=max(0.5, abs(expected) * 0.01)):
            kind = "tax_dot_decimal_x1000"
        else:
            kind = "tax_other"
        out.append({"field": "tax", "kind": kind, **tax})

    number = fields.get("invoice_number")
    if number and not number.get("correct"):
        expected = norm_number(number.get("expected"))
        got = norm_number(number.get("got"))
        if not got:
            kind = "number_empty"
        elif _edit_ratio(expected, got) >= 0.8:
            kind = "number_digit_or_shape"
        else:
            kind = "number_other"
        out.append({"field": "invoice_number", "kind": kind, **number})

    vendor = fields.get("vendor_name")
    if vendor and not vendor.get("correct"):
        expected = norm_vendor(vendor.get("expected"))
        got = norm_vendor(vendor.get("got"))
        if not got:
            kind = "vendor_empty"
        elif expected and got and (expected in got or got in expected):
            kind = "vendor_alias_contains"
        else:
            kind = "vendor_other"
        out.append({"field": "vendor_name", "kind": kind, **vendor})

    currency = fields.get("currency")
    if currency and not currency.get("correct"):
        out.append({"field": "currency", "kind": "currency_wrong_iso", **currency})

    invoice_date = fields.get("invoice_date")
    if invoice_date and not invoice_date.get("correct"):
        out.append({"field": "invoice_date", "kind": "date_wrong", **invoice_date})

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", required=True, type=Path)
    ap.add_argument("--rows", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--markdown", required=True, type=Path)
    args = ap.parse_args()

    scores = json.loads(args.scores.read_text(encoding="utf-8"))
    rows = {
        str(row["id"]): row
        for row in (
            json.loads(line) for line in args.rows.read_text(encoding="utf-8").splitlines() if line.strip()
        )
    }
    counts = Counter()
    by_field = defaultdict(Counter)
    examples = defaultdict(list)
    docs = []
    for doc in scores.get("documents") or []:
        doc_id = str(doc["doc_id"])
        issues = categorize(doc, rows.get(doc_id) or {})
        if not issues:
            continue
        docs.append({"doc_id": doc_id, "category": doc.get("category"), "issues": issues})
        for issue in issues:
            counts[issue["kind"]] += 1
            by_field[issue["field"]][issue["kind"]] += 1
            if len(examples[issue["kind"]]) < 8:
                examples[issue["kind"]].append(
                    {
                        "doc_id": doc_id,
                        "field": issue["field"],
                        "expected": issue.get("expected"),
                        "got": issue.get("got"),
                    }
                )

    payload = {
        "scores": str(args.scores),
        "documents": len(scores.get("documents") or []),
        "documents_with_errors": len(docs),
        "counts": dict(sorted(counts.items())),
        "by_field": {field: dict(sorted(counter.items())) for field, counter in sorted(by_field.items())},
        "examples": dict(sorted(examples.items())),
        "docs": docs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Real Portuguese probe — residual categories",
        "",
        f"Documents: {payload['documents']}; with at least one wrong field: {payload['documents_with_errors']}.",
        "",
        "| Category | Count |",
        "|---|---:|",
    ]
    for kind, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{kind}` | {count} |")
    lines += ["", "## Examples", ""]
    for kind, items in sorted(examples.items()):
        lines.append(f"### `{kind}`")
        for item in items:
            lines.append(
                f"- `{item['doc_id']}` {item['field']}: expected `{item['expected']}`, got `{item['got']}`"
            )
        lines.append("")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "docs"}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
