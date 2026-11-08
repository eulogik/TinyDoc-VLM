#!/usr/bin/env python3
"""Deterministic arithmetic review checks for invoice predictions.

The model does not reliably set the benchmark's `flagged` field (0/7 on the
synthetic probe). Arithmetic is checkable without a model: if the printed
subtotal, tax, and total disagree, or the line items do not sum to the subtotal,
a human should look at the document. This module turns that into a small,
fixed rule set. It never changes extracted values and never sees gold answers.

Two checks:

- `subtotal_tax_total`: |subtotal + tax - total| > tolerance
- `line_sum_subtotal`: |sum(line amounts) - subtotal| > tolerance

Tolerance is `max(abs_tol, rel_tol * max(1, |expected|))`, so rounding noise on
large amounts does not fire. Credit notes keep their printed signs; a negative
document is not flagged for being negative.

This is a review-routing signal. It cannot catch a wrong VAT rate that still
adds up, or a multi-invoice file whose totals happen to reconcile.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        text = str(value).strip().replace(",", "")
        return float(text) if text else None
    except (TypeError, ValueError):
        return None


def _tolerance(expected: float, rel_tol: float, abs_tol: float) -> float:
    return max(abs_tol, rel_tol * max(1.0, abs(expected)))


def _line_total(lines: Any) -> float | None:
    if not isinstance(lines, list) or not lines:
        return None
    amounts = [
        _number(item.get("amount"))
        for item in lines
        if isinstance(item, dict)
    ]
    amounts = [amount for amount in amounts if amount is not None]
    if not amounts:
        return None
    return sum(amounts)


def check_prediction(
    prediction: dict[str, Any],
    rel_tol: float = 0.001,
    abs_tol: float = 0.02,
) -> list[dict[str, Any]]:
    """Return arithmetic issues for one normalized invoice prediction."""
    issues: list[dict[str, Any]] = []
    subtotal = _number(prediction.get("subtotal"))
    tax = _number(prediction.get("tax"))
    total = _number(prediction.get("total_amount", prediction.get("total")))

    if subtotal is not None and tax is not None and total is not None:
        diff = (subtotal + tax) - total
        limit = _tolerance(total, rel_tol, abs_tol)
        if abs(diff) > limit:
            issues.append(
                {
                    "check": "subtotal_tax_total",
                    "expected": round(total, 2),
                    "computed": round(subtotal + tax, 2),
                    "difference": round(diff, 2),
                    "tolerance": round(limit, 2),
                }
            )

    line_sum = _line_total(prediction.get("line_items"))
    if line_sum is not None and subtotal is not None:
        diff = line_sum - subtotal
        limit = _tolerance(subtotal, rel_tol, abs_tol)
        if abs(diff) > limit:
            issues.append(
                {
                    "check": "line_sum_subtotal",
                    "expected": round(subtotal, 2),
                    "computed": round(line_sum, 2),
                    "difference": round(diff, 2),
                    "tolerance": round(limit, 2),
                }
            )

    return issues


def flag_predictions(
    predictions: dict[str, dict[str, Any]],
    rel_tol: float = 0.001,
    abs_tol: float = 0.02,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Return a copy with arithmetic flags and a compact report."""
    out: dict[str, dict[str, Any]] = {}
    flagged: list[str] = []
    reasons: dict[str, list[str]] = {}
    for doc_id, prediction in predictions.items():
        issues = check_prediction(prediction, rel_tol, abs_tol)
        updated = dict(prediction)
        if issues:
            updated["flagged"] = True
            updated["flag_reasons"] = [issue["check"] for issue in issues]
            updated["arithmetic_issues"] = issues
            flagged.append(doc_id)
            reasons[doc_id] = [issue["check"] for issue in issues]
        out[doc_id] = updated
    return out, {"n": len(predictions), "flagged": flagged, "reasons": reasons}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds", required=True, help="benchmark predictions JSON")
    ap.add_argument("--out", required=True, help="flagged predictions JSON")
    ap.add_argument("--report", default=None, help="write the flag report JSON")
    ap.add_argument("--rel-tol", type=float, default=0.001)
    ap.add_argument("--abs-tol", type=float, default=0.02)
    args = ap.parse_args()

    payload = json.loads(Path(args.preds).read_text())
    documents = payload.get("documents") if isinstance(payload, dict) else payload
    if isinstance(documents, list):
        documents = {doc["doc_id"]: doc for doc in documents}
    flagged, report = flag_predictions(documents, args.rel_tol, args.abs_tol)

    out = dict(payload) if isinstance(payload, dict) else {}
    out["documents"] = flagged
    out["notes"] = (out.get("notes") or "") + " Arithmetic review flags applied; values unchanged."
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
