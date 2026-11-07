#!/usr/bin/env python3
"""Group invoice field misses into the failure modes quoted in the plan.

Reads the benchmark scorer output and the runner rows; writes a small JSON so
the residual taxonomy in `docs/v3_plan.md` is reproducible from artifacts.
Also evaluates a naive `sum(lines) == subtotal` / `subtotal + tax == total`
checker, which is deliberately expected to fail on adjustment-heavy invoices.

No model calls.

Usage:
    python evaluation/invoice/residual_report.py \
        --scores evaluation/invoice/results/scores_invoice_qwen2.5vl_3b.json \
        --preds evaluation/invoice/results/preds_invoice_qwen2.5vl_3b.json \
        --manifest evaluation/invoice/manifest_v1.jsonl \
        --out evaluation/invoice/results/residual_invoice_qwen2.5vl_3b.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

COUNTED = ("subtotal", "tax", "total", "currency")
GROUPS = (
    ("multipage", "page-1-only totals", lambda d: d["category"] == "multipage"),
    ("tax_inclusive", "gross used as subtotal", lambda d: d["doc_id"].startswith("tax-inclusive")),
    ("dual_currency", "wrong currency column", lambda d: d["doc_id"].startswith("currency-dual")),
    (
        "shipping_discount",
        "tax base with shipping/discount",
        lambda d: d["category"] == "clean"
        and ("shipping" in d["doc_id"] or "discount" in d["doc_id"]),
    ),
    (
        "multi_invoice",
        "one invoice's tax",
        lambda d: "multiple" in d["doc_id"],
    ),
    ("freight", "freight totals", lambda d: d["category"] == "freight"),
    ("photo", "photo blur/rotation digits", lambda d: d["category"] == "photo"),
    ("eu_decimal", "two-column decimal shift", lambda d: "two-column" in d["doc_id"]),
)


def _num(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def analyze(docs: list[dict], preds: dict, expected_flag: dict[str, bool]) -> dict:
    groups: dict[str, dict] = {
        key: {"label": label, "docs": 0, "field_misses": defaultdict(int)}
        for key, label, _ in GROUPS
    }
    ungrouped: defaultdict[str, int] = defaultdict(int)
    for doc in docs:
        matched = False
        for key, _label, predicate in GROUPS:
            if not predicate(doc):
                continue
            groups[key]["docs"] += 1
            matched = True
            for field in COUNTED:
                result = (doc.get("fields") or {}).get(field)
                if result is not None and not result.get("correct"):
                    groups[key]["field_misses"][field] += 1
        if not matched:
            for field in COUNTED:
                result = (doc.get("fields") or {}).get(field)
                if result is not None and not result.get("correct"):
                    ungrouped[field] += 1

    tp = fp = fn = tn = 0
    false_positives: list[str] = []
    missed: list[str] = []
    for doc in docs:
        doc_id = doc["doc_id"]
        pred = preds.get(doc_id) or {}
        subtotal = _num(pred.get("subtotal"))
        tax = _num(pred.get("tax")) or 0.0
        total = _num(pred.get("total_amount"))
        lines = [_num(item.get("amount")) for item in pred.get("line_items") or []]
        lines = [value for value in lines if value is not None]
        checks = []
        if subtotal is not None and len(lines) >= 2:
            checks.append(abs(sum(lines) - subtotal) > max(0.02, 0.005 * abs(subtotal)))
        if subtotal is not None and total is not None:
            checks.append(abs(subtotal + tax - total) > max(0.02, 0.005 * abs(total)))
        predicted = any(checks)
        gold = expected_flag.get(doc_id, False)
        if predicted and gold:
            tp += 1
        elif predicted:
            fp += 1
            false_positives.append(doc_id)
        elif gold:
            fn += 1
            missed.append(doc_id)
        else:
            tn += 1

    return {
        "groups": {
            key: {
                "label": value["label"],
                "docs": value["docs"],
                "field_misses": dict(sorted(value["field_misses"].items())),
            }
            for key, value in groups.items()
        },
        "ungrouped_field_misses": dict(sorted(ungrouped.items())),
        "naive_consistency_check": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "false_positives": false_positives,
            "missed_flagged_docs": missed,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    scores = json.loads(Path(args.scores).read_text())
    preds = json.loads(Path(args.preds).read_text())["documents"]
    with Path(args.manifest).open() as handle:
        manifest = [json.loads(line) for line in handle if line.strip()]
    expected_flag = {row["id"]: bool(row["expected"].get("should_flag")) for row in manifest}

    report = analyze(scores.get("documents") or [], preds, expected_flag)
    outgoing = {"scores": args.scores, "preds": args.preds, "manifest": args.manifest, **report}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(outgoing, indent=2) + "\n")
    print(json.dumps(report["groups"], indent=2))
    print(json.dumps(report["naive_consistency_check"], indent=2))
    print(f"report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
