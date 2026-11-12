#!/usr/bin/env python3
"""Score the real Portuguese invoice/receipt probe (field subset).

The matching rules mirror the Invoice Extraction Benchmark v1 scorer (MIT,
Copyright (c) 2026 InvoiceParser Pro) for the fields this probe has: fuzzy
vendor names, normalized invoice numbers, ISO dates, ISO currency, and
cent-exact money. Two deliberate extensions for Portuguese documents:

- legal-form stopwords include ``lda``, ``sa``, and ``unipessoal``;
- dates are parsed day-first, matching the Portuguese source.

Only the fields listed in a document's ``expected["scored"]`` are scored.
Missing gold values are never imputed.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

CENT = 0.005
LEGAL = {
    "lda",
    "sa",
    "unipessoal",
    "the",
    "company",
    "and",
    "co",
    "ltd",
    "limited",
    "inc",
    "llc",
    "plc",
    "gmbh",
}
EMPTY = {"", "n/a", "na", "none", "null", "unknown", "-"}
SIX_FIELDS = (
    "vendor_name",
    "invoice_number",
    "invoice_date",
    "currency",
    "tax",
    "total",
)


def ascii_text(value: Any) -> str:
    return unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode()


def norm_vendor(value: Any) -> str:
    text = ascii_text(value).casefold().replace("&", " and ")
    words = [word for word in re.findall(r"[a-z0-9]+", text) if word not in LEGAL]
    return " ".join(words)


def vendor_match(got: Any, expected: Any) -> tuple[bool, float]:
    a, b = norm_vendor(got), norm_vendor(expected)
    if not a or not b:
        return (not a and not b), 0.0
    if a == b:
        return True, 1.0
    ratio = SequenceMatcher(None, a, b).ratio()
    short, long_ = sorted((a, b), key=len)
    contains = short in long_ and len(short) >= 0.6 * len(long_)
    return (ratio >= 0.85 or contains), round(ratio, 3)


def norm_number(value: Any) -> str:
    text = ascii_text(value).upper().strip()
    text = re.sub(r"[\s#:]", "", text)
    return "" if text.lower() in EMPTY else text


def norm_date(value: Any) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}", text):
        return text[:10]
    from build_pt_manifest import parse_date

    return parse_date(text)


def parse_amount(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    from build_pt_manifest import parse_amount as parse_pt_amount

    return parse_pt_amount(value)


def score_money(got: Any, expected: float, *, none_is_zero: bool = False) -> tuple[bool, float | None, str]:
    number = parse_amount(got)
    if number is None and none_is_zero:
        number = 0.0
    if number is None:
        return False, None, "missing"
    diff = abs(number - float(expected))
    return diff < CENT, number, "" if diff < CENT else f"off by {diff:,.2f}"


def score_document(expected: dict[str, Any], prediction: dict[str, Any] | None) -> dict[str, Any]:
    fields: dict[str, dict[str, Any]] = {}
    scored = [name for name in expected.get("scored") or [] if name in SIX_FIELDS]
    for name in scored:
        want = expected.get(name)
        got = (prediction or {}).get(name)
        if name == "total":
            got = (prediction or {}).get("total_amount", got)
        detail = ""
        if name == "vendor_name":
            correct, ratio = vendor_match(got, want)
            detail = f"similarity {ratio}"
        elif name == "invoice_number":
            correct = norm_number(got) == norm_number(want)
        elif name == "invoice_date":
            got_norm = norm_date(got)
            correct = got_norm == want
            got = got_norm
        elif name == "currency":
            got_norm = str(got or "").strip().upper()
            correct = got_norm == str(want or "").strip().upper()
            got = got_norm
        elif name in ("tax", "total"):
            correct, got, detail = score_money(
                got,
                float(want),
                none_is_zero=(name == "tax" and float(want) == 0),
            )
        else:
            raise ValueError(f"unsupported field {name!r}")
        fields[name] = {
            "field": name,
            "correct": bool(correct),
            "expected": want,
            "got": got,
            "detail": detail,
        }
    fully = bool(fields) and all(result["correct"] for result in fields.values())
    six = len(scored) == len(SIX_FIELDS)
    return {
        "fields": fields,
        "fully_correct": fully,
        "all_six_scored": six,
        "fully_correct_six": fully if six else None,
    }


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> list[float] | None:
    if n <= 0:
        return None
    try:
        from scipy.stats import beta
    except Exception:  # noqa: BLE001 - optional dependency
        return None
    lower = 0.0 if k <= 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    upper = 1.0 if k >= n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return [lower, upper]


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(max(round(q * (len(ordered) - 1)), 0), len(ordered) - 1)
    return ordered[index]


def score_manifest(
    rows: Iterable[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    *,
    system: str = "",
    run_date: str = "",
    notes: str = "",
    latencies: list[float] | None = None,
    parse_errors: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    documents = []
    per_field: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "scored": 0})
    per_category: dict[str, dict[str, int]] = defaultdict(
        lambda: {"docs": 0, "correct": 0, "scored": 0, "fully_correct": 0, "fully_correct_six": 0, "six_scored": 0}
    )
    field_correct = field_scored = 0
    fully_correct = 0
    six_correct = six_scored = 0

    for row in rows:
        doc_id = str(row["id"])
        prediction = predictions.get(doc_id)
        result = score_document(row["expected"], prediction)
        fields = result["fields"]
        correct = sum(1 for item in fields.values() if item["correct"])
        scored = len(fields)
        field_correct += correct
        field_scored += scored
        fully_correct += int(result["fully_correct"])
        if result["all_six_scored"]:
            six_scored += 1
            six_correct += int(bool(result["fully_correct_six"]))
        category = row.get("category") or "unknown"
        cat = per_category[category]
        cat["docs"] += 1
        cat["correct"] += correct
        cat["scored"] += scored
        cat["fully_correct"] += int(result["fully_correct"])
        if result["all_six_scored"]:
            cat["six_scored"] += 1
            cat["fully_correct_six"] += int(bool(result["fully_correct_six"]))
        for name, item in fields.items():
            per_field[name]["scored"] += 1
            per_field[name]["correct"] += int(item["correct"])
        documents.append(
            {
                "doc_id": doc_id,
                "category": category,
                "difficulty": row.get("difficulty"),
                "predicted": prediction is not None,
                "fields": fields,
                "fully_correct": result["fully_correct"],
                "all_six_scored": result["all_six_scored"],
                "fully_correct_six": result["fully_correct_six"],
            }
        )

    per_category_out = {}
    for name, cat in sorted(per_category.items()):
        per_category_out[name] = {
            "docs": cat["docs"],
            "field_accuracy": cat["correct"] / cat["scored"] if cat["scored"] else 0.0,
            "field_correct": cat["correct"],
            "field_scored": cat["scored"],
            "fully_correct": cat["fully_correct"],
            "fully_correct_rate": cat["fully_correct"] / cat["docs"] if cat["docs"] else 0.0,
            "fully_correct_six": cat["fully_correct_six"],
            "six_scored": cat["six_scored"],
            "fully_correct_six_rate": (
                cat["fully_correct_six"] / cat["six_scored"] if cat["six_scored"] else None
            ),
        }

    scores = {
        "meta": {"system": system, "run_date": run_date, "notes": notes},
        "summary": {
            "documents": len(documents),
            "fields": {
                name: {"correct": value["correct"], "scored": value["scored"]}
                for name, value in sorted(per_field.items())
            },
            "field_correct": field_correct,
            "field_scored": field_scored,
            "field_accuracy": field_correct / field_scored if field_scored else 0.0,
            "fully_correct": fully_correct,
            "fully_correct_rate": fully_correct / len(documents) if documents else 0.0,
            "fully_correct_six": six_correct,
            "six_scored": six_scored,
            "fully_correct_six_rate": six_correct / six_scored if six_scored else None,
            "parse_errors": parse_errors,
        },
        "documents": documents,
    }
    summary = {
        "system": system,
        "run_date": run_date,
        "n_docs": len(documents),
        "field_accuracy": scores["summary"]["field_accuracy"],
        "field_correct": field_correct,
        "field_scored": field_scored,
        "fully_correct": fully_correct,
        "fully_correct_rate": scores["summary"]["fully_correct_rate"],
        "fully_correct_six": six_correct,
        "six_scored": six_scored,
        "fully_correct_six_rate": scores["summary"]["fully_correct_six_rate"],
        "ci95_exact": {
            "field_accuracy": clopper_pearson(field_correct, field_scored),
            "fully_correct_rate": clopper_pearson(fully_correct, len(documents)),
            "fully_correct_six_rate": clopper_pearson(six_correct, six_scored),
            "note": (
                "Clopper-Pearson. The field interval treats fields as independent "
                "and is optimistic because fields cluster within documents; cite "
                "the document-level intervals for fully-correct rates."
            ),
        },
        "per_field": {
            name: {
                "correct": value["correct"],
                "scored": value["scored"],
                "accuracy": value["correct"] / value["scored"] if value["scored"] else 0.0,
            }
            for name, value in sorted(per_field.items())
        },
        "per_category": per_category_out,
        "latency_ms": {
            "p50": _pct(latencies or [], 0.5),
            "p90": _pct(latencies or [], 0.9),
            "n": len(latencies or []),
        },
        "parse_errors": parse_errors,
    }
    return scores, summary


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    documents = raw.get("documents", raw) if isinstance(raw, dict) else raw
    if isinstance(documents, list):
        return {str(doc["doc_id"]): doc for doc in documents}
    return {str(key): value for key, value in documents.items() if value is not None}


def load_rows(path: Path) -> tuple[list[dict[str, Any]], int, list[float]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    errors = sum(1 for row in rows if row.get("error"))
    latencies = [float(row.get("latency_ms") or 0.0) for row in rows]
    return rows, errors, latencies


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--preds", required=True, type=Path)
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--scores", required=True, type=Path)
    ap.add_argument("--summary", required=True, type=Path)
    ap.add_argument("--system", default="")
    ap.add_argument("--run-date", default="")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    manifest = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    predictions = load_predictions(args.preds)
    _rows, errors, latencies = load_rows(args.rows)
    scores, summary = score_manifest(
        manifest,
        predictions,
        system=args.system or f"TinyDoc engine eval ({args.preds.stem})",
        run_date=args.run_date,
        notes=args.notes,
        latencies=latencies,
        parse_errors=errors,
    )
    write_json(args.scores, scores)
    write_json(args.summary, summary)
    print(
        json.dumps(
            {
                "documents": summary["n_docs"],
                "field_accuracy": round(summary["field_accuracy"], 4),
                "fully_correct": summary["fully_correct"],
                "fully_correct_six": summary["fully_correct_six"],
                "six_scored": summary["six_scored"],
                "parse_errors": errors,
                "p50_ms": summary["latency_ms"]["p50"],
                "scores": str(args.scores),
                "summary": str(args.summary),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
