#!/usr/bin/env python3
"""Page-by-page invoice extraction with a deterministic merge.

The all-pages request works on short invoices but the 8-page, 110–117-line
documents failed with runaway generation. This runner sends **one page per
request**, asks for that page's line items and header fields, then merges the
pages in code:

- header fields: first non-empty value across pages (page 1 normally has them)
- line items: concatenated in page order
- subtotal / tax / total: the printed values from the final page, with a line-sum
  or arithmetic fallback when a printed value is missing

The prompt deliberately excludes "carried forward", "brought forward", and total
rows from line items, and asks for the line total rather than the unit price.
This is a method probe; it needs the same explicit approval as any multi-document
inference run (AGENTS.md).

Usage:
    python evaluation/invoice/run_invoice_pagewise.py \
        --manifest evaluation/invoice/manifest_multipage.jsonl \
        --model qwen3-vl:2b-instruct \
        --out evaluation/invoice/results/rows_invoice_qwen3-vl_2b_mp_pagewise.jsonl \
        --preds evaluation/invoice/results/preds_invoice_qwen3-vl_2b_mp_pagewise.json
"""

from __future__ import annotations

import argparse
import base64
import copy
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "sdk"))

from arithmetic_checks import check_prediction
from run_invoice_eval import (
    _check_host,
    _currency,
    _date,
    _doc_type,
    _extract_json,
    _num,
    _read_image,
    ask_ollama,
    normalize_prediction,
)

# The per-page prompt is owned by the SDK so the measured evaluation path and
# the shipped product path cannot drift (tests/test_invoice_prompt_parity.py).
from tinydoc.invoice import (
    PAGE_PROMPT_V2,
    page_prompt,
    retry_tax_is_plausible,
    tax_retry_reason,
)

PAGE_PROMPT = PAGE_PROMPT_V2


SUMMARY_ROW = re.compile(
    r"^(carried|brought)\s+forward$"
    r"|^(continued(\s+on\s+next\s+page)?|page\s+\d+\s+of\s+\d+)$"
    r"|^(sub\s*total|net\s+amount|tax|vat|sales\s+tax|total|amount\s+due|"
    r"balance\s+due|total\s+due)(\s*\(?\s*\d+(?:\.\d+)?\s*%?\s*\)?)?$",
    re.IGNORECASE,
)


def is_summary_row(description: str) -> bool:
    """True for running-total / tax / page labels that are not line items."""
    return bool(SUMMARY_ROW.match((description or "").strip()))


CURRENCY_SYMBOLS = {
    "N$": "NAD",
    "NAD$": "NAD",
    "NAD": "NAD",
    "USD": "USD",
    "EUR": "EUR",
    "GBP": "GBP",
    "R": "ZAR",
    "ZAR": "ZAR",
    "US$": "USD",
    "$": "USD",
    "€": "EUR",
    "£": "GBP",
    "K": "ZMW",
    "ZK": "ZMW",
    "ZMW": "ZMW",
}


def symbol_currency(value) -> str | None:
    """Map a printed symbol to ISO without guessing from the vendor name."""
    if value is None:
        return None
    return CURRENCY_SYMBOLS.get(str(value).strip().upper().replace(" ", ""))


def find_retry_image(directory: str | None, doc_id: str) -> str | None:
    """Return the prepared high-resolution image for one document, if any."""
    if not directory:
        return None
    base = Path(directory)
    for suffix in (".jpg", ".jpeg", ".png", ".webp"):
        candidate = base / f"{doc_id}{suffix}"
        if candidate.exists():
            return str(candidate)
    return None


def _first(values):
    for value in values:
        if value not in (None, "", []):
            return value
    return None


def _last(values):
    """Last non-empty value; the final page prints the document totals."""
    for value in reversed(list(values)):
        if value not in (None, "", []):
            return value
    return None


def merge_pages(pages: list[dict], doc_id: str = "") -> dict:
    """Merge page-level extractions into one normalized invoice prediction."""
    pages = [page for page in pages if isinstance(page, dict)]
    currency = symbol_currency(
        _last(page.get("currency_symbol") for page in pages)
    ) or _currency(_last(page.get("currency") for page in pages))
    lines: list[dict] = []
    for page in pages:
        for item in page.get("line_items") or []:
            if not isinstance(item, dict):
                continue
            amount = _num(item.get("amount", item.get("total", item.get("line_total"))))
            description = str(item.get("description", "") or "").strip()
            if is_summary_row(description):
                continue
            if not description and amount is None:
                continue
            lines.append({"description": description, "amount": amount})

    final = pages[-1] if pages else {}
    printed_subtotal = _num(final.get("subtotal"))
    subtotal = printed_subtotal
    tax = _num(final.get("tax"))
    printed_total = _num(final.get("total", final.get("total_amount")))
    total = printed_total
    line_sum = sum(item["amount"] for item in lines if item.get("amount") is not None)

    if subtotal is None and lines:
        subtotal = round(line_sum, 2)
    if total is None and subtotal is not None and tax is not None:
        total = round(subtotal + tax, 2)
    if tax is None and total is not None and subtotal is not None:
        tax = round(total - subtotal, 2)

    merged = {
        "vendor_name": str(_first(page.get("vendor_name") for page in pages) or ""),
        "invoice_number": str(_first(page.get("invoice_number") for page in pages) or ""),
        "invoice_date": _date(
            _first(page.get("invoice_date") for page in pages), doc_id, currency
        ),
        "due_date": _date(_first(page.get("due_date") for page in pages), doc_id, currency),
        "currency": currency,
        "subtotal": subtotal,
        "tax": tax,
        "total_amount": total,
        "document_type": _doc_type(_first(page.get("document_type") for page in pages)),
        "line_items": lines,
        "flagged": False,
        "references": [],
    }
    normalized = normalize_prediction(merged, doc_id)
    normalized, corrections = reconcile_totals(
        normalized,
        preserve_total=printed_total is not None,
        subtotal_is_printed=printed_subtotal is not None,
    )
    if corrections:
        normalized["corrections"] = corrections
    return normalized


def _tolerance(value: float, rel_tol: float = 0.001, abs_tol: float = 0.02) -> float:
    return max(abs_tol, rel_tol * max(1.0, abs(value)))


def reconcile_totals(
    merged: dict,
    rel_tol: float = 0.001,
    abs_tol: float = 0.02,
    preserve_total: bool = True,
    subtotal_is_printed: bool = True,
) -> tuple[dict, list[str]]:
    """Repair a total only when the extracted line items corroborate the subtotal.

    This is internal consistency, not gold. A printed total is authoritative:
    subtotal/tax repairs must stay consistent with it, and the total is replaced
    only when the page did not print one or the printed value is structurally
    impossible (for example, 12.00 under a printed 100.00 subtotal). The old
    unconditional
    ``total = subtotal + tax`` rewrite double-counted VAT on tax-inclusive
    receipts (50 of 100 real Portuguese receipts), so ``preserve_total`` defaults
    to true; callers pass false only when they know the total was derived.
    """
    corrections: list[str] = []
    lines = merged.get("line_items") or []
    amounts = [item.get("amount") for item in lines if item.get("amount") is not None]
    subtotal = merged.get("subtotal")
    tax = merged.get("tax")
    total = merged.get("total_amount")
    if not amounts or subtotal is None:
        return merged, corrections

    line_sum = round(sum(amounts), 2)

    # A printed subtotal can be a digit off while the total and tax are right.
    # Prefer the subtotal implied by total - tax when the line items corroborate
    # it; this is two independent internal checks, not gold.
    if tax is not None and total is not None:
        implied = round(total - tax, 2)
        corroborated = abs(line_sum - implied) <= _tolerance(implied, rel_tol, abs_tol)
        disagreement = subtotal is None or abs(subtotal - implied) > _tolerance(
            implied, rel_tol, abs_tol
        )
        if corroborated and disagreement:
            merged["subtotal"] = implied
            subtotal = implied
            corrections.append("subtotal_from_total_tax")

    # A small residual between subtotal + tax and total is usually a digit or
    # rounding error in the subtotal. Resolve it toward total - tax only when the
    # residual is tiny relative to the document, and record the correction.
    if tax is not None and total is not None and subtotal is not None:
        residual = (subtotal + tax) - total
        small_limit = max(0.05, 0.0001 * max(1.0, abs(total)))
        if 0.005 < abs(residual) <= small_limit:
            merged["subtotal"] = round(total - tax, 2)
            subtotal = merged["subtotal"]
            corrections.append("subtotal_from_total_tax_residual")

    # A nonzero tax below 0.1% of the total is almost always a leading-digit
    # truncation (e.g. 26.69 for 26269.37). Derive it from total - subtotal.
    if tax is not None and total is not None and subtotal is not None and tax > 0:
        implied_tax = round(total - subtotal, 2)
        if (
            abs(total) > abs(subtotal)
            and tax < 0.001 * abs(total)
            and implied_tax > 0.01 * abs(total)
            and abs(tax - implied_tax) > _tolerance(total, rel_tol, abs_tol)
        ):
            merged["tax"] = implied_tax
            tax = implied_tax
            corrections.append("tax_from_total_subtotal_truncated")

    if abs(line_sum - subtotal) > _tolerance(subtotal, rel_tol, abs_tol):
        return merged, corrections

    if tax is not None and total is not None and subtotal_is_printed:
        implied_tax = round(total - subtotal, 2)
        plausible = abs(total) > abs(subtotal) and abs(implied_tax) <= 0.5 * abs(subtotal)
        if plausible and abs(tax - implied_tax) > _tolerance(total, rel_tol, abs_tol):
            merged["tax"] = implied_tax
            tax = implied_tax
            corrections.append("tax_from_total_subtotal")

    if tax is not None:
        computed = round(subtotal + tax, 2)
        # A printed total below a printed subtotal by a wide margin is a
        # truncation (12.00 for 120.00), not a tax-inclusive receipt (where
        # subtotal and total are normally equal).
        impossible = (
            subtotal_is_printed
            and total is not None
            and (subtotal - total) > max(1.0, 0.05 * abs(subtotal))
        )
        if total is None:
            merged["total_amount"] = computed
            corrections.append("total_from_subtotal_tax")
        elif (impossible or not preserve_total) and abs(computed - total) > _tolerance(
            total, rel_tol, abs_tol
        ):
            merged["total_amount"] = computed
            corrections.append("total_from_subtotal_tax")
    elif total is not None:
        merged["tax"] = round(total - subtotal, 2)
        corrections.append("tax_from_total_subtotal")

    return merged, corrections


def merge_is_sane(merged: dict, pages: list[dict]) -> list[str]:
    """Cheap structural checks that do not use gold answers."""
    issues: list[str] = []
    if not pages:
        issues.append("no page parsed")
        return issues
    if not merged.get("vendor_name") and not merged.get("invoice_number"):
        issues.append("no vendor or invoice number")
    if merged.get("total_amount") is None:
        issues.append("no total")
    if len(merged.get("line_items") or []) > 250:
        issues.append("runaway line-item count")
    return issues


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument(
        "--preds-first-pass",
        default=None,
        help="Optional: write the pre-retry predictions for a paired comparison",
    )
    ap.add_argument("--host", default=os.environ.get("TINYDOC_OLLAMA_HOST", "http://127.0.0.1:11434"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--num-ctx", type=int, default=int(os.environ.get("TINYDOC_OLLAMA_NUM_CTX", "8192")))
    ap.add_argument("--num-predict", type=int, default=int(os.environ.get("TINYDOC_OLLAMA_NUM_PREDICT", "2048")))
    ap.add_argument("--timeout", type=float, default=float(os.environ.get("TINYDOC_OLLAMA_TIMEOUT", "600")))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--retry-tax-px",
        type=int,
        default=0,
        help=(
            "Re-read the final page at this resolution when the merged tax is "
            "structurally implausible (0 disables; 2400 matches the pilot)"
        ),
    )
    ap.add_argument(
        "--retry-images",
        default=None,
        help="Directory of <doc-id>.jpg/.png images prepared at --retry-tax-px",
    )
    ap.add_argument(
        "--prompt-version",
        type=int,
        choices=(1, 2, 3, 4),
        default=1,
        help=(
            "1 = original (canonical); 2 = currency-symbol variant (tested, not better); "
            "3 = header-first receipt experiment (rejected: loops); 4 = v1 + receipt rules"
        ),
    )
    args = ap.parse_args()

    _check_host(args.host)
    manifest = [
        json.loads(line)
        for line in Path(args.manifest).read_text().splitlines()
        if line.strip()
    ]
    if args.limit:
        manifest = manifest[: args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    rows: list[dict] = []
    if args.resume and out_path.exists():
        rows = [
            json.loads(line)
            for line in out_path.read_text().splitlines()
            if line.strip()
        ]
        done = {row["id"] for row in rows}
        print(f"resume: {len(done)} rows already done", flush=True)

    started = time.time()
    for index, item in enumerate(manifest, start=1):
        doc_id = item["id"]
        if doc_id in done:
            continue
        image_paths = [str(path) for path in item.get("image_paths") or []]
        if not image_paths and item.get("image_path"):
            image_paths = [str(item["image_path"])]
        row = {
            "id": doc_id,
            "category": item.get("category"),
            "difficulty": item.get("difficulty"),
            "model": args.model,
            "prompt_version": args.prompt_version,
            "run_date": time.strftime("%Y-%m-%d"),
            "num_ctx": args.num_ctx,
            "num_predict": args.num_predict,
            "image_paths": image_paths,
            "pages": [],
            "pred": None,
            "error": None,
            "latency_ms": None,
        }
        started_doc = time.time()
        page_preds: list[dict] = []
        page_errors: list[str] = []
        for page_number, image_path in enumerate(image_paths, start=1):
            raw = ""
            page_started = time.time()
            try:
                b64 = base64.b64encode(_read_image(Path(image_path))).decode()
                response = ask_ollama(
                    args.host,
                    args.model,
                    page_prompt(page_number, len(image_paths), args.prompt_version),
                    [b64],
                    args.num_ctx,
                    args.timeout,
                    args.num_predict,
                )
                raw = response["content"]
                parsed = _extract_json(raw)
                if parsed is None:
                    page_errors.append(f"page {page_number}: no JSON object")
                else:
                    page_preds.append(parsed)
                row["pages"].append(
                    {
                        "page": page_number,
                        "raw": raw,
                        "latency_ms": (time.time() - page_started) * 1000,
                        "prompt_eval_count": response.get("prompt_eval_count"),
                        "eval_count": response.get("eval_count"),
                        "done_reason": response.get("done_reason"),
                        "error": None if parsed is not None else "no JSON object",
                    }
                )
            except Exception as exc:  # noqa: BLE001 - record per-page failure and continue
                page_errors.append(f"page {page_number}: {type(exc).__name__}: {exc}")
                row["pages"].append(
                    {
                        "page": page_number,
                        "raw": raw,
                        "latency_ms": (time.time() - page_started) * 1000,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

        merged = merge_pages(page_preds, doc_id)
        row["pred_first_pass"] = (
            copy.deepcopy(merged) if args.retry_tax_px else None
        )
        retry = None
        if args.retry_tax_px:
            reason = tax_retry_reason(merged)
            if reason:
                retry = {
                    "trigger": reason,
                    "page": len(image_paths),
                    "px": args.retry_tax_px,
                    "tax_before": merged.get("tax"),
                    "applied": False,
                }
                retry_image = find_retry_image(args.retry_images, doc_id)
                if retry_image is None:
                    retry["error"] = "no retry image"
                else:
                    retry_started = time.time()
                    try:
                        b64 = base64.b64encode(
                            _read_image(Path(retry_image))
                        ).decode()
                        response = ask_ollama(
                            args.host,
                            args.model,
                            page_prompt(
                                len(image_paths),
                                len(image_paths),
                                args.prompt_version,
                            ),
                            [b64],
                            args.num_ctx,
                            args.timeout,
                            args.num_predict,
                        )
                        raw = response["content"]
                        parsed = _extract_json(raw)
                        retry["record"] = {
                            "page": len(image_paths),
                            "raw": raw,
                            "prompt_eval_count": response.get("prompt_eval_count"),
                            "eval_count": response.get("eval_count"),
                            "done_reason": response.get("done_reason"),
                            "error": (
                                None if parsed is not None else "no JSON object"
                            ),
                        }
                        candidate = (parsed or {}).get("tax")
                        retry["tax_candidate"] = candidate
                        if retry_tax_is_plausible(merged, candidate):
                            merged["tax"] = round(float(_num(candidate)), 2)
                            retry["applied"] = True
                            retry["tax_after"] = merged["tax"]
                    except Exception as exc:  # noqa: BLE001 - keep the first pass
                        retry["error"] = f"{type(exc).__name__}: {exc}"
                    retry["latency_ms"] = (time.time() - retry_started) * 1000
                row["retry"] = retry

        sanity = merge_is_sane(merged, page_preds)
        if page_errors:
            sanity.extend(page_errors)
        row["pred"] = merged
        row["review_flags"] = [issue["check"] for issue in check_prediction(merged)]
        row["error"] = "; ".join(sanity) if sanity else None
        row["latency_ms"] = (time.time() - started_doc) * 1000
        rows.append(row)
        with out_path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        retry_note = ""
        if retry:
            retry_note = (
                f" retry={retry['trigger']} applied={retry['applied']}"
            )
        print(
            f"[{index}/{len(manifest)}] {doc_id} pages={len(image_paths)} "
            f"parsed={len(page_preds)} lines={len(merged.get('line_items') or [])} "
            f"total={merged.get('total_amount')} err={row['error']} "
            f"lat={row['latency_ms'] / 1000:.1f}s{retry_note}",
            flush=True,
        )

    documents = {}
    for row in rows:
        if row.get("error"):
            continue
        documents[row["id"]] = row["pred"]
    preds = {
        "system": f"TinyDoc engine eval ({args.model})",
        "run_date": time.strftime("%Y-%m-%d"),
        "page_mode": "pagewise",
        "notes": (
            "Page-by-page extraction with deterministic merge; synthetic benchmark; "
            "see evaluation/invoice/README.md"
        ),
        "documents": documents,
    }
    Path(args.preds).parent.mkdir(parents=True, exist_ok=True)
    Path(args.preds).write_text(json.dumps(preds, indent=2) + "\n")

    if args.preds_first_pass:
        first_documents = {}
        for row in rows:
            if row.get("error"):
                continue
            if row.get("pred_first_pass") is not None:
                first_documents[row["id"]] = row["pred_first_pass"]
        first_preds = dict(preds)
        first_preds["system"] = f"TinyDoc engine eval first pass ({args.model})"
        first_preds["notes"] = (
            "Pre-retry page-wise predictions from the same run; used to measure "
            "the high-resolution tax retry."
        )
        first_preds["documents"] = first_documents
        Path(args.preds_first_pass).parent.mkdir(parents=True, exist_ok=True)
        Path(args.preds_first_pass).write_text(
            json.dumps(first_preds, indent=2) + "\n"
        )

    latencies = [row["latency_ms"] for row in rows if row.get("latency_ms")]
    retried = [row for row in rows if row.get("retry")]
    print(
        json.dumps(
            {
                "model": args.model,
                "n": len(rows),
                "parsed": sum(1 for row in rows if not row.get("error")),
                "errors": sum(1 for row in rows if row.get("error")),
                "retried": len(retried),
                "retry_applied": sum(
                    1 for row in retried if row["retry"].get("applied")
                ),
                "p50_latency_ms": statistics.median(latencies) if latencies else None,
                "elapsed_s": time.time() - started,
                "rows": str(out_path),
                "preds": str(args.preds),
                "preds_first_pass": args.preds_first_pass,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
