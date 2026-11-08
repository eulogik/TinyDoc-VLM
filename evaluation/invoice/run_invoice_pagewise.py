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

PAGE_PROMPT = """You are an invoice data extractor reading page {page} of {total} of ONE invoice. Reply with JSON only.
Keys:
- vendor_name: supplier name exactly as printed, or ""
- invoice_number: invoice number exactly as printed, or ""
- invoice_date: YYYY-MM-DD, or null
- due_date: YYYY-MM-DD, or null
- currency: ISO 4217 code (USD, EUR, GBP, ZAR, NAD, ZMW)
- document_type: "invoice", "credit_note", or "other"
- line_items: array of {{"description": "...", "amount": number}} for line items printed on THIS page only
- subtotal: net amount before tax printed on this page, or null
- tax: tax/VAT amount printed on this page, or null
- total: amount due printed on this page, or null

Rules:
- Extract only what is printed on THIS page. Never repeat items from another page.
- Exclude the rows "carried forward", "brought forward", "subtotal", "tax", "VAT",
  "sales tax", "total", "continued", and "page X of Y" from line_items.
- amount is the printed line total (quantity x unit price), never the unit price.
- If this is not the final page, subtotal/tax/total are normally null. Only use a
  printed page total when it is explicitly labelled on this page.
- Transcribe, do not guess. Use null for a value this page does not print.
- Dates: keep the printed day/month order; do not swap them.
- Amounts are numbers with a dot decimal and no currency symbols (e.g. 1234.56).
- JSON only, no markdown, no commentary."""


def page_prompt(page: int, total: int) -> str:
    return PAGE_PROMPT.format(page=page, total=total)


SUMMARY_ROW = re.compile(
    r"^(carried|brought)\s+forward$|^(sub\s*total|tax|vat|sales\s+tax|total|"
    r"amount\s+due|balance\s+due|continued|page\s+\d+\s+of\s+\d+)$",
    re.IGNORECASE,
)


def is_summary_row(description: str) -> bool:
    """True for running-total / tax / page labels that are not line items."""
    return bool(SUMMARY_ROW.match((description or "").strip()))


def _first(values):
    for value in values:
        if value not in (None, "", []):
            return value
    return None


def merge_pages(pages: list[dict], doc_id: str = "") -> dict:
    """Merge page-level extractions into one normalized invoice prediction."""
    pages = [page for page in pages if isinstance(page, dict)]
    currency = _currency(_first(page.get("currency") for page in pages))
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
    subtotal = _num(final.get("subtotal"))
    tax = _num(final.get("tax"))
    total = _num(final.get("total", final.get("total_amount")))
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
    return normalize_prediction(merged, doc_id)


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
    ap.add_argument("--host", default=os.environ.get("TINYDOC_OLLAMA_HOST", "http://127.0.0.1:11434"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--num-ctx", type=int, default=int(os.environ.get("TINYDOC_OLLAMA_NUM_CTX", "8192")))
    ap.add_argument("--num-predict", type=int, default=int(os.environ.get("TINYDOC_OLLAMA_NUM_PREDICT", "2048")))
    ap.add_argument("--timeout", type=float, default=float(os.environ.get("TINYDOC_OLLAMA_TIMEOUT", "600")))
    ap.add_argument("--resume", action="store_true")
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
                    page_prompt(page_number, len(image_paths)),
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
        sanity = merge_is_sane(merged, page_preds)
        if page_errors:
            sanity.extend(page_errors)
        row["pred"] = merged
        row["error"] = "; ".join(sanity) if sanity else None
        row["latency_ms"] = (time.time() - started_doc) * 1000
        rows.append(row)
        with out_path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(
            f"[{index}/{len(manifest)}] {doc_id} pages={len(image_paths)} "
            f"parsed={len(page_preds)} lines={len(merged.get('line_items') or [])} "
            f"total={merged.get('total_amount')} err={row['error']} "
            f"lat={row['latency_ms'] / 1000:.1f}s",
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

    latencies = [row["latency_ms"] for row in rows if row.get("latency_ms")]
    print(
        json.dumps(
            {
                "model": args.model,
                "n": len(rows),
                "parsed": sum(1 for row in rows if not row.get("error")),
                "errors": sum(1 for row in rows if row.get("error")),
                "p50_latency_ms": statistics.median(latencies) if latencies else None,
                "elapsed_s": time.time() - started,
                "rows": str(out_path),
                "preds": str(args.preds),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
