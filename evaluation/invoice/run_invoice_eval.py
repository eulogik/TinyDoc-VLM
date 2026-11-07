#!/usr/bin/env python3
"""Run a TinyDoc engine over the invoice benchmark (Ollama, no cloud).

Writes both a per-document JSONL (raw output, latency, errors) and a
predictions.json in the benchmark's format. Score it with the benchmark's own
scorer:

    cd "/Volumes/KIOXIA 1TB/tinydoc/invoice-extraction-benchmark"
    python -m scorer.evaluate <predictions.json> --out scores.json \
        --markdown scores.md --allow-missing

Inference over multiple documents needs explicit approval (AGENTS.md).
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import os
import re
import statistics
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "sdk"))

from tinydoc.pipeline import _extract_json

INVOICE_PROMPT = """You are an invoice data extractor. Read the invoice image and reply with JSON only.
Keys:
- vendor_name: supplier name exactly as printed
- invoice_number: invoice number exactly as printed, or "" if none
- invoice_date: YYYY-MM-DD
- due_date: YYYY-MM-DD, or null if not printed
- currency: ISO 4217 code (USD, EUR, GBP, ZAR, NAD, ZMW)
- subtotal: net amount before tax, as a number
- tax: tax/VAT amount, as a number
- total: amount due, as a number
- document_type: "invoice", "credit_note", or "other"
- line_items: array of {"description": "...", "amount": number}
- flagged: true if the figures do not add up or the file holds several invoices
- references: array of trip, load, or container references

Rules:
- Transcribe, do not guess. Use null for a field the document does not print.
- invoice_number: copy every character, including hyphens and leading zeros.
  Watch lookalikes 0/O, 1/I/l, 5/S, 8/B.
- Dates: keep the printed day/month order; do not swap them.
- Amounts are numbers with a dot decimal and no currency symbols (e.g. 1234.56).
- Credit notes keep the printed sign; document_type carries the decision.
- JSON only, no markdown, no commentary."""

MULTIPAGE_RULES = """
- This request has several page images, in order. Read every page, not just the first.
- The invoice may continue across pages. Use the document-wide totals from the
  final/total page, not a page subtotal or a running subtotal.
- If the file holds several invoices, set flagged true.
"""


def multipage_prompt() -> str:
    return INVOICE_PROMPT.replace(
        "Read the invoice image", "Read every page image of the invoice"
    ) + MULTIPAGE_RULES


def _row_image_paths(row: dict) -> list[str]:
    paths = row.get("image_paths")
    if paths:
        return [str(path) for path in paths]
    if row.get("image_path"):
        return [str(row["image_path"])]
    raise ValueError(f"manifest row {row.get('id')!r} has no image path")


def _num(value):
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()").replace(" ", "")
    text = re.sub(r"[^0-9,.\-]", "", text)
    if not text:
        return None
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        # Treat a single comma as a decimal separator only when it ends in 2 digits.
        if re.search(r",\d{2}$", text):
            text = text.replace(",", ".")
        else:
            text = text.replace(",", "")
    try:
        number = float(text)
    except ValueError:
        return None
    return -number if negative else number


def _currency(value) -> str | None:
    """Normalize ISO codes and common currency symbols without answer hints."""
    if value is None:
        return None
    text = str(value).strip().upper().replace(" ", "")
    if not text:
        return None
    iso = re.search(r"\b(USD|EUR|GBP|ZAR|NAD|ZMW)\b", text)
    if iso:
        return iso.group(1)
    if text in ("N$", "NAD$", "N$D"):
        return "NAD"
    if text in ("R", "ZAR"):
        return "ZAR"
    if text in ("€", "EUR"):
        return "EUR"
    if text in ("£", "GBP"):
        return "GBP"
    if text in ("K", "ZK", "ZMW"):
        return "ZMW"
    if text in ("US$", "USD$"):
        return "USD"
    if "$" in text:
        return "USD"
    return text


def _date(value, doc_id: str = "", currency: str | None = None):
    """Normalize common invoice dates to ISO, resolving day/month ambiguity.

    The benchmark's scorer uses day-first parsing, which silently flips US
    month-first dates. We resolve ambiguity from the document id/currency:
    US documents and USD invoices are month-first, everything else day-first.
    """
    if value is None or value == "":
        return None
    if isinstance(value, dt.date):
        return value.isoformat()
    text = str(value).strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}", text):
        return text[:10]

    parts = re.findall(r"\d{1,4}", text)
    if len(parts) >= 3:
        a, b, c = parts[0], parts[1], parts[2]
        if len(a) == 4:
            year, month, day = a, b, c
        else:
            us_hint = "-us-" in doc_id.lower() or (currency or "").upper() == "USD"
            if int(a) > 12:
                day, month, year = a, b, c
            elif int(b) > 12 or us_hint:
                month, day, year = a, b, c
            else:
                day, month, year = a, b, c
            if len(year) == 2:
                year = "20" + year
        try:
            return dt.date(int(year), int(month), int(day)).isoformat()
        except ValueError:
            return None

    try:
        from dateutil import parser

        us_hint = "-us-" in doc_id.lower() or (currency or "").upper() == "USD"
        return parser.parse(text, dayfirst=not us_hint).date().isoformat()
    except Exception:  # noqa: BLE001 - leave unparseable text for the scorer
        return text


def _doc_type(value) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if text in ("credit", "credit_note", "creditnote", "cn"):
        return "credit_note"
    if text in ("statement", "other", "not_invoice", "notinvoice"):
        return "other"
    return "invoice"


def _lines(value) -> list[dict]:
    out = []
    for item in value or []:
        if not isinstance(item, dict):
            continue
        amount = _num(item.get("amount", item.get("total", item.get("line_total"))))
        out.append({"description": str(item.get("description", "") or ""), "amount": amount})
    return out


def normalize_prediction(obj: dict, doc_id: str = "") -> dict:
    obj = obj or {}
    currency = _currency(obj.get("currency"))
    return {
        "vendor_name": str(obj.get("vendor_name", obj.get("vendor", "")) or ""),
        "invoice_number": str(obj.get("invoice_number", obj.get("invoice_no", "")) or ""),
        "invoice_date": _date(obj.get("invoice_date"), doc_id, currency),
        "due_date": _date(obj.get("due_date"), doc_id, currency),
        "currency": currency,
        "subtotal": _num(obj.get("subtotal")),
        "tax": _num(obj.get("tax")),
        "total_amount": _num(obj.get("total", obj.get("total_amount"))),
        "document_type": _doc_type(obj.get("document_type")),
        "line_items": _lines(obj.get("line_items")),
        "flagged": bool(obj.get("flagged", False)),
        "references": obj.get("references") or [],
    }


def ask_ollama(
    host: str,
    model: str,
    prompt: str,
    images_b64: list[str],
    num_ctx: int,
    timeout: float,
) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": images_b64}],
        "stream": False,
        "options": {"temperature": 0, "num_predict": 768, "num_ctx": num_ctx},
    }
    req = urllib.request.Request(
        f"{host}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode())
    return (body.get("message") or {}).get("content", "") or ""


def _read_image(path: Path, attempts: int = 3, delay: float = 2.0) -> bytes:
    """Read an image, retrying transient external-disk permission errors."""
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return path.read_bytes()
        except OSError as exc:
            last = exc
            if attempt + 1 < attempts:
                time.sleep(delay * (attempt + 1))
    assert last is not None
    raise last


def _check_host(host: str, timeout: float = 10.0) -> None:
    """Fail loudly before a long run if Ollama is not answering."""
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout) as resp:
            resp.read(1024)
    except Exception as exc:  # noqa: BLE001 - surface the reason and stop
        raise SystemExit(f"Ollama not reachable at {host}: {type(exc).__name__}: {exc}")


def _load_resume_rows(path: Path) -> tuple[list[dict], int]:
    """Read completed rows; failed rows are returned as a retry count, not kept."""
    if not path.exists():
        return [], 0
    loaded: list[dict] = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                loaded.append(json.loads(line))
    done = [r for r in loaded if not r.get("error")]
    return done, len(loaded) - len(done)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", default="qwen2.5vl:3b")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    ap.add_argument(
        "--num-ctx", type=int, default=int(os.environ.get("TINYDOC_OLLAMA_NUM_CTX", "8192"))
    )
    ap.add_argument(
        "--timeout", type=float, default=float(os.environ.get("TINYDOC_OLLAMA_TIMEOUT", "600"))
    )
    ap.add_argument("--out", required=True, help="per-document JSONL")
    ap.add_argument("--preds", required=True, help="benchmark predictions.json")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="skip documents already present in --out and reuse their rows",
    )
    args = ap.parse_args()

    manifest = Path(args.manifest)
    with manifest.open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print(f"no rows in {manifest}", file=sys.stderr)
        return 2

    host = args.host.rstrip("/")
    _check_host(host)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_rows: list[dict] = []
    predictions: dict[str, dict] = {}
    latencies: list[float] = []
    if args.resume and out_path.exists():
        done_rows, failed = _load_resume_rows(out_path)
        # Rewrite with only successful rows so failed attempts are retried
        # instead of being frozen into the run by an outage.
        with out_path.open("w") as handle:
            for done in done_rows:
                handle.write(json.dumps(done) + "\n")
        out_rows.extend(done_rows)
        for done in done_rows:
            predictions[done["id"]] = done.get("pred") or {}
            latencies.append(float(done.get("latency_ms", 0.0)))
        done_ids = {r["id"] for r in done_rows}
        rows = [r for r in rows if r["id"] not in done_ids]
        print(
            f"resume: {len(done_ids)} done, {failed} failed will retry, {len(rows)} remaining",
            flush=True,
        )
        if not rows:
            print("nothing to do", flush=True)

    # Append as we go so a crash or disk hiccup never loses completed inference.
    total_docs = len(out_rows) + len(rows)
    with out_path.open("a") as sink:
        for row in rows:
            image_paths = _row_image_paths(row)
            images_b64 = [
                base64.b64encode(_read_image(Path(path))).decode() for path in image_paths
            ]
            prompt = INVOICE_PROMPT if len(images_b64) == 1 else multipage_prompt()
            t0 = time.time()
            raw = ""
            error = None
            pred = None
            try:
                raw = ask_ollama(
                    host, args.model, prompt, images_b64, args.num_ctx, args.timeout
                )
                pred = _extract_json(raw)
                if pred is None:
                    error = "no JSON object in response"
            except Exception as exc:  # noqa: BLE001 - keep the row, count it as missing
                error = f"{type(exc).__name__}: {exc}"
            latency = (time.time() - t0) * 1000
            latencies.append(latency)

            normalized = normalize_prediction(pred or {}, str(row["id"]))
            predictions[row["id"]] = normalized
            done_row = {
                "id": row["id"],
                "category": row.get("category"),
                "difficulty": row.get("difficulty"),
                "image_path": image_paths[0],
                "image_paths": image_paths,
                "raw": raw,
                "pred": normalized,
                "latency_ms": latency,
                "error": error,
            }
            out_rows.append(done_row)
            sink.write(json.dumps(done_row) + "\n")
            sink.flush()
            if len(out_rows) % 10 == 0:
                parsed = sum(1 for r in out_rows if r["error"] is None)
                print(
                    f"[{len(out_rows)}/{total_docs}] parsed={parsed}/{len(out_rows)} "
                    f"p50={statistics.median(latencies):.0f}ms",
                    flush=True,
                )

    preds_path = Path(args.preds)
    preds_path.parent.mkdir(parents=True, exist_ok=True)
    preds_path.write_text(
        json.dumps(
            {
                "system": f"TinyDoc engine eval ({args.model})",
                "run_date": time.strftime("%Y-%m-%d"),
                "notes": "Page 1 only; synthetic benchmark; see evaluation/invoice/README.md",
                "documents": predictions,
            },
            indent=2,
        )
        + "\n"
    )

    errors = sum(1 for r in out_rows if r["error"])
    print(
        json.dumps(
            {
                "model": args.model,
                "manifest": str(manifest),
                "n": len(out_rows),
                "parsed": len(out_rows) - errors,
                "errors": errors,
                "p50_latency_ms": statistics.median(latencies),
                "rows": str(out_path),
                "preds": str(preds_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
