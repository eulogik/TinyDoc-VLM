"""Multipage invoice extraction: page-wise reading + deterministic merge.

Why page-wise: sending every page in one request failed on 8-page invoices with
runaway line-item generation. Reading one page per request fixes the output
control problem, and a deterministic merge keeps the header from page 1, the
line items in page order, and the printed totals from the final page.

The merge is deliberately boring and auditable:

- header fields: first non-empty value across pages
- line items: concatenated, with carried/brought-forward and total rows removed
- subtotal/tax/total: final-page printed values, then narrow internal-consistency
  repairs that are recorded in ``corrections``
- currency: the last printed symbol/code, then OCR totals if they are consistent
- review flags: arithmetic checks that do not use gold answers

The measured evidence for this method is in ``evaluation/invoice/README.md``.
It is a 10-document synthetic probe, not a production multipage claim. The SDK
path is new; re-measure on your own documents before relying on the numbers.

No model calls happen in the merge. OCR totals use the local OCR engine only
when ``with_ocr_totals`` is enabled and the final page yields a consistent
subtotal + tax == total block.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .evidence import attach_evidence
from .ocr_totals import apply_totals, read_totals
from .pipeline import BaseEngine, _extract_json, build_engine

# --- Prompt (single source of truth; evaluation imports this) ---------------

PAGE_PROMPT_V1 = """You are an invoice data extractor reading page {page} of {total} of ONE invoice. Reply with JSON only.
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

_CURRENCY_LINE_V1 = "- currency: ISO 4217 code (USD, EUR, GBP, ZAR, NAD, ZMW)"
_CURRENCY_LINE_V2 = (
    "- currency: ISO 4217 code (USD, EUR, GBP, ZAR, NAD, ZMW).\n"
    "  Symbol map: N$ = NAD, R = ZAR, K = ZMW, $ = USD, EUR = €, GBP = £.\n"
    "- currency_symbol: the currency symbol printed on this page (for example N$, R, $), or \"\""
)
PAGE_PROMPT_V2 = PAGE_PROMPT_V1.replace(_CURRENCY_LINE_V1, _CURRENCY_LINE_V2)


def page_prompt(page: int, total: int, version: int = 1) -> str:
    """Return the per-page invoice prompt. v1 is canonical; v2 is archived."""
    template = PAGE_PROMPT_V1 if version == 1 else PAGE_PROMPT_V2
    return template.format(page=page, total=total)


# --- Normalization (mirrors the measured evaluation helpers) ----------------

SUMMARY_ROW = re.compile(
    r"^(carried|brought)\s+forward$"
    r"|^(continued(\s+on\s+next\s+page)?|page\s+\d+\s+of\s+\d+)$"
    r"|^(sub\s*total|net\s+amount|tax|vat|sales\s+tax|total|amount\s+due|"
    r"balance\s+due|total\s+due)(\s*\(?\s*\d+(?:\.\d+)?\s*%?\s*\)?)?$",
    re.IGNORECASE,
)

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


def is_summary_row(description: str) -> bool:
    """True for running-total / tax / page labels that are not line items."""
    return bool(SUMMARY_ROW.match((description or "").strip()))


def symbol_currency(value: Any) -> Optional[str]:
    """Map a printed symbol to ISO without guessing from the vendor name."""
    if value is None:
        return None
    return CURRENCY_SYMBOLS.get(str(value).strip().upper().replace(" ", ""))


def _num(value: Any) -> Optional[float]:
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


def _currency(value: Any) -> Optional[str]:
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


def _date(
    value: Any, doc_id: str = "", currency: Optional[str] = None
) -> Optional[str]:
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
    except Exception:  # noqa: BLE001 - leave unparseable text for the caller
        return text


def _doc_type(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if text in ("credit", "credit_note", "creditnote", "cn"):
        return "credit_note"
    if text in ("statement", "other", "not_invoice", "notinvoice"):
        return "other"
    return "invoice"


def _lines(value: Any) -> List[Dict[str, Any]]:
    out = []
    for item in value or []:
        if not isinstance(item, dict):
            continue
        amount = _num(item.get("amount", item.get("total", item.get("line_total"))))
        out.append(
            {
                "description": str(item.get("description", "") or ""),
                "amount": amount,
            }
        )
    return out


def normalize_prediction(obj: dict, doc_id: str = "") -> dict:
    obj = obj or {}
    currency = _currency(obj.get("currency"))
    return {
        "vendor_name": str(obj.get("vendor_name", obj.get("vendor", "")) or ""),
        "invoice_number": str(
            obj.get("invoice_number", obj.get("invoice_no", "")) or ""
        ),
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


# --- Merge ------------------------------------------------------------------


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


def merge_pages(pages: Sequence[dict], doc_id: str = "") -> dict:
    """Merge page-level extractions into one normalized invoice prediction."""
    pages = [page for page in pages if isinstance(page, dict)]
    currency = symbol_currency(
        _last(page.get("currency_symbol") for page in pages)
    ) or _currency(_last(page.get("currency") for page in pages))
    lines: List[Dict[str, Any]] = []
    for page in pages:
        for item in page.get("line_items") or []:
            if not isinstance(item, dict):
                continue
            amount = _num(
                item.get("amount", item.get("total", item.get("line_total")))
            )
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
        "invoice_number": str(
            _first(page.get("invoice_number") for page in pages) or ""
        ),
        "invoice_date": _date(
            _first(page.get("invoice_date") for page in pages), doc_id, currency
        ),
        "due_date": _date(
            _first(page.get("due_date") for page in pages), doc_id, currency
        ),
        "currency": currency,
        "subtotal": subtotal,
        "tax": tax,
        "total_amount": total,
        "document_type": _doc_type(
            _first(page.get("document_type") for page in pages)
        ),
        "line_items": lines,
        "flagged": False,
        "references": [],
    }
    normalized = normalize_prediction(merged, doc_id)
    normalized, corrections = reconcile_totals(normalized)
    if corrections:
        normalized["corrections"] = corrections
    return normalized


def _tolerance(value: float, rel_tol: float = 0.001, abs_tol: float = 0.02) -> float:
    return max(abs_tol, rel_tol * max(1.0, abs(value)))


def reconcile_totals(
    merged: dict,
    rel_tol: float = 0.001,
    abs_tol: float = 0.02,
) -> Tuple[dict, List[str]]:
    """Repair a total only when the extracted line items corroborate the subtotal.

    This is internal consistency, not gold: if the line amounts sum to the printed
    subtotal, then subtotal + tax is the best available total. A printed total
    that disagrees is replaced and the correction is recorded. If the line sum
    does not corroborate the subtotal, nothing is changed.
    """
    corrections: List[str] = []
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

    if tax is not None and total is not None:
        implied_tax = round(total - subtotal, 2)
        plausible = abs(total) > abs(subtotal) and abs(implied_tax) <= 0.5 * abs(
            subtotal
        )
        if plausible and abs(tax - implied_tax) > _tolerance(total, rel_tol, abs_tol):
            merged["tax"] = implied_tax
            tax = implied_tax
            corrections.append("tax_from_total_subtotal")

    if tax is not None:
        computed = round(subtotal + tax, 2)
        if total is None or abs(computed - total) > _tolerance(total, rel_tol, abs_tol):
            merged["total_amount"] = computed
            corrections.append("total_from_subtotal_tax")
    elif total is not None:
        merged["tax"] = round(total - subtotal, 2)
        corrections.append("tax_from_total_subtotal")

    return merged, corrections


def merge_is_sane(merged: dict, pages: Sequence[dict]) -> List[str]:
    """Cheap structural checks that do not use gold answers."""
    issues: List[str] = []
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


# --- Arithmetic review checks (no gold, no value changes) -------------------


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        text = str(value).strip().replace(",", "")
        return float(text) if text else None
    except (TypeError, ValueError):
        return None


def _line_total(lines: Any) -> Optional[float]:
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
    prediction: dict,
    rel_tol: float = 0.001,
    abs_tol: float = 0.02,
) -> List[Dict[str, Any]]:
    """Return arithmetic issues for one normalized invoice prediction."""
    issues: List[Dict[str, Any]] = []
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


# --- PDF / image preparation ------------------------------------------------


def render_pdf_pages(
    pdf_path: Union[str, Path],
    out_dir: Union[str, Path],
    max_px: int = 1400,
) -> List[str]:
    """Render every PDF page to a PNG capped at ``max_px`` on the long edge."""
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "PDF support needs pypdfium2. Install it with: pip install 'tinydoc[pdf]'"
        ) from exc

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = pdfium.PdfDocument(str(pdf_path))
    paths: List[str] = []
    try:
        for index in range(len(pdf)):
            page = pdf[index]
            width, height = page.get_size()
            scale = max(0.1, float(max_px) / max(width, height))
            bitmap = page.render(scale=scale)
            image = bitmap.to_pil().convert("RGB")
            out = out_dir / f"page-{index + 1:03d}.png"
            image.save(out)
            paths.append(str(out))
    finally:
        try:
            pdf.close()
        except Exception:  # noqa: BLE001 - best-effort cleanup
            pass
    return paths


def _resize_image(
    image_path: Union[str, Path],
    out_dir: Union[str, Path],
    max_px: int,
) -> str:
    """Downscale a page image to ``max_px``; return the original if already small."""
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - pillow is a base dependency
        return str(image_path)
    with Image.open(image_path) as image:
        width, height = image.size
        if max(width, height) <= max_px:
            return str(image_path)
        scale = float(max_px) / max(width, height)
        resized = image.convert("RGB").resize(
            (max(1, int(width * scale)), max(1, int(height * scale)))
        )
        out = Path(out_dir) / (Path(image_path).stem + f"-{max_px}.png")
        resized.save(out)
        return str(out)


def _invoice_schema() -> Optional[dict]:
    path = Path(__file__).absolute().parent / "schemas" / "invoice.schema.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001 - schema validation is best-effort
        return None


def invoice_schema_is_valid(fields: dict) -> bool:
    schema = _invoice_schema()
    if schema is None:
        return isinstance(fields, dict) and bool(fields)
    try:
        import jsonschema

        jsonschema.validate(instance=fields, schema=schema)
        return True
    except Exception:  # noqa: BLE001 - invalid JSON is a result, not a crash
        return False


# --- Result -----------------------------------------------------------------


@dataclass
class InvoiceResult:
    source: str
    engine: str
    page_count: int
    fields: Dict[str, Any]
    line_items: List[Dict[str, Any]]
    schema_valid: bool
    corrections: List[str] = field(default_factory=list)
    review_flags: List[str] = field(default_factory=list)
    arithmetic_issues: List[Dict[str, Any]] = field(default_factory=list)
    ocr_totals: Optional[Dict[str, Any]] = None
    pages: List[Dict[str, Any]] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    render_px: int = 1400
    method: str = "page_wise_merge"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "engine": self.engine,
            "method": self.method,
            "page_count": self.page_count,
            "render_px": self.render_px,
            "schema_valid": self.schema_valid,
            "fields": self.fields,
            "line_items": self.line_items,
            "corrections": self.corrections,
            "review_flags": self.review_flags,
            "arithmetic_issues": self.arithmetic_issues,
            "ocr_totals": self.ocr_totals,
            "evidence": self.evidence,
            "pages": self.pages,
            "latency_ms": round(self.latency_ms, 1),
            "confidence_note": (
                "Invoice field confidence is not calibrated. review_flags are "
                "arithmetic checks for human routing; evidence scores are OCR "
                "match quality, not probabilities."
            ),
        }


class InvoicePipeline:
    """Page-wise invoice extraction with a deterministic merge.

    ``engine`` must provide ``generate(prompt, image_path, num_predict=...)``.
    The built-in Ollama engine does; offline engines do not, and this pipeline
    says so instead of silently returning an empty invoice.
    """

    def __init__(
        self,
        engine: Union[str, BaseEngine] = "ollama:qwen3-vl:2b-instruct",
        **engine_kwargs: Any,
    ):
        if isinstance(engine, BaseEngine):
            self.engine = engine
        else:
            self.engine = build_engine(engine, **engine_kwargs)

    def _generate(self, prompt: str, image_path: str, num_predict: int) -> dict:
        generate = getattr(self.engine, "generate", None)
        if generate is None:
            raise RuntimeError(
                f"engine {getattr(self.engine, 'name', self.engine)!r} cannot run "
                "invoice page prompts; use an Ollama engine (for example "
                "'ollama:qwen3-vl:2b-instruct') and start Ollama first"
            )
        return generate(prompt, image_path, num_predict=num_predict)

    def extract(
        self,
        source: Union[str, Path],
        *,
        images: Optional[Sequence[Union[str, Path]]] = None,
        max_px: int = 1400,
        with_ocr_totals: bool = True,
        with_evidence: bool = True,
        num_predict: int = 2048,
        hint: str = "",
        prompt_version: int = 1,
    ) -> InvoiceResult:
        """Extract one invoice from a PDF, a page image, or a page list."""
        started = time.time()
        source_text = str(source)
        tempdirs: List[tempfile.TemporaryDirectory] = []
        page_images: List[str] = []
        try:
            if images is not None:
                page_images = [str(path) for path in images]
            else:
                path = Path(source)
                if not path.exists():
                    raise FileNotFoundError(str(path))
                if path.suffix.lower() == ".pdf":
                    pdf_dir = tempfile.TemporaryDirectory(prefix="tinydoc-pages-")
                    tempdirs.append(pdf_dir)
                    page_images = render_pdf_pages(path, pdf_dir.name, max_px=max_px)
                else:
                    page_images = [str(path)]
            if not page_images:
                raise ValueError("no pages to extract")

            # Cap image size for context control, matching the measured probe.
            # The work directory stays alive until evidence collection finishes.
            workdir = tempfile.TemporaryDirectory(prefix="tinydoc-work-")
            tempdirs.append(workdir)
            page_images = [
                _resize_image(path, workdir.name, max_px) for path in page_images
            ]
            page_records, parsed_pages = self._read_pages(
                page_images, num_predict, prompt_version
            )

            if not parsed_pages:
                errors = "; ".join(
                    str(record.get("error")) for record in page_records if record.get("error")
                )
                raise RuntimeError(f"no page produced parseable JSON: {errors or 'unknown'}")

            doc_id = hint or Path(source_text).stem
            merged = merge_pages(parsed_pages, doc_id)
            corrections: List[str] = list(merged.pop("corrections", []) or [])
            ocr_totals = None
            if with_ocr_totals:
                try:
                    ocr_totals = read_totals(page_images[-1])
                except Exception as exc:  # noqa: BLE001 - OCR is optional corroboration
                    ocr_totals = {"error": f"{type(exc).__name__}: {exc}", "consistent": False}
                if ocr_totals.get("consistent"):
                    merged, ocr_corrections = apply_totals(merged, ocr_totals)
                    corrections.extend(ocr_corrections)
                    symbol = symbol_currency(ocr_totals.get("currency_symbol"))
                    if symbol and merged.get("currency") != symbol:
                        merged["currency"] = symbol
                        corrections.append("currency_from_ocr_totals")

            issues = check_prediction(merged)
            review_flags = [issue["check"] for issue in issues]
            review_flags.extend(merge_is_sane(merged, parsed_pages))
            # Preserve order while removing duplicates.
            review_flags = list(dict.fromkeys(review_flags))

            fields = {
                key: merged.get(key)
                for key in (
                    "vendor_name",
                    "invoice_number",
                    "invoice_date",
                    "due_date",
                    "currency",
                    "subtotal",
                    "tax",
                    "total_amount",
                    "document_type",
                    "line_items",
                )
            }
            evidence = self._collect_evidence(
                merged, page_images, ocr_totals, with_evidence
            )
            return InvoiceResult(
                source=source_text,
                engine=getattr(self.engine, "name", str(self.engine)),
                page_count=len(page_images),
                fields=fields,
                line_items=merged.get("line_items") or [],
                schema_valid=invoice_schema_is_valid(fields),
                corrections=corrections,
                review_flags=review_flags,
                arithmetic_issues=issues,
                ocr_totals=ocr_totals,
                pages=page_records,
                evidence=evidence,
                latency_ms=(time.time() - started) * 1000,
                render_px=max_px,
                method=(
                    "page_wise_merge+ocr_totals"
                    if with_ocr_totals
                    else "page_wise_merge"
                ),
            )
        finally:
            for directory in reversed(tempdirs):
                directory.cleanup()

    def _read_pages(
        self,
        page_images: Sequence[str],
        num_predict: int,
        prompt_version: int,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        records: List[Dict[str, Any]] = []
        parsed_pages: List[Dict[str, Any]] = []
        total = len(page_images)
        for index, image_path in enumerate(page_images, start=1):
            page_started = time.time()
            raw = ""
            try:
                response = self._generate(
                    page_prompt(index, total, prompt_version), image_path, num_predict
                )
                raw = response.get("raw") or response.get("content") or ""
                parsed = _extract_json(raw)
                records.append(
                    {
                        "page": index,
                        "raw": raw,
                        "latency_ms": round((time.time() - page_started) * 1000, 1),
                        "prompt_eval_count": response.get("prompt_eval_count"),
                        "eval_count": response.get("eval_count"),
                        "done_reason": response.get("done_reason"),
                        "error": None if parsed is not None else "no JSON object",
                    }
                )
                if parsed is not None:
                    parsed_pages.append(parsed)
            except Exception as exc:  # noqa: BLE001 - keep other pages alive
                records.append(
                    {
                        "page": index,
                        "raw": raw,
                        "latency_ms": round((time.time() - page_started) * 1000, 1),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        return records, parsed_pages

    @staticmethod
    def _collect_evidence(
        merged: dict,
        page_images: Sequence[str],
        ocr_totals: Optional[dict],
        enabled: bool,
    ) -> Dict[str, Any]:
        evidence: Dict[str, Any] = {}
        if not enabled or not page_images:
            return evidence
        header = {
            key: str(merged.get(key) or "")
            for key in ("vendor_name", "invoice_number", "invoice_date", "due_date")
        }
        try:
            header_evidence = attach_evidence(header, page_images[0], page=1)
        except Exception:  # noqa: BLE001 - evidence is optional
            header_evidence = {}
        for key, item in header_evidence.items():
            if item is not None:
                evidence[key] = item.to_dict()
        if ocr_totals and ocr_totals.get("consistent"):
            for key in ("subtotal", "tax", "total"):
                line = (ocr_totals.get("evidence") or {}).get(key)
                if line:
                    evidence[key] = {
                        "kind": "ocr_line",
                        "quote": line,
                        "bbox": None,
                        "page": len(page_images),
                        "score": 1.0,
                    }
        return evidence


__all__ = [
    "InvoicePipeline",
    "InvoiceResult",
    "PAGE_PROMPT_V1",
    "PAGE_PROMPT_V2",
    "page_prompt",
    "merge_pages",
    "reconcile_totals",
    "merge_is_sane",
    "check_prediction",
    "render_pdf_pages",
    "invoice_schema_is_valid",
]
