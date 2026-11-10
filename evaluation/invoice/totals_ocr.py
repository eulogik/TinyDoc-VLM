#!/usr/bin/env python3
"""Read printed invoice totals with OCR as a deterministic cross-check.

The vision model occasionally drops leading digits on amounts with space
thousands separators (``R25 083.76`` read as ``2 508.36``). The printed totals
are large, labelled, and arithmetic-checkable, so OCR is a good independent
reader for exactly three values: subtotal, tax, and total.

This module never sees gold answers. It returns a candidate total block only
when OCR can read all three labels and the three values are internally
consistent (``subtotal + tax == total`` within tolerance, tax rate plausible).
Applying it is a separate step and always records a correction.

Tesseract is used when available; RapidOCR is a fallback. No model calls.
"""

from __future__ import annotations

import argparse
import json
import re
from typing import Any

SUBTOTAL = re.compile(
    r"\b(sub\s*total|net\s+amount|net\s+total|netto|zwischensumme)\b",
    re.IGNORECASE,
)
TAX = re.compile(
    r"\b(vat|sales\s+tax|tax|gst|mwst|ust|umsatzsteuer|iva|tva|btw)\b",
    re.IGNORECASE,
)
TOTAL = re.compile(
    r"\b(total|amount\s+due|balance\s+due|total\s+due|gesamt|summe|brutto)\b",
    re.IGNORECASE,
)
AMOUNT = re.compile(r"-?\d[\d\s.,]*\d|\d")
# OCR often renders a leading 1 as I/l/| right after a currency symbol.
CONFUSABLE_ONE = re.compile(r"(?<=[R$€£N])[Il|](?=\d)|(?<=\d)[Il|](?=\d)")
SYMBOL = re.compile(r"(NAD\$|N\$|US\$|ZK|ZMW|ZAR|R|\$|€|£|K)\s*(?=-?\d)")
CURRENCY_CODE = re.compile(r"\b(NAD|ZAR|USD|EUR|GBP|ZMW)\b", re.IGNORECASE)


def parse_amount(text: str) -> float | None:
    """Parse a printed amount with space/comma/dot separators.

    Handles ``25 083.76``, ``25,083.76``, ``25.083,76``, ``R 1 234``.
    Returns None when the text does not contain a usable number.
    """
    if text is None:
        return None
    raw = str(text).strip()
    negative = raw.startswith("(") and raw.endswith(")")
    cleaned = re.sub(r"[^0-9,.\-]", "", raw)
    if not cleaned or cleaned in {"-", ".", ","}:
        return None
    negative = negative or cleaned.startswith("-")
    cleaned = cleaned.lstrip("-")
    if "," in cleaned and "." in cleaned:
        if cleaned.rfind(",") > cleaned.rfind("."):
            cleaned = cleaned.replace(".", "").replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")
    elif "," in cleaned:
        if re.search(r",\d{1,2}$", cleaned):
            cleaned = cleaned.replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")
    elif "." in cleaned and re.search(r"\.\d{3}$", cleaned):
        # A trailing group of three digits is a thousands separator.
        cleaned = cleaned.replace(".", "")
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return -value if negative else value


def _amount_after(text: str, match: re.Match[str]) -> float | None:
    tail = text[match.end() :]
    for found in AMOUNT.finditer(tail):
        # Ignore a tax rate printed before the amount ("VAT 15% R3 762.56").
        if tail[found.end() :].lstrip().startswith("%"):
            continue
        return parse_amount(found.group(0))
    return None


def _is_table_header(line: str) -> bool:
    lower = line.lower()
    return ("description" in lower or "qty" in lower) and (
        "price" in lower or "total" in lower or "amount" in lower
    )


def _candidates(
    lines: list[str],
    pattern: re.Pattern[str],
    exclude: re.Pattern[str] | None = None,
) -> list[tuple[int, float, str, str]]:
    found: list[tuple[int, float, str, str]] = []
    for index, raw in enumerate(lines):
        line = CONFUSABLE_ONE.sub("1", raw.strip())
        if not line or _is_table_header(line):
            continue
        if exclude is not None and exclude.search(line):
            continue
        for match in pattern.finditer(line):
            value = _amount_after(line, match)
            if value is not None:
                found.append((index, value, line, _symbol(line)))
    return found


def _symbol(line: str) -> str:
    match = SYMBOL.search(line)
    if match:
        return match.group(1)
    code = CURRENCY_CODE.search(line)
    return code.group(1).upper() if code else ""


def parse_totals_lines(lines: list[str]) -> dict[str, Any]:
    """Find a consistent subtotal/tax/total block, using arithmetic to choose.

    Invoice tables repeat the words "total", "tax", and even "subtotal" in
    headers and line descriptions. Candidates are therefore combined and kept
    only when ``subtotal + tax == total`` and the implied tax rate is plausible.
    """
    subtotals = _candidates(lines, SUBTOTAL)
    taxes = _candidates(
        lines, TAX, exclude=re.compile(r"\binvoice\b|\binv\b", re.IGNORECASE)
    )
    totals = _candidates(
        lines, TOTAL, exclude=re.compile(r"\bsub\s*total\b", re.IGNORECASE)
    )
    best: tuple[tuple[int, int, int], dict[str, Any]] | None = None
    for s_index, subtotal, s_line, s_symbol in subtotals:
        for t_index, tax, t_line, t_symbol in taxes:
            for g_index, total, g_line, g_symbol in totals:
                candidate = {"subtotal": subtotal, "tax": tax, "total": total}
                if not totals_are_consistent(candidate):
                    continue
                # Prefer a block printed in reading order and lower on the page.
                score = (min(s_index, t_index, g_index), max(s_index, t_index, g_index))
                if best is None or score > best[0]:
                    best = (
                        score,
                        {
                            "subtotal": subtotal,
                            "tax": tax,
                            "total": total,
                            "currency_symbol": s_symbol or t_symbol or g_symbol,
                            "evidence": {
                                "subtotal": s_line,
                                "tax": t_line,
                                "total": g_line,
                            },
                        },
                    )
    if best is None:
        return {
            "subtotal": None,
            "tax": None,
            "total": None,
            "currency_symbol": "",
            "evidence": {},
        }
    return best[1]


def _tolerance(value: float, rel_tol: float = 0.001, abs_tol: float = 0.02) -> float:
    return max(abs_tol, rel_tol * max(1.0, abs(value)))


def totals_are_consistent(
    totals: dict[str, Any],
    rel_tol: float = 0.0,
    abs_tol: float = 0.05,
) -> bool:
    subtotal = totals.get("subtotal")
    tax = totals.get("tax")
    total = totals.get("total")
    if subtotal is None or tax is None or total is None:
        return False
    if total <= 0 or subtotal <= 0:
        return False
    if abs((subtotal + tax) - total) > _tolerance(total, rel_tol, abs_tol):
        return False
    rate = tax / subtotal
    return 0.0 <= rate <= 0.35


def ocr_lines(image_path: str) -> list[str]:
    """OCR a page into text lines using tesseract, then RapidOCR."""
    try:
        import pytesseract
        from PIL import Image

        with Image.open(image_path) as image:
            text = pytesseract.image_to_string(image, config="--psm 6")
        return [line for line in text.splitlines() if line.strip()]
    except Exception:  # noqa: BLE001, S110 - fall through to the optional engine
        pass
    try:
        from rapidocr_onnxruntime import RapidOCR

        result, _ = RapidOCR()(image_path)
        return [str(item[1]) for item in (result or [])]
    except Exception:  # noqa: BLE001 - no OCR engine available
        return []


def read_totals(image_path: str) -> dict[str, Any]:
    """OCR one page and return a consistent printed-total block, or nulls."""
    lines = ocr_lines(image_path)
    totals = parse_totals_lines(lines)
    totals["consistent"] = totals_are_consistent(totals)
    if not totals["consistent"]:
        totals["subtotal"] = None
        totals["tax"] = None
        totals["total"] = None
    return totals


def apply_totals(
    prediction: dict[str, Any],
    totals: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Replace printed totals when OCR is consistent and disagrees with the model."""
    corrections: list[str] = []
    if not totals.get("consistent"):
        return prediction, corrections
    updated = dict(prediction)
    changed = []
    for name, key in (("subtotal", "subtotal"), ("tax", "tax"), ("total", "total_amount")):
        value = totals.get(name)
        current = updated.get(key)
        if value is None:
            continue
        try:
            current_number = float(current) if current is not None else None
        except (TypeError, ValueError):
            current_number = None
        # Printed money is compared to the cent: a 29-cent difference is a
        # model misread, not rounding noise.
        if current_number is None or abs(current_number - value) > 0.005:
            updated[key] = value
            changed.append(name)
    if changed:
        corrections.append("totals_from_ocr:" + "+".join(changed))
    return updated, corrections


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", required=True)
    args = ap.parse_args()
    print(json.dumps(read_totals(args.image), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
