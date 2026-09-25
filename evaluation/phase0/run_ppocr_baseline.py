"""Competitor baseline: PP-OCR (RapidOCR ONNX) + rule-based field extraction.

The fair local competitor to ReceiptPipeline is "best free OCR + heuristics" —
no VLM, no API key, no GPU. This runs on the same 100-doc SROIE eval and scores
with the same score_example / field_match harness as every other engine, so the
comparison is paired and exact.

Heuristics are deliberately standard (no per-doc tuning, no gold leakage):
  company : first line containing business tokens (SDN BHD / BHD / LLC / INC ...)
            else first plausible name line
  date    : first date-regex match, preferring a line labelled "date"
  total   : amount on the last line containing "total" (grand/amount due), else None
  address : lines with street/postal cues (JALAN, LOT, NO., ROAD, 5-digit postcode)
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/Users/eulogikdeveloper/Documents/TinyDoc-VLM")
sys.path.insert(0, str(ROOT / "evaluation/phase0"))

from metrics import score_example, macro_prf  # noqa: E402

BUSINESS_TOKENS = (
    "sdn bhd", "bhd", "sdn", "llc", "ltd", "inc", "corporation", "company",
    "trading", "restaurant", "restoran", "market", "mart", "store", "shop",
    "enterprise", "venture", "holdings", "resources", "supply", "bakery",
    "pharmacy", "clinic", "hardware", "enterprise",
)
ADDRESS_CUES = (
    "jalan", "jaln", "lorong", "road", "rd", "st", "street", "avenue", "ave",
    "lot", "no", "kg", "kampung", "taman", "menara", "desa", "perusahaan",
    "perindustrian", "kawasan", "bangunan", "ground floor", "floor", "pt ",
)
DATE_RE = re.compile(r"\b(\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4})\b")
MONEY_RE = re.compile(r"\d{1,3}(?:,\d{3})*\.\d{2}")
POSTCODE_RE = re.compile(r"\b\d{5}\b")


def build_lines(words: list[dict]) -> list[str]:
    """Cluster word boxes into reading-order lines (rows by y, words by x)."""
    if not words:
        return []
    rows: dict[int, list[dict]] = defaultdict(list)
    for w in words:
        rows[round(w["top"] / 12)].append(w)  # ~12px row tolerance
    lines = []
    for _, row in sorted(rows.items()):
        row.sort(key=lambda w: w["left"])
        lines.append(" ".join(w["text"] for w in row).strip())
    return [ln for ln in lines if ln]


def ocr_lines(engine, image_path: Path) -> list[str]:
    result, _ = engine(str(image_path))
    if not result:
        return []
    words = []
    for box, text, _score in result:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        words.append({
            "text": text,
            "left": min(xs),
            "top": min(ys),
            "right": max(xs),
            "bottom": max(ys),
        })
    return build_lines(words)


def extract_company(lines: list[str]) -> str:
    lowered = [(ln, ln.lower()) for ln in lines]
    for ln, low in lowered:
        if any(tok in low for tok in BUSINESS_TOKENS) and not DATE_RE.search(ln):
            return ln
    for ln, low in lowered:
        if len(ln) >= 4 and not DATE_RE.search(ln) and not POSTCODE_RE.search(ln):
            return ln
    return ""


def extract_date(lines: list[str]) -> str:
    for ln in lines:
        if "date" in ln.lower():
            m = DATE_RE.search(ln)
            if m:
                return m.group(1)
    for ln in lines:
        m = DATE_RE.search(ln)
        if m:
            return m.group(1)
    return ""


def extract_total(lines: list[str]) -> str:
    for ln in reversed(lines):
        low = ln.lower()
        if "total" in low and "qty" not in low and "supplies" not in low:
            amounts = MONEY_RE.findall(ln)
            if amounts:
                return amounts[-1]
    amounts = MONEY_RE.findall(" ".join(lines))
    return amounts[-1] if amounts else ""


def extract_address(lines: list[str]) -> str:
    parts = []
    for ln in lines:
        low = ln.lower()
        if any(cue in low for cue in ADDRESS_CUES) or POSTCODE_RE.search(ln):
            if "total" in low or DATE_RE.search(ln):
                continue
            parts.append(ln)
    return ", ".join(parts[:4])


def main() -> int:
    from rapidocr_onnxruntime import RapidOCR

    engine = RapidOCR()
    items = json.loads((ROOT / "evaluation/phase0/results/sroie_eval.json").read_text())

    preds = []
    for i, item in enumerate(items):
        img = Path(item["image_path"])
        if not img.is_absolute():
            img = ROOT / img
        lines = ocr_lines(engine, img) if img.exists() else []
        pred = {
            "company": extract_company(lines),
            "date": extract_date(lines),
            "address": extract_address(lines),
            "total": extract_total(lines),
        }
        preds.append((pred, {f: item["gold"].get(f, "") for f in pred}))
        if (i + 1) % 25 == 0:
            print(f"[{i + 1}/{len(items)}]", flush=True)

    scores = [score_example(p, g) for p, g in preds]
    agg = macro_prf(scores)
    per_field = {
        f: sum(1 for s in scores if s["fields"][f]["hit"]) / len(scores)
        for f in ("company", "date", "address", "total")
    }

    out = {
        "engine": "rapidocr_ppocr + heuristics (no VLM, no API)",
        "n": len(scores),
        "field_f1": agg["field_f1"],
        "per_field_hit_rate": per_field,
        "scorer": "identical score_example/macro_prf harness as all other engines",
    }
    dest = ROOT / "evaluation/phase0/results/scores_ppocr_heuristics.json"
    dest.write_text(json.dumps(out, indent=2))
    with (ROOT / "evaluation/phase0/results/preds_ppocr_heuristics.jsonl").open("w") as fh:
        for (pred, gold), s in zip(preds, scores):
            fh.write(json.dumps({"pred": pred, "gold": gold, "score": s}) + "\n")

    print(json.dumps(out, indent=2))
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
