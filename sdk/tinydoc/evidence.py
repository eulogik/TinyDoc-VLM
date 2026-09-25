"""OCR-span evidence: locate field values in page text (no VLM boxes).

Evidence kind is always labeled ``ocr_span`` — text quote + word bbox from
Tesseract. VLM predicted boxes are Phase 2+ only when supervised GT exists.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Evidence:
    kind: str  # "ocr_span"
    quote: str
    bbox: Optional[List[float]]  # x0,y0,x1,y1 in pixels (absolute)
    page: int
    score: float  # 0..1 match quality of quote vs field value

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[\$£€]", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


_RAPIDOCR_CACHE: Dict[str, Any] = {"engine": None, "failed": False}


def _rapidocr_engine():
    """Cached RapidOCR singleton; None if unavailable (loaded once per process)."""
    if _RAPIDOCR_CACHE["failed"]:
        return None
    if _RAPIDOCR_CACHE["engine"] is None:
        try:
            from rapidocr_onnxruntime import RapidOCR

            _RAPIDOCR_CACHE["engine"] = RapidOCR()
        except Exception:
            _RAPIDOCR_CACHE["failed"] = True
            return None
    return _RAPIDOCR_CACHE["engine"]


def _rapidocr_words(image_path: str) -> List[Dict[str, Any]]:
    """RapidOCR/PP-OCR word boxes; [] if unavailable or no text."""
    engine = _rapidocr_engine()
    if engine is None:
        return []
    try:
        result, _ = engine(image_path)
    except Exception:
        return []
    if not result:
        return []
    words: List[Dict[str, Any]] = []
    for box, text, _score in result:
        txt = (text or "").strip()
        if not txt:
            continue
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        left, top = min(xs), min(ys)
        words.append(
            {
                "text": txt,
                "left": left,
                "top": top,
                "width": max(xs) - left,
                "height": max(ys) - top,
            }
        )
    return words


def _tesseract_words(image_path: str) -> List[Dict[str, Any]]:
    """Tesseract TSV word boxes; empty list if OCR unavailable."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return []
    try:
        img = Image.open(image_path)
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    except Exception:
        return []
    words = []
    n = len(data.get("text", []))
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue
        try:
            conf = float(data["conf"][i])
        except (ValueError, TypeError):
            conf = -1.0
        if conf < 0:
            continue
        words.append(
            {
                "text": txt,
                "left": float(data["left"][i]),
                "top": float(data["top"][i]),
                "width": float(data["width"][i]),
                "height": float(data["height"][i]),
            }
        )
    return words


def _ocr_words(image_path: str, engine: str = "auto") -> List[Dict[str, Any]]:
    """Word boxes for evidence anchoring.

    engine="rapidocr" (default choice measured in evidence_ab.json: coverage
    +13 pts, gold-in-quote +12 pts vs tesseract) — PP-OCR via onnxruntime.
    engine="tesseract" — original engine, kept as fallback/dependency-light path.
    engine="auto" — rapidocr when installed, else tesseract.
    """
    if engine == "tesseract":
        return _tesseract_words(image_path)
    words = _rapidocr_words(image_path)
    if words:
        return words
    if engine == "rapidocr":
        return []
    return _tesseract_words(image_path)


def _union_bbox(words: List[Dict[str, Any]]) -> Optional[List[float]]:
    if not words:
        return None
    x0 = min(w["left"] for w in words)
    y0 = min(w["top"] for w in words)
    x1 = max(w["left"] + w["width"] for w in words)
    y1 = max(w["top"] + w["height"] for w in words)
    return [x0, y0, x1, y1]


def _token_match_score(pred: str, gold_tokens: List[str]) -> float:
    """Fraction of gold tokens present in pred (order-insensitive, normalized)."""
    if not gold_tokens:
        return 0.0
    p = set(_norm(pred).split())
    g = set(gold_tokens)
    if not g:
        return 0.0
    return len(p & g) / len(g)


def locate_field_evidence(
    field_value: str,
    image_path: str,
    words: Optional[List[Dict[str, Any]]] = None,
    page: int = 1,
    window: int = 6,
    ocr_engine: str = "auto",
) -> Optional[Evidence]:
    """Best contiguous word window approximating ``field_value``."""
    if not field_value or not str(field_value).strip():
        return None
    if words is None:
        words = _ocr_words(image_path, engine=ocr_engine)
    if not words:
        return None

    gold_toks = _norm(str(field_value)).split()
    if not gold_toks:
        return None

    n = len(words)
    best_score = 0.0
    best_slice: List[Dict[str, Any]] = []
    # try windows sized to gold token count ±2, up to `window`
    sizes = sorted({max(1, len(gold_toks) + d) for d in range(-2, 3)} | {1, 3, 5, window})
    for size in sizes:
        if size > n:
            continue
        for i in range(0, n - size + 1):
            chunk = words[i : i + size]
            chunk_text = " ".join(w["text"] for w in chunk)
            sc = _token_match_score(chunk_text, gold_toks)
            # prefer tighter windows slightly
            sc = sc * (1.0 - 0.01 * max(0, size - len(gold_toks)))
            if sc > best_score:
                best_score = sc
                best_slice = chunk

    if not best_slice or best_score < 0.34:
        # fallback: single best word by token overlap
        for w in words:
            sc = _token_match_score(w["text"], gold_toks)
            if sc > best_score:
                best_score = sc
                best_slice = [w]
        if best_score < 0.34:
            return None

    quote = " ".join(w["text"] for w in best_slice)
    return Evidence(
        kind="ocr_span",
        quote=quote,
        bbox=_union_bbox(best_slice),
        page=page,
        score=round(float(min(best_score, 1.0)), 4),
    )


def attach_evidence(
    fields: Dict[str, Any],
    image_path: str,
    page: int = 1,
    ocr_engine: str = "auto",
) -> Dict[str, Optional[Evidence]]:
    words = _ocr_words(image_path, engine=ocr_engine)
    out: Dict[str, Optional[Evidence]] = {}
    for k, v in (fields or {}).items():
        if k == "evidence":
            continue
        out[k] = locate_field_evidence(str(v or ""), image_path, words=words, page=page)
    return out
