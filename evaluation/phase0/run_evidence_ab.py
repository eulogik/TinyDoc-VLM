"""Evidence-layer A/B: Tesseract vs RapidOCR (PP-OCR) word boxes.

Uses the identical committed field values from the shipped evidence run, so the
only variable is where the word boxes come from. For each field value we attach
evidence via locate_field_evidence with words from each engine and measure:

  coverage        fraction of non-empty fields that get an evidence box
  mean_score      evidence match score (0..1)
  gold_in_quote   fraction where the evidence quote actually contains the GOLD
                  value (normalized mutual containment) — the product question:
                  "can a human see the provenance of this answer?"

Adoption rule (persisted): switch the evidence engine only if gold_in_quote
improves and coverage does not fall by more than 5 points.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path("/Users/eulogikdeveloper/Documents/TinyDoc-VLM")
sys.path.insert(0, str(ROOT / "evaluation/phase0"))
sys.path.insert(0, str(ROOT / "sdk"))

from tinydoc.evidence import locate_field_evidence  # noqa: E402

PREDS = ROOT / "evaluation/phase0/results/preds_pipeline_ollama_qwen2.5vl_3b_evidence.jsonl"
FIELDS = ("company", "date", "address", "total")


def norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def rapid_words(engine, image_path: Path) -> list[dict]:
    result, _ = engine(str(image_path))
    if not result:
        return []
    words = []
    for box, text, _score in result:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        words.append({
            "text": text,
            "left": int(min(xs)),
            "top": int(min(ys)),
            "width": int(max(xs) - min(xs)),
            "height": int(max(ys) - min(ys)),
        })
    return words


def quote_covers_gold(quote: str, gold: str) -> bool:
    q, g = norm(quote), norm(gold)
    if not q or not g:
        return False
    return g in q or q in g


def evaluate(word_sets: dict[str, list[dict]], row: dict) -> dict:
    """word_sets: engine name -> words for this doc. Returns per-engine tallies."""
    out = {}
    for engine_name, words in word_sets.items():
        cov = hits = n = 0
        score_sum = 0.0
        for f in FIELDS:
            pred_val = str(row["pred"].get(f) or "").strip()
            if not pred_val:
                continue
            n += 1
            ev = locate_field_evidence(pred_val, row["image_path"], words=words)
            if ev is not None:
                cov += 1
                score_sum += ev.score
                if quote_covers_gold(ev.quote, str(row["gold"].get(f) or "")):
                    hits += 1
        out[engine_name] = {
            "fields": n,
            "covered": cov,
            "gold_in_quote": hits,
            "score_sum": score_sum,
        }
    return out


def main() -> int:
    from rapidocr_onnxruntime import RapidOCR

    rapid = RapidOCR()
    rows = [json.loads(l) for l in PREDS.open()]
    print(f"rows={len(rows)}", flush=True)

    engines: dict[str, dict] = {}
    n_docs = 0
    for i, row in enumerate(rows):
        img = Path(row["image_path"])
        if not img.is_absolute():
            img = ROOT / img
        if not img.exists() or not row.get("pred"):
            continue
        n_docs += 1

        from tinydoc.evidence import _ocr_words  # pytesseract, per-doc

        tesseract = _ocr_words(str(img))
        word_sets = {"tesseract": tesseract, "rapidocr": rapid_words(rapid, img)}
        per_doc = evaluate(word_sets, row)
        for engine_name, tally in per_doc.items():
            acc = engines.setdefault(
                engine_name,
                {"fields": 0, "covered": 0, "gold_in_quote": 0, "score_sum": 0.0},
            )
            for k in ("fields", "covered", "gold_in_quote", "score_sum"):
                acc[k] += tally[k]
        if (i + 1) % 25 == 0:
            print(f"[{i + 1}/{len(rows)}]", flush=True)

    summary = {}
    for engine_name, acc in engines.items():
        summary[engine_name] = {
            "docs": n_docs,
            "coverage": round(acc["covered"] / acc["fields"], 4) if acc["fields"] else 0.0,
            "mean_score": round(acc["score_sum"] / acc["covered"], 4) if acc["covered"] else 0.0,
            "gold_in_quote_rate": round(acc["gold_in_quote"] / acc["fields"], 4) if acc["fields"] else 0.0,
        }

    t, r = summary["tesseract"], summary["rapidocr"]
    switch = (
        r["gold_in_quote_rate"] > t["gold_in_quote_rate"]
        and r["coverage"] >= t["coverage"] - 0.05
    )
    decision = (
        "adopt_rapidocr" if switch else "keep_tesseract"
    )
    out = {
        "summary": summary,
        "decision": decision,
        "rule": "adopt rapidocr only if gold_in_quote improves and coverage drop <= 5 points",
    }
    dest = ROOT / "evaluation/phase0/results/evidence_ab.json"
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
