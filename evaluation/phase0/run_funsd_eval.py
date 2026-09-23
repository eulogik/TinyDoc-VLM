#!/usr/bin/env python3
"""Second-vertical eval: FUNSD forms via Receipt-style pipeline (measured).

FUNSD = form key-information extraction. Fields we measure:
  - header  : document title / HEADER spans joined
  - questions+answers as a flat "kv" string for optional diagnostic
Primary metric: header field F1 (normalize_text) + schema-ish presence rates.

This is intentionally a *different* vertical from SROIE receipts to test
generalization of the free engine (no fine-tuning).

Usage:
  python evaluation/phase0/run_funsd_eval.py --engine ollama --limit 50
  python evaluation/phase0/run_funsd_eval.py --engine ollama --limit 50 --with-evidence

Writes:
  results/preds_funsd_{engine}{suffix}.jsonl
  results/scores_funsd_{engine}{suffix}.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).absolute().parent))
from metrics import anls, macro_prf, normalize_text  # noqa: E402

ROOT = Path(__file__).absolute().parent.parent.parent
SDK = ROOT / "sdk"
if str(SDK) not in sys.path:
    sys.path.insert(0, str(SDK))

from tinydoc.pipeline import ReceiptPipeline, sanitize_fields  # noqa: E402

RESULTS = Path(__file__).absolute().parent / "results"
FUNSD_JSON = ROOT / "evaluation" / "data" / "funsd" / "funsd.json"
FUNSD_ROOT = ROOT / "evaluation" / "data"

# FUNSD labels → coarse form fields we score
HEADER_LABELS = {"B-HEADER", "I-HEADER"}
QUESTION_LABELS = {"B-QUESTION", "I-QUESTION"}
ANSWER_LABELS = {"B-ANSWER", "I-ANSWER"}


def build_funsd_gold(item: Dict[str, Any]) -> Dict[str, str]:
    """Aggregate word-level NER tags into header / questions / answers strings."""
    words = item.get("words") or []
    labels = item.get("labels") or []
    headers: List[str] = []
    questions: List[str] = []
    answers: List[str] = []
    n = min(len(words), len(labels))
    cur_h: List[str] = []
    cur_q: List[str] = []
    cur_a: List[str] = []

    def flush():
        nonlocal cur_h, cur_q, cur_a
        if cur_h:
            headers.append(" ".join(cur_h))
            cur_h = []
        if cur_q:
            questions.append(" ".join(cur_q))
            cur_q = []
        if cur_a:
            answers.append(" ".join(cur_a))
            cur_a = []

    for i in range(n):
        w, lab = str(words[i]), str(labels[i])
        if lab in HEADER_LABELS:
            if cur_q or cur_a:
                flush()
            cur_h.append(w)
        elif lab in QUESTION_LABELS:
            if cur_h or cur_a:
                flush()
            cur_q.append(w)
        elif lab in ANSWER_LABELS:
            if cur_h or cur_q:
                # B-ANSWER after question without flush of q — keep q, start a
                if cur_q and lab.startswith("B-"):
                    questions.append(" ".join(cur_q))
                    cur_q = []
                if cur_h:
                    flush()
            cur_a.append(w)
        else:
            flush()
    flush()

    header = ", ".join(headers[:3]) if headers else (headers[0] if headers else "")
    # first question often the form title echo; keep all joined for presence
    q_str = " | ".join(questions[:20])
    a_str = " | ".join(answers[:20])
    return {
        "company": header or (questions[0] if questions else ""),
        "date": answers[0] if answers else "",
        "address": q_str,
        "total": a_str,
    }


def prompt_for_funsd() -> str:
    # Reuse receipt-shaped keys so schema_validate still runs — mapped above.
    return (
        "Extract form fields. Reply with JSON only, exactly four keys:\n"
        '{"company":"...","date":"...","address":"...","total":"..."}\n'
        "Rules:\n"
        "- company: form title / header line(s).\n"
        "- date: any date value on the form.\n"
        "- address: main questions/labels joined with | (up to 10).\n"
        "- total: main answer values joined with | (up to 10).\n"
        "Transcribe as printed. JSON only, no markdown."
    )


def score_pair(pred: Dict[str, str], gold: Dict[str, str]) -> Dict[str, Any]:
    """Micro field match using normalize_text containment for all text fields."""
    per = {}
    hits = 0
    for f in ("company", "date", "address", "total"):
        p = str(pred.get(f, "") or "")
        g = str(gold.get(f, "") or "")
        pt, gt = normalize_text(p), normalize_text(g)
        if not gt:
            hit = not pt
        elif not pt:
            hit = False
        else:
            hit = pt == gt or gt in pt or pt in gt
            # for multi-part address/answer lists: token F1 soft path still binary via containment
            if not hit and f in ("address", "total"):
                pt_t, gt_t = set(pt.split()), set(gt.split())
                if pt_t and gt_t:
                    inter = len(pt_t & gt_t)
                    hit = inter / len(gt_t) >= 0.5
        per[f] = {"hit": hit, "pred": p[:500], "gold": g[:500], "anls": anls(p, g)}
        if hit:
            hits += 1
    return {
        "fields": per,
        "precision": hits / 4,
        "recall": hits / 4,
        "f1": hits / 4,
        "anls_mean": sum(v["anls"] for v in per.values()) / 4,
        "schema_valid": all(
            isinstance(pred.get(k), str) and pred.get(k)
            for k in ("company", "date", "address", "total")
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", default="ollama")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--with-evidence", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not FUNSD_JSON.exists():
        print(f"missing {FUNSD_JSON}", file=sys.stderr)
        raise SystemExit(2)
    items = json.loads(FUNSD_JSON.read_text())
    # deterministic holdout-style sample
    import random

    rng = random.Random(args.seed)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    idx = idx[: args.limit]
    subset = [items[i] for i in sorted(idx)]

    kwargs = {}
    if args.model:
        kwargs["model"] = args.model
    # Override prompt via monkeypatching OllamaEngine.extract_fields prompt?
    # Simpler: build pipeline then wrap engine.
    pipe = ReceiptPipeline(args.engine, **kwargs)

    # Patch prompt for FUNSD by wrapping extract_fields
    orig = pipe.engine.extract_fields

    def extract_funsd(image_path: str) -> Dict[str, str]:
        import base64
        import urllib.request

        eng = pipe.engine
        # unwrap routed if needed — only patch OllamaEngine path
        target = eng
        if hasattr(eng, "primary"):
            target = eng.primary
        if type(target).__name__ != "OllamaEngine":
            return orig(image_path)
        b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
        prompt = prompt_for_funsd()
        payload = {
            "model": target.model,
            "messages": [{"role": "user", "content": prompt, "images": [b64]}],
            "stream": False,
            "options": {"temperature": 0, "num_predict": 512},
        }
        req = urllib.request.Request(
            f"{target.host}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            body = json.loads(resp.read().decode())
        raw = (body.get("message") or {}).get("content", "") or ""
        from tinydoc.pipeline import _extract_json, FIELDS as PF

        obj = _extract_json(raw) or {}
        return {k: str(obj.get(k, "")) for k in PF}

    if hasattr(pipe.engine, "extract_fields"):
        pipe.engine.extract_fields = extract_funsd  # type: ignore
        if hasattr(pipe.engine, "primary"):
            pipe.engine.primary.extract_fields = extract_funsd  # type: ignore

    preds: List[Dict] = []
    scores: List[Dict] = []
    t0 = time.time()
    for i, item in enumerate(subset):
        img_path = str(FUNSD_ROOT / item["image"])
        gold = build_funsd_gold(item)
        try:
            doc = pipe.extract(img_path, with_evidence=args.with_evidence)
            pred = sanitize_fields(doc.fields)
            lat = doc.latency_ms
            schema_ok = doc.schema_valid
            conf = doc.confidence
            error = None
        except Exception as e:
            traceback.print_exc()
            pred = {k: "" for k in ("company", "date", "address", "total")}
            lat = 0.0
            schema_ok = False
            conf = 0.0
            error = str(e)
        sc = score_pair(pred, gold)
        rec = {
            "id": Path(item["image"]).stem,
            "image_path": img_path,
            "pred": pred,
            "gold": gold,
            "latency_ms": lat,
            "schema_valid": schema_ok,
            "confidence": conf,
            "error": error,
            "score": sc,
        }
        preds.append(rec)
        scores.append(sc)
        if (i + 1) % 5 == 0 or i == 0 or (i + 1) == len(subset):
            agg = macro_prf(scores)
            print(
                f"  [{i+1}/{len(subset)}] f1={agg['field_f1']:.3f} "
                f"anls={agg['anls_mean']:.3f} schema={agg['schema_valid_rate']:.3f}"
            )

    agg = macro_prf(scores)
    agg["engine"] = f"funsd_pipeline:{pipe.engine.name}"
    agg["vertical"] = "funsd_forms"
    agg["n_examples"] = len(scores)
    agg["avg_latency_ms"] = sum(p["latency_ms"] for p in preds) / max(len(preds), 1)
    agg["with_evidence"] = bool(args.with_evidence)
    agg["schema_valid_rate"] = sum(1 for p in preds if p["schema_valid"]) / max(
        len(preds), 1
    )
    agg["mean_confidence"] = sum(p["confidence"] for p in preds) / max(len(preds), 1)
    per = {}
    for f in ("company", "date", "address", "total"):
        hits = sum(1 for s in scores if s["fields"][f]["hit"])
        anls_v = sum(s["fields"][f]["anls"] for s in scores) / max(len(scores), 1)
        per[f] = {"hit_rate": hits / max(len(scores), 1), "anls_mean": anls_v}
    agg["per_field"] = per

    ev_tag = "_evidence" if args.with_evidence else ""
    safe = args.engine.replace(":", "_")
    stem = f"funsd_{safe}{ev_tag}"
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / f"preds_{stem}.jsonl").open("w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    (RESULTS / f"scores_{stem}.json").write_text(json.dumps(agg, indent=2))
    print(json.dumps(agg, indent=2))
    print(f"wrote {stem} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
