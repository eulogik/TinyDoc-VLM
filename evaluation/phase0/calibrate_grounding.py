#!/usr/bin/env python3
"""Calibrate grounding_gate thresholds against BASE models (no adapter).

Records base image-dependence / mean entropy / QA acc so adapter runs can be
compared relatively (adapter entropy >= alpha * base, etc.).

Writes results/grounding_gate.{engine}.json

Usage:
  python evaluation/phase0/calibrate_grounding.py --engine smolvlm2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).absolute().parent))
from engines import build_engine  # noqa: E402

ROOT = Path(__file__).absolute().parent.parent.parent
# Prefer the real 12-page synthetic eval manifest (known layout diversity)
GATE_MANIFEST = ROOT / "data" / "eval_768" / "manifest.json"
RESULTS = Path(__file__).absolute().parent / "results"


def run_gate(engine_name: str, limit_pages: int | None = None, limit_qa: int | None = None):
    """Mirror training/grounding_gate.py logic but through phase0 engines + SROIE QA
    when using sroie eval, or eval_768 pages when present."""
    import re
    import torch
    from PIL import Image

    RESULTS.mkdir(parents=True, exist_ok=True)
    engine = build_engine(engine_name)

    # Build page list
    pages = []
    if GATE_MANIFEST.exists():
        man = json.loads(GATE_MANIFEST.read_text())
        for m in man[:limit_pages]:
            pages.append((m["doc_type"], m["page"], m.get("qa", [])))
    else:
        # fallback: sroie eval images, synthetic QA from gold fields
        eval_set = json.loads((RESULTS / "sroie_eval.json").read_text())[
            : limit_pages or 12
        ]
        for e in eval_set:
            g = e["gold"]
            qa = [
                {"question": "What is the total amount?", "answer": g["total"]},
                {"question": "Who is the company?", "answer": g["company"]},
                {"question": "What is the date?", "answer": g["date"]},
            ]
            pages.append(("receipt", e["image_path"], qa))

    probe_q = "What is the total amount?"

    def norm(s: str) -> str:
        return re.sub(r"[\$£€]|\s+", "", (s or "").lower()).strip(" .,:;\"'`!?")

    def gen(img_path: str, q: str) -> str:
        from metrics import extract_json  # noqa
        # reuse engine raw path with a free-form question via smolvlm generate-like
        # For simplicity, ask engine to extract then map — not ideal for dep probe.
        # Better: if engine is smolvlm2, call its generate. Use extract for all:
        out = engine.extract(img_path)
        f = out.get("fields", {})
        if "total" in q.lower():
            return str(f.get("total", ""))
        if "company" in q.lower() or "who" in q.lower():
            return str(f.get("company", ""))
        if "date" in q.lower():
            return str(f.get("date", ""))
        return out.get("raw", "")

    # 1) image-dependence: same question across pages
    vals = []
    for dt, path, qa in pages:
        try:
            vals.append(gen(path, probe_q))
        except Exception as e:
            print("dep fail", e)
            vals.append("")
    diff = sum(
        1
        for i in range(len(vals))
        for j in range(i + 1, len(vals))
        if norm(vals[i]) != norm(vals[j])
    )
    total_pairs = len(vals) * (len(vals) - 1) // 2
    dep = diff / total_pairs if total_pairs else 0.0

    # 2) first-token entropy (only if engine exposes model)
    ents = []
    ent_backend = None
    if engine_name == "smolvlm2" and hasattr(engine, "model"):
        ent_backend = "torch"
        for dt, path, qa in pages[:6]:
            q = (qa[0]["question"] if qa else probe_q)
            try:
                img = Image.open(path).convert("RGB")
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": q},
                        ],
                    }
                ]
                text = engine.processor.apply_chat_template(
                    messages, add_generation_prompt=True
                )
                inp = engine.processor(text=[text], images=[[img]], return_tensors="pt")
                inp = {
                    k: (v.to(engine.device) if hasattr(v, "to") else v)
                    for k, v in inp.items()
                    if k != "image_token_id"
                }
                with torch.no_grad():
                    logits = engine.model(**inp).logits[0, -1].float()
                p = torch.softmax(logits, -1)
                e = float(-(p * p.clamp_min(1e-12).log()).sum())
                ents.append(e)
                print(f"entropy {e:.3f} for {q[:50]}")
            except Exception as ex:
                print("entropy fail", ex)
    mean_ent = sum(ents) / len(ents) if ents else None

    # 3) QA accuracy (loose)
    def loose(p: str, t: str) -> bool:
        def n(s):
            return re.sub(r"[\$£€]|\s+", "", (s or "").lower())

        p, t = n(p), n(t)
        return bool(t) and (p == t or t in p or p in t)

    correct = n_q = 0
    for dt, path, qa in pages:
        for item in qa[: limit_qa or 2]:
            n_q += 1
            pred = gen(path, item["question"])
            correct += int(loose(pred, item["answer"]))

    result = {
        "engine": engine_name,
        "n_pages": len(pages),
        "image_dependence": dep,
        "mean_entropy": mean_ent,
        "entropy_backend": ent_backend,
        "qa_acc": correct / n_q if n_q else None,
        "n_qa": n_q,
        "provisional_absolute_gates": {
            "PASS_DEP": 0.70,
            "PASS_ENT": 1.0,
            "PASS_QA": 0.15,
            "warning": "absolute gates NOT calibrated; use base-relative rules once adapter exists",
        },
        "recommended_relative_rules_vs_base": {
            "entropy": "adapter_mean_ent >= 0.5 * base_mean_ent (if base measured)",
            "dependence": "adapter_dep >= base_dep - 0.10",
            "qa_acc": "adapter_qa >= max(base_qa, 0.15)",
        },
        "absolute_gate_pass_flags": {
            "pass_dep": dep >= 0.70,
            "pass_ent": (mean_ent is not None and mean_ent >= 1.0),
            "pass_qa": (correct / n_q >= 0.15) if n_q else None,
        },
    }
    out = RESULTS / f"grounding_calib_{engine_name.replace(':', '_')}.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"wrote {out}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="smolvlm2")
    ap.add_argument("--limit-pages", type=int, default=12)
    ap.add_argument("--limit-qa", type=int, default=2)
    args = ap.parse_args()
    run_gate(args.engine, args.limit_pages, args.limit_qa)


if __name__ == "__main__":
    main()
