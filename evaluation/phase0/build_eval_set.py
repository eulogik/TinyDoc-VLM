#!/usr/bin/env python3
"""Build Phase-0 SROIE eval set (100 receipts) + inventory + leakage report.

Vertical: SROIE receipt KIE (company, date, address, total).

Split policy:
  - Hold out 100 images for eval (seed 42), stratified by company fingerprint
    so rare merchants appear in eval when possible.
  - Those image basenames are written to sroie_holdout.txt — future training
    runs MUST exclude them (layout/image leakage guard).

Also writes phase0_inventory.json: source counts, prompt mix, layout risks.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

FIELDS = ("company", "date", "address", "total")
# evaluation/ is a symlink to the backup disk; Path.resolve() would leave the repo.
# Anchor on CWD (run from repo root) or on absolute __file__ without resolving symlinks.
_abs_file = Path(__file__).absolute()
ROOT = _abs_file.parent.parent.parent
if not (ROOT / "data" / "training" / "manifest.jsonl").exists():
    ROOT = Path.cwd()
    if not (ROOT / "data" / "training" / "manifest.jsonl").exists():
        raise SystemExit(f"cannot locate repo root from {_abs_file} or {Path.cwd()}")
MANIFEST = ROOT / "data" / "training" / "manifest.jsonl"
OUT_DIR = Path(__file__).absolute().parent / "results"


def load_sroie_json_rows():
    rows = []
    with open(MANIFEST) as f:
        for line in f:
            d = json.loads(line)
            if d.get("source") != "sroie":
                continue
            if "JSON" not in d.get("prompt", ""):
                continue
            rows.append(d)
    return rows


def inventory():
    stats = json.loads((ROOT / "data" / "training" / "stats.json").read_text())
    prompt_mix = Counter()
    source = Counter()
    doc_type = Counter()
    real_n = 0
    with open(MANIFEST) as f:
        for line in f:
            d = json.loads(line)
            if d.get("source") == "synthetic":
                continue
            real_n += 1
            source[d.get("source", "?")] += 1
            doc_type[d.get("doc_type") or "(none)"] += 1
            p = d.get("prompt", "")
            if p.startswith("Answer") or p.startswith("What") or p.startswith("Who") or p.startswith("Answer the question"):
                prompt_mix["qa"] += 1
            elif p.startswith("Extract all text"):
                prompt_mix["ocr"] += 1
            elif "JSON" in p:
                prompt_mix["kie_json"] += 1
            else:
                prompt_mix["other"] += 1
    return {
        "stats_file": stats,
        "real_pairs": real_n,
        "real_by_source": dict(source),
        "real_by_prompt": dict(prompt_mix),
        "real_by_doc_type": dict(doc_type),
        "layout_leakage_risks": [
            "Random 500-eval split in smolvlm2_qlora.py mixes sources; no layout-family holdout.",
            "SROIE/FUNSD share template-like receipts/forms across train/eval if split by pair not image.",
            "docmatix (37,675) dominates real mix (~81%) — answers may share document templates.",
            "No company/template fingerprint split for KIE verticals.",
        ],
        "vertical_choice": {
            "name": "sroie_receipt_kie",
            "fields": list(FIELDS),
            "rationale": "Standard KIE field F1; 971 images with structured JSON GT on disk; clear business mapping to invoice/receipt extraction pricing.",
        },
    }


def build_eval(rows, n=100, seed=42):
    # group by image
    by_img = defaultdict(list)
    for r in rows:
        by_img[r["image_path"]].append(r)

    # company fingerprint
    def company(path: str) -> str:
        for r in by_img[path]:
            try:
                return str(json.loads(r["target"]).get("company", ""))
            except Exception:
                continue
        return ""

    images = sorted(by_img.keys())
    by_company = defaultdict(list)
    for img in images:
        by_company[company(img)].append(img)

    rng = random.Random(seed)
    eval_imgs = []
    # take at least one per company until n, then fill randomly
    for comp, imgs in sorted(by_company.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        rng.shuffle(imgs)
        if imgs:
            eval_imgs.append(imgs[0])
        if len(eval_imgs) >= n:
            break
    remaining = [i for i in images if i not in set(eval_imgs)]
    rng.shuffle(remaining)
    while len(eval_imgs) < n and remaining:
        eval_imgs.append(remaining.pop())
    eval_imgs = eval_imgs[:n]
    missing = [p for p in eval_imgs if not Path(p).exists()]
    if missing:
        print(f"WARNING: {len(missing)} missing images, replacing")
        for p in missing:
            eval_imgs.remove(p)
            if remaining:
                eval_imgs.append(remaining.pop())

    eval_set = []
    for img in eval_imgs:
        gold = None
        for r in by_img[img]:
            try:
                gold = json.loads(r["target"])
                break
            except Exception:
                continue
        if not gold:
            continue
        eval_set.append(
            {
                "id": Path(img).stem,
                "image_path": img,
                "source": "sroie",
                "prompt": "Extract the document as JSON:",
                "gold": {k: str(gold.get(k, "")) for k in FIELDS},
                "company": str(gold.get("company", "")),
            }
        )
    return eval_set


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    inv = inventory()
    rows = load_sroie_json_rows()
    inv["sroie_json_pairs"] = len(rows)
    inv["sroie_unique_images"] = len({r["image_path"] for r in rows})

    eval_set = build_eval(rows, n=args.n, seed=args.seed)
    inv["eval_n"] = len(eval_set)
    inv["eval_unique_companies"] = len({e["company"] for e in eval_set})

    (OUT_DIR / "sroie_eval.json").write_text(json.dumps(eval_set, indent=2))
    holdout = sorted({e["image_path"] for e in eval_set})
    (OUT_DIR / "sroie_holdout.txt").write_text("\n".join(holdout) + "\n")
    (OUT_DIR / "phase0_inventory.json").write_text(json.dumps(inv, indent=2))

    print(json.dumps({k: inv[k] for k in (
        "real_pairs", "real_by_source", "real_by_prompt",
        "sroie_json_pairs", "eval_n", "eval_unique_companies", "vertical_choice",
    )}, indent=2))
    print(f"wrote {OUT_DIR/'sroie_eval.json'}")
    print(f"wrote {OUT_DIR/'sroie_holdout.txt'}")


if __name__ == "__main__":
    main()
