"""Build Stage-2 SFT data: 420 real SROIE receipts + 1500 synthetic receipts.

Why: Stage-1 (23 epochs on 420 docs, no augmentation, no weight decay) memorized
(train loss -> 0.0004, val F1 0.30, test 0.23). The literature converges:
2-3 epochs (Donut official; receipt-kie-vlm twin: 2 epochs/140 steps, JSON
19->85%), weight_decay 0.01 + warmup on <500-example sets, early stopping.
More diverse examples per epoch beats more epochs on the same examples
(epoch-exposure study). Synthetic receipts with PERFECT labels (in-repo
generator, Faker content) + receipt-grade augmentation supply that diversity.

Gold convention matches SROIE (transcribe what is printed; totals without
currency symbols, as SROIE golds do).

Frozen: val-40 (sroie_val.json) and test-100 (sroie_eval_clean.json) are NEVER
touched — same bar for every config. Synthetic-vs-clean/val perceptual check
asserts no leakage by construction accident.

Writes (all under --out, default KIOXIA):
  hf_train2/  HF dataset (image, question, answer, messages)
  hf_train2/data/*.parquet  trainer-readable
  stage2_manifest_summary.json (in evaluation results)
"""

from __future__ import annotations

import argparse
import io
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "sdk"))
sys.path.insert(0, str(ROOT / "evaluation" / "phase0"))
sys.path.insert(0, str(ROOT))

from build_stage1_manifest import (
    FIELDS,
    answer_json,
    correlations,
    messages_row,
    thumb,
)
from tinydoc.pipeline import EXTRACT_PROMPT

from data.synthetic.generator import ContentGenerator
from data.synthetic.pil_renderer import (
    augment_receipt_strong,
    render_receipt,
)

RESULTS = ROOT / "evaluation/phase0/results"


def synth_gold(content: dict) -> dict:
    """SROIE-convention gold for a synthetic receipt (transcribe as printed)."""
    total = str(content.get("total", ""))
    total = re.sub(r"[^\d.]", "", total)  # "$258.72" -> "258.72" (SROIE style)
    return {
        "company": str(content.get("store_name", "")).strip(),
        "date": str(content.get("txn_date", "")).strip(),
        "address": str(content.get("store_address", "")).strip(),
        "total": total.strip(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-synth", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--out", default="/Volumes/KIOXIA 1TB/tinydoc/stage2_data")
    ap.add_argument("--stage1", default="/Volumes/KIOXIA 1TB/tinydoc/stage1_data")
    ap.add_argument("--leak-corr", type=float, default=0.90)
    args = ap.parse_args()

    from datasets import Dataset, load_from_disk
    from PIL import Image

    rng = random.Random(args.seed)
    out = Path(args.out)
    img_dir = out / "synth_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # 1) synthetic receipts (seeded Faker content + strong augmentation)
    synth_rows = []
    for k in range(args.n_synth):
        r = random.Random(args.seed * 1000003 + k)
        content = ContentGenerator.receipt()
        base = render_receipt(content)
        img = augment_receipt_strong(base, r)
        gold = synth_gold(content)
        assert all(gold[f] for f in FIELDS), f"incomplete synth gold {k}: {gold}"
        name = f"synth_{k:04d}.jpg"
        img.convert("RGB").save(img_dir / name, quality=90)
        ans = answer_json(gold)
        synth_rows.append({
            "image": img,
            "question": EXTRACT_PROMPT,
            "answer": ans,
            "messages": messages_row(EXTRACT_PROMPT, ans),
            "gold": gold,
            "synthetic": True,
        })
    print(f"synthetic receipts: {len(synth_rows)}")

    # 2) real rows (the leakage-filtered Stage-1 train, unchanged)
    tr = load_from_disk(str(Path(args.stage1) / "train"))
    real_rows = []
    for ex in tr:
        img = ex["image"]
        if not isinstance(img, Image.Image):
            img = Image.open(io.BytesIO(img["bytes"]))
        # Stage-1 train rows carry no gold column; gold == the answer JSON
        try:
            ans_obj = json.loads(ex["answer"])
        except Exception:  # noqa: BLE001
            ans_obj = {}
        gold = {f: str(ans_obj.get(f, "") or "").strip() for f in FIELDS}
        assert all(gold[f] for f in FIELDS), f"incomplete real gold: {gold}"
        ans = answer_json(gold)
        real_rows.append({
            "image": img,
            "question": EXTRACT_PROMPT,
            "answer": ans,
            "messages": messages_row(EXTRACT_PROMPT, ans),
            "gold": gold,
            "synthetic": False,
        })
    print(f"real rows: {len(real_rows)}")

    # 3) shuffle (seeded) -> all train; val/test stay frozen
    all_rows = real_rows + synth_rows
    rng.shuffle(all_rows)
    print(f"stage2 train: {len(all_rows)} (real {len(real_rows)} + synth {len(synth_rows)})")

    # 4) integrity: synthetic must not duplicate clean test / val
    ref_thumbs = []
    for jf in ("sroie_eval_clean.json", "sroie_val.json"):
        for c in json.loads((RESULTS / jf).read_text()):
            with Image.open(c["image_path"]) as im:
                ref_thumbs.append(thumb(im, Image.Resampling.BILINEAR))
    R = np.stack(ref_thumbs)
    S = np.stack([thumb(Image.open(img_dir / f"synth_{k:04d}.jpg"),
                        Image.Resampling.BILINEAR) for k in range(len(synth_rows))])
    cross = int((correlations(S, R) >= args.leak_corr).sum())
    print(f"synthetic-vs-clean/val pairs >= {args.leak_corr}: {cross}")
    assert cross == 0, f"synthetic leakage into frozen sets: {cross} pairs"

    # 5) write HF dataset + trainer parquet
    cols = ["image", "question", "answer", "messages"]
    Dataset.from_list([{k: r[k] for k in cols} for r in all_rows]).save_to_disk(out / "hf_train2")
    (
        load_from_disk(str(out / "hf_train2"))
        .to_parquet(str(out / "hf_train2" / "data" / "train2-00000-of-00001.parquet"))
    )
    summary = {
        "n_real": len(real_rows),
        "n_synth": len(synth_rows),
        "n_train": len(all_rows),
        "seed": args.seed,
        "frozen_val": "evaluation/phase0/results/sroie_val.json (40)",
        "frozen_test": "evaluation/phase0/results/sroie_eval_clean.json (100)",
        "synthetic_leak_pairs": cross,
        "question": "sdk.tinydoc.pipeline.EXTRACT_PROMPT (identical to eval prompt)",
        "gold_convention": "SROIE style (transcribe as printed; totals without currency)",
    }
    (RESULTS / "stage2_manifest_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
