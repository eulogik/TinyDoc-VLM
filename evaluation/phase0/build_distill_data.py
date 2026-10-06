"""Build distillation SFT data: 3B teacher labels on train-420 (+ synth-1500 gold).

Rationale (2026-10-05/06): LoRA-on-gold SFT moves 2.2B test F1 not at all
(0.536 base -> 0.535-0.538 across lr/data/res/rank configs; paired 4/4/92).
The one untested mechanism with a real signal is the teacher's transcription
style (3B address hit 0.83 vs 0.13-0.20).

Construction (no fragile joins):
  real rows: distill_train/images/train_NNNN.jpg + teacher_*.jsonl answers
             (labeling read exactly these files, so pairing is by construction;
             415/420 usable — 5 skipped incomplete, counted below).
  synth rows: hf_train2 rows NOT byte-matching current stage1/train
              (stage1[0] matches exactly 1 hf_train2 row; real count asserted
              420, synth complement asserted 1500).

Provenance note 2026-10-06: exported train images are NOT pixel-equal to
current stage1/train rows (same size/mode, different pixels; cause
undetermined — possibly a manifest rebuild between writes). This script
therefore never joins across builds; the leak-freedom that matters is
ASSERTED directly: zero >=0.90 perceptual pairs between the distill train
images and the frozen clean-100 / val-40 sets.

Gold is NEVER consulted for real rows.
Writes: distill_data/hf_distill (+ data/*.parquet for the trainer).
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "evaluation" / "phase0"))
sys.path.insert(0, str(ROOT / "sdk"))

from build_stage1_manifest import (
    FIELDS,
    answer_json,
    correlations,
    messages_row,
    thumb,
)
from tinydoc.pipeline import EXTRACT_PROMPT

RESULTS = ROOT / "evaluation/phase0/results"


def jpeg_bytes(img) -> bytes:
    from PIL import Image as _I

    if not isinstance(img, _I.Image):
        img = _I.open(img)
    b = io.BytesIO()
    img.convert("RGB").save(b, format="JPEG", quality=90)
    return b.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--droot", default="/Volumes/KIOXIA 1TB/tinydoc")
    ap.add_argument("--seed", type=int, default=20261005)
    ap.add_argument("--leak-corr", type=float, default=0.90)
    args = ap.parse_args()

    from datasets import Dataset, load_from_disk
    from PIL import Image

    droot = Path(args.droot)

    # 1) real rows: export images + teacher answers (paired by construction)
    teach_rows, n_skip = [], 0
    for jf in sorted(glob.glob(str(droot / "distill_train" / "teacher_*.jsonl"))):
        with open(jf) as _fh:
            _lines = _fh.readlines()
        for line in _lines:
            r = json.loads(line)
            t = r.get("teacher", {})
            if r.get("error") or not all(str(t.get(f, "")).strip() for f in FIELDS):
                n_skip += 1
                continue
            gold = {f: str(t[f]).strip() for f in FIELDS}
            ans = answer_json(gold)
            img = Image.open(r["image_path"]).convert("RGB")
            teach_rows.append({"image": img, "question": EXTRACT_PROMPT,
                               "answer": ans,
                               "messages": messages_row(EXTRACT_PROMPT, ans)})
    print(f"teacher real rows: {len(teach_rows)} (skipped: {n_skip})")
    assert len(teach_rows) >= 400, f"too few teacher rows: {len(teach_rows)}"

    # 2) synth rows: hf_train2 complement of current stage1/train (byte match)
    st = load_from_disk(str(droot / "stage1_data" / "train"))
    st_keys = set()
    for ex in st:
        st_keys.add(jpeg_bytes(ex["image"]))
    print(f"stage1 train keys: {len(st_keys)}")
    syn = load_from_disk(str(droot / "stage2_data" / "hf_train2"))
    synth_rows, n_real = [], 0
    for r in syn:
        if jpeg_bytes(r["image"]) in st_keys:
            n_real += 1
            continue
        synth_rows.append({"image": r["image"], "question": EXTRACT_PROMPT,
                           "answer": r["answer"], "messages": r["messages"]})
    print(f"hf_train2 real: {n_real}, synth: {len(synth_rows)}")
    assert n_real == 420, f"expected 420 real rows in hf_train2, got {n_real}"
    assert len(synth_rows) == 1500, f"expected 1500 synth rows, got {len(synth_rows)}"

    all_rows = teach_rows + synth_rows
    rng = random.Random(args.seed)
    rng.shuffle(all_rows)
    print(f"distill train: {len(all_rows)} (teacher {len(teach_rows)} + synth {len(synth_rows)})")

    # 3) leak-freedom asserted directly on the final image set
    ref = []
    for jf in ("sroie_eval_clean.json", "sroie_val.json"):
        for c in json.loads((RESULTS / jf).read_text()):
            with Image.open(c["image_path"]) as im:
                ref.append(thumb(im, Image.Resampling.BILINEAR))
    R = np.stack(ref)
    S = np.stack([thumb(r["image"], Image.Resampling.BILINEAR) for r in all_rows])
    cross = int((correlations(S, R) >= args.leak_corr).sum())
    print(f"distill-vs-frozen pairs >= {args.leak_corr}: {cross}")
    assert cross == 0, f"distill leakage into frozen sets: {cross} pairs"

    out = droot / "distill_data" / "hf_distill"
    cols = ["image", "question", "answer", "messages"]
    Dataset.from_list([{k: r[k] for k in cols} for r in all_rows]).save_to_disk(out)
    load_from_disk(str(out)).to_parquet(str(out / "data" / "distill-00000-of-00001.parquet"))
    summary = {"n_teacher_real": len(teach_rows), "n_skipped": n_skip,
               "n_synth": len(synth_rows), "n_train": len(all_rows),
               "leak_pairs": cross, "seed": args.seed,
               "question": "sdk.tinydoc.pipeline.EXTRACT_PROMPT"}
    (droot / "distill_data" / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
