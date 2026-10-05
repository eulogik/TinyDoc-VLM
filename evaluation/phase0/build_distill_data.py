"""Build distillation SFT data: 3B teacher labels on train-420 (+ synth-1500 gold).

Rationale (2026-10-05): LoRA-on-gold SFT moves 2.2B test F1 not at all
(0.536 base -> 0.535-0.538 across lr/data/res/rank configs; paired 4/4/92).
The one untested mechanism with a real signal is the teacher's transcription
style (3B address hit 0.83 vs 0.13-0.20).

Matching (deterministic): the exported train images
(distill_train/images/train_NNNN.jpg) are byte-identical decodes of the real
rows inside hf_train2 (same PIL encode path), so teacher targets are joined
to hf_train2 rows by JPEG-byte equality at quality=90. The script ASSERTS
every teacher row matched exactly once; a silent fallback to gold would waste
the run, so mismatch is a hard failure, not a warning.

Inputs: distill_train/teacher_*.jsonl + stage2_data/hf_train2.
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

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "evaluation" / "phase0"))
sys.path.insert(0, str(ROOT / "sdk"))

from build_stage1_manifest import FIELDS, answer_json, messages_row
from tinydoc.pipeline import EXTRACT_PROMPT


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
    args = ap.parse_args()

    from datasets import Dataset, load_from_disk
    from PIL import Image

    droot = Path(args.droot)

    # 1) teacher targets keyed by image bytes
    teach: dict[bytes, dict] = {}
    n_files = n_skip = 0
    for jf in sorted(glob.glob(str(droot / "distill_train" / "teacher_*.jsonl"))):
        n_files += 1
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
            teach[jpeg_bytes(img)] = (ans, messages_row(EXTRACT_PROMPT, ans))
    print(f"teacher files: {n_files}, usable targets: {len(teach)}, skipped: {n_skip}")
    assert n_files >= 28, f"teacher labeling incomplete: only {n_files} chunk files"
    assert len(teach) >= 400, f"too few teacher targets: {len(teach)}"

    # 2) join onto hf_train2 (1920 rows): real rows get teacher targets
    syn = load_from_disk(str(droot / "stage2_data" / "hf_train2"))
    all_rows, n_rep = [], 0
    for r in syn:
        img = r["image"]
        key = jpeg_bytes(img)
        if key in teach:
            ans, msg = teach[key]
            all_rows.append({"image": img, "question": EXTRACT_PROMPT,
                             "answer": ans, "messages": msg})
            n_rep += 1
        else:
            all_rows.append({"image": img, "question": EXTRACT_PROMPT,
                             "answer": r["answer"], "messages": r["messages"]})
    print(f"matched real rows: {n_rep} (synth untouched: {len(all_rows) - n_rep})")
    assert n_rep == len(teach), f"match fault: {n_rep} rows for {len(teach)} targets"
    assert n_rep == 420, f"expected all 420 real rows replaced, got {n_rep}"

    rng = random.Random(args.seed)
    rng.shuffle(all_rows)
    out = droot / "distill_data" / "hf_distill"
    cols = ["image", "question", "answer", "messages"]
    Dataset.from_list([{k: r[k] for k in cols} for r in all_rows]).save_to_disk(out)
    load_from_disk(str(out)).to_parquet(str(out / "data" / "distill-00000-of-00001.parquet"))
    summary = {"n_teacher_targets": len(teach), "n_replaced": n_rep,
               "n_train": len(all_rows), "seed": args.seed,
               "question": "sdk.tinydoc.pipeline.EXTRACT_PROMPT"}
    (droot / "distill_data" / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
