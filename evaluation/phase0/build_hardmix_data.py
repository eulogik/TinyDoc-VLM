"""Build chained-SFT dataset from hardmine mix (Stage-2I).

Reads $DROOT/hardmine/mix.jsonl (70 docs: 35 teacher-address-miss + 35 easy,
gold answers) and writes $DROOT/hardmix/hf_hardmix (+ data/*.parquet).
Warm-start target: stage2e global_400 (val 0.6188). ~9 iters/epoch at
eff.batch 8; RTARGET=40 (~4 epochs).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "evaluation" / "phase0"))
sys.path.insert(0, str(ROOT / "sdk"))

from build_stage1_manifest import answer_json, messages_row
from tinydoc.pipeline import EXTRACT_PROMPT


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--droot", default="/Volumes/KIOXIA 1TB/tinydoc")
    args = ap.parse_args()

    from datasets import Dataset, load_from_disk
    from PIL import Image

    droot = Path(args.droot)
    with open(droot / "hardmine" / "mix.jsonl") as _fh:
        _lines = [l for l in _fh.read().splitlines() if l.strip()]
    mix = [json.loads(l) for l in _lines]
    rows = []
    for m in mix:
        img = Image.open(m["image_path"]).convert("RGB")
        ans = answer_json(m["gold"])
        rows.append({"image": img, "question": EXTRACT_PROMPT, "answer": ans,
                     "messages": messages_row(EXTRACT_PROMPT, ans)})
    print(f"hardmix rows: {len(rows)}")
    assert len(rows) == 70, f"expected 70 mix rows, got {len(rows)}"
    cols = ["image", "question", "answer", "messages"]
    out = droot / "hardmix" / "hf_hardmix"
    Dataset.from_list([{k: r[k] for k in cols} for r in rows]).save_to_disk(out)
    load_from_disk(str(out)).to_parquet(str(out / "data" / "hardmix-00000-of-00001.parquet"))
    print(f"wrote {out} + parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
