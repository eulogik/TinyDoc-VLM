"""Mine address-hard train docs for chained SFT (Stage-2I).

Hardness signal (no test touched, no extra inference): train docs where the
3B teacher's address disagrees with pool gold. The teacher reads addresses at
0.83, so its misses mark genuinely hard receipts (vs length, which does NOT
predict misses on val: hit-len 74 vs miss-len 69).

Gold recovery: export images are perceptual-matched (64x64 corr >= 0.99)
against the pool parquet; gold comes from pool SROIE tags. Deterministic and
exact — the join key is content, not filenames.

Output: $DROOT/hardmine/hard.jsonl (id, image_path, gold) + hard_ids.txt.
Chained recipe consumes it: warm-start stage2e + short run on hard+easy mix.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "evaluation" / "phase0"))
sys.path.insert(0, str(ROOT / "sdk"))

from build_stage1_manifest import FIELDS, correlations, parse_gold, thumb
from metrics import field_match


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--droot", default="/Volumes/KIOXIA 1TB/tinydoc")
    ap.add_argument("--seed", type=int, default=20261006)
    args = ap.parse_args()

    from datasets import load_dataset
    from PIL import Image

    droot = Path(args.droot)
    out = droot / "hardmine"
    out.mkdir(exist_ok=True)

    # 1) pool images + gold
    ds = load_dataset("parquet",
                      data_dir=str(droot / "data" / "sroie-labels" / "data"),
                      split="train")
    pool_thumbs, pool_gold = [], []

    def _as_pil(v):
        import io as _io2
        from PIL import Image as _I
        if isinstance(v, _I.Image):
            return v
        return _I.open(_io2.BytesIO(v["bytes"]))

    for r in ds:
        im = _as_pil(r["image"]).convert("RGB")
        pool_thumbs.append(thumb(im, Image.Resampling.BILINEAR))
        pool_gold.append(parse_gold(r["text"]))
    P = np.stack(pool_thumbs)
    print(f"pool: {len(pool_gold)}")

    # 2) export images -> pool gold via perceptual match
    man = json.loads((droot / "distill_train" / "manifest.json").read_text())
    exp_thumbs = []
    for m in man:
        with Image.open(m["image_path"]) as im:
            exp_thumbs.append(thumb(im.convert("RGB"), Image.Resampling.BILINEAR))
    E = np.stack(exp_thumbs)
    C = correlations(E, P)
    best, bestv = C.argmax(1), C.max(1)
    assert (bestv >= 0.99).all(), \
        f"{int((bestv < 0.99).sum())} export images without pool match"
    exp_gold = [pool_gold[int(i)] for i in best]
    print(f"export->pool matched: {len(exp_gold)}/{len(man)} "
          f"(min corr {float(bestv.min()):.4f})")

    # 3) teacher labels by export id
    teach: dict[str, dict] = {}
    for jf in sorted(glob.glob(str(droot / "distill_train" / "teacher_*.jsonl"))):
        with open(jf) as _fh:
            _lines = _fh.readlines()
        for line in _lines:
            r = json.loads(line)
            if r.get("error"):
                continue
            t = r.get("teacher", {})
            if all(str(t.get(f, "")).strip() for f in FIELDS):
                teach[r["id"]] = {f: str(t[f]).strip() for f in FIELDS}
    print(f"teacher labels: {len(teach)}")

    # 4) hard = teacher-address miss vs pool gold; easy = rest
    by_id = {m["id"]: (m, g) for m, g in zip(man, exp_gold)}
    hard, easy = [], []
    for tid, t in teach.items():
        if tid not in by_id:
            continue
        m, g = by_id[tid]
        (hard if not field_match(t.get("address", ""), g.get("address", ""),
                                 "address") else easy).append(
            {"id": tid, "image_path": m["image_path"], "gold": g})
    print(f"hard: {len(hard)}, easy: {len(easy)}")
    rng = random.Random(args.seed)
    rng.shuffle(hard)
    rng.shuffle(easy)
    mix = hard + easy[:len(hard)]
    rng.shuffle(mix)
    with open(out / "hard.jsonl", "w") as _f:
        _f.write("\n".join(json.dumps(r) for r in hard) + "\n")
    with open(out / "mix.jsonl", "w") as _f:
        _f.write("\n".join(json.dumps(r) for r in mix) + "\n")
    with open(out / "stats.json", "w") as _f:
        json.dump({"n_hard": len(hard), "n_easy": len(easy),
                   "n_mix": len(mix), "seed": args.seed}, _f, indent=2)
    print(f"mix: {len(mix)} (hard {len(hard)} + easy {len(hard)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
