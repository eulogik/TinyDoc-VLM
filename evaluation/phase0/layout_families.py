#!/usr/bin/env python3
"""Layout-family clustering for the SROIE holdout (image-level → template-level).

Current split is image-level only. This assigns each holdout image a layout
family via 8×8 dHash (horizontal-gradient bits; more stable than aHash on
photo receipts), then holds out ~15–20% of families as an *unseen-layout*
eval subset.

Measured on n=100 holdout (pairwise Hamming, dHash8): min=2, median=27/64.
At max_link=16 → ~50 families (multi-image families present; max size ~15).
Unseen selection prefers mid/small families (size ≤ 3) so the mega-cluster
stays in seen (keeps n_unseen usable without dumping the largest template).

Outputs:
  - results/layout_families.json       (per-image family id + dhash)
  - results/sroie_holdout_unseen.txt   (images in held-out families)
  - results/sroie_holdout_seen.txt     (remaining holdout images)

Run after clean_holdout.py so training data never contains these paths.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image

_abs = Path(__file__).absolute()
RESULTS = _abs.parent / "results"
DEFAULT_HOLDOUT = RESULTS / "sroie_holdout.txt"


def dhash(image_path: Path, hash_size: int = 8) -> str:
    """8×8 dHash: grayscale → (size+1)×size → horizontal gradient bits → hex."""
    with Image.open(image_path) as im:
        g = im.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
        pixels = list(g.tobytes())
    # row-major (size+1) × size
    bits = 0
    nbits = hash_size * hash_size
    for r in range(hash_size):
        row_base = r * (hash_size + 1)
        for c in range(hash_size):
            left = pixels[row_base + c]
            right = pixels[row_base + c + 1]
            bits = (bits << 1) | (1 if right > left else 0)
    width = (nbits + 3) // 4
    return f"{bits:0{width}x}"


def hamming_hex(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def cluster_families(items: list[dict], max_link_dist: int = 16) -> dict[str, int]:
    """Greedy single-link clustering on dHash Hamming distance.

    Deterministic: process images in sorted path order; attach to first family
    whose representative is within max_link_dist, else open a new family.
    """
    ordered = sorted(items, key=lambda x: x["image_path"])
    reps: list[str] = []
    assign: dict[str, int] = {}
    for it in ordered:
        h = it["dhash"]
        placed = None
        for fi, rh in enumerate(reps):
            if hamming_hex(h, rh) <= max_link_dist:
                placed = fi
                break
        if placed is None:
            placed = len(reps)
            reps.append(h)
        assign[it["image_path"]] = placed
    return assign


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT)
    ap.add_argument("--out-json", type=Path, default=RESULTS / "layout_families.json")
    ap.add_argument("--out-unseen", type=Path, default=RESULTS / "sroie_holdout_unseen.txt")
    ap.add_argument("--out-seen", type=Path, default=RESULTS / "sroie_holdout_seen.txt")
    ap.add_argument("--hash-size", type=int, default=8)
    ap.add_argument("--max-link", type=int, default=16, help="max Hamming distance to join family")
    ap.add_argument(
        "--holdout-family-frac",
        type=float,
        default=0.18,
        help="fraction of families held out as unseen-layout",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    paths = [ln.strip() for ln in args.holdout.read_text().splitlines() if ln.strip()]
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        print(f"WARNING: {len(missing)} missing images skipped")
    paths = [p for p in paths if Path(p).exists()]

    items = [
        {"image_path": p, "dhash": dhash(Path(p), args.hash_size)} for p in paths
    ]

    assign = cluster_families(items, max_link_dist=args.max_link)

    by_family: dict[int, list[str]] = defaultdict(list)
    for p, fi in assign.items():
        by_family[fi].append(p)

    fam_ids = sorted(by_family.keys(), key=lambda f: (-len(by_family[f]), f))
    rng = random.Random(args.seed)
    order = fam_ids[:]
    rng.shuffle(order)
    n_hold = max(1, round(len(order) * args.holdout_family_frac))
    # Whole families only. Shuffled order is deterministic (seed=42).
    # Prefer mid/small families for unseen so the mega-cluster stays in seen
    # (keeps n_unseen usable without dumping the largest template into unseen).
    mid_small = [f for f in order if len(by_family[f]) <= 3]
    large = [f for f in order if len(by_family[f]) > 3]
    pick = mid_small + large
    unseen_fams = set(pick[:n_hold])
    seen_fams = set(order) - unseen_fams

    unseen_paths = sorted(p for f in unseen_fams for p in by_family[f])
    seen_paths = sorted(p for f in seen_fams for p in by_family[f])

    records = []
    for it in items:
        p = it["image_path"]
        fi = assign[p]
        records.append(
            {
                "image_path": p,
                "family_id": fi,
                "dhash": it["dhash"],
                "split": "unseen_layout" if fi in unseen_fams else "seen_layout",
                "family_size": len(by_family[fi]),
            }
        )

    payload = {
        "n_images": len(records),
        "n_families": len(by_family),
        "family_sizes": {str(f): len(by_family[f]) for f in sorted(by_family)},
        "max_link_dist": args.max_link,
        "hash_size": args.hash_size,
        "hash": "dhash",
        "holdout_family_frac": args.holdout_family_frac,
        "unseen_family_ids": sorted(unseen_fams),
        "n_unseen_images": len(unseen_paths),
        "n_seen_images": len(seen_paths),
        "images": records,
    }
    args.out_json.write_text(json.dumps(payload, indent=2) + "\n")
    args.out_unseen.write_text("\n".join(unseen_paths) + ("\n" if unseen_paths else ""))
    args.out_seen.write_text("\n".join(seen_paths) + ("\n" if seen_paths else ""))

    multi = sum(1 for v in by_family.values() if len(v) > 1)
    print(
        json.dumps(
            {
                "n_images": payload["n_images"],
                "n_families": payload["n_families"],
                "multi_image_families": multi,
                "max_family_size": max((len(v) for v in by_family.values()), default=0),
                "n_unseen_images": payload["n_unseen_images"],
                "n_seen_images": payload["n_seen_images"],
                "unseen_family_ids": payload["unseen_family_ids"],
            },
            indent=2,
        )
    )
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_unseen}")
    print(f"wrote {args.out_seen}")


if __name__ == "__main__":
    main()
