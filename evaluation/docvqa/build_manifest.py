#!/usr/bin/env python3
"""Build a deterministic DocVQA reading-probe manifest.

DocVQA is a public document-reading benchmark (questions + short answers). This
probe uses one question per image, sampled deterministically, to measure whether
the base reader can resolve document text at all. It is not an extraction metric
and not a product claim: DocVQA is public and may be in the base model's
pretraining.

The pair file is JSONL with `image_path`, `prompt`, `target` (the lmms-lab
mirror used by this repo). No model is called here.

Usage:
    python evaluation/docvqa/build_manifest.py \
        --pairs data/datasets/docvqa/docvqa.json \
        --data-dir data/datasets/docvqa \
        --out evaluation/docvqa/manifest_docvqa_200.jsonl \
        --limit 200
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

QUESTION_PREFIX = re.compile(r"^\s*answer the question\s*:\s*", re.IGNORECASE)


def clean_question(prompt: str) -> str:
    return QUESTION_PREFIX.sub("", str(prompt or "")).strip()


def select_pairs(rows: list[dict], limit: int, seed: int = 42) -> list[dict]:
    """Sample one question per image, deterministically."""
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    picked: list[dict] = []
    seen: set[str] = set()
    for row in shuffled:
        image = Path(str(row.get("image_path", ""))).name
        if not image or image in seen:
            continue
        seen.add(image)
        picked.append(row)
        if len(picked) >= limit:
            break
    return picked


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pairs_path = Path(args.pairs)
    with pairs_path.open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        print(f"no pairs in {pairs_path}")
        return 2

    selected = select_pairs(rows, args.limit, args.seed)
    data_dir = Path(args.data_dir)
    out_rows = []
    missing = 0
    for index, row in enumerate(selected, 1):
        image = Path(str(row["image_path"])).name
        if not (data_dir / "images" / image).exists():
            missing += 1
            continue
        out_rows.append(
            {
                "id": f"docvqa-{index:04d}",
                "image": image,
                "question": clean_question(row.get("prompt", "")),
                "target": str(row.get("target", "")),
                "source": row.get("source", "docvqa"),
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for row in out_rows:
            handle.write(json.dumps(row) + "\n")

    first = out_rows[0]["question"] if out_rows else None
    print(
        json.dumps(
            {
                "pairs": str(pairs_path),
                "data_dir": str(data_dir),
                "out": str(out),
                "n": len(out_rows),
                "missing_images": missing,
                "seed": args.seed,
                "unique_images": len({r["image"] for r in out_rows}),
                "first_question": first,
            },
            indent=2,
        )
    )
    print("sources", dict(Counter(r["source"] for r in out_rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
