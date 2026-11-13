#!/usr/bin/env python3
"""Draw a disjoint real-document holdout for the Portuguese probe.

The seed-42 probe is the development set: it is where failure modes are found
and fixes are designed. This script draws a second 100-document sample from the
remaining 843 unique documents so a fix can be validated without reusing the
development documents. The holdout is frozen before any fixed-prompt run.

Usage:
    python evaluation/invoice/make_pt_holdout.py \
        --manifest evaluation/invoice/manifest_pt_real.jsonl \
        --exclude evaluation/invoice/manifest_pt_real_probe100.jsonl \
        --out evaluation/invoice/manifest_pt_real_holdout100.jsonl \
        --size 100 --seed 43
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument(
        "--exclude",
        required=True,
        type=Path,
        nargs="+",
        help="One or more manifests whose document ids must not be sampled",
    )
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--size", type=int, default=100)
    ap.add_argument("--seed", type=int, default=43)
    args = ap.parse_args()

    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    excluded = set()
    for path in args.exclude:
        excluded.update(
            str(row["id"])
            for row in (
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        )
    pool = [row for row in rows if str(row["id"]) not in excluded]
    if len(pool) < args.size:
        raise SystemExit(f"only {len(pool)} documents left after exclusion; need {args.size}")

    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in pool:
        by_source[row["category"]].append(row)

    total = len(pool)
    company = by_source.get("real_pt_company", [])
    personal = by_source.get("real_pt_personal", [])
    n_company = min(len(company), round(args.size * len(company) / total))
    n_personal = min(len(personal), args.size - n_company)
    n_company = min(len(company), args.size - n_personal)

    rng = random.Random(args.seed)
    sample = []
    for category, count in (("real_pt_company", n_company), ("real_pt_personal", n_personal)):
        candidates = sorted(by_source.get(category, []), key=lambda row: row["id"])
        sample.extend(rng.sample(candidates, count))
    sample.sort(key=lambda row: row["id"])

    overlap = {row["id"] for row in sample} & excluded
    if overlap:
        raise SystemExit(f"holdout overlaps development probe: {sorted(overlap)[:3]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in sample:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "out": str(args.out),
                "size": len(sample),
                "seed": args.seed,
                "sources": dict(sorted(Counter(row["category"] for row in sample).items())),
                "pool": total,
                "field_scored": dict(
                    sorted(Counter(field for row in sample for field in row["expected"]["scored"]).items())
                ),
                "overlap": len(overlap),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
