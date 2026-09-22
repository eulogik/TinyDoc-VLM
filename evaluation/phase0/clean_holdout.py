#!/usr/bin/env python3
"""Produce a training manifest that excludes the Phase-0 SROIE holdout.

Contamination (measured): the 100 holdout images appear as 204 full-path rows
in data/training/manifest.jsonl (source=sroie). Name-only extras under other
paths were 0 for this holdout.

Outputs (large file → KIOXIA via data/training symlink):
  - data/training/manifest_train_clean.jsonl  (original − holdout paths)
  - evaluation/phase0/results/holdout_clean_report.json

Free-engine numbers remain valid (engines were not trained on our data).
Any future adapter training MUST use the cleaned manifest; adapter eval uses
the holdout that is absent from that cleaned file.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

_abs = Path(__file__).absolute()
ROOT = _abs.parent.parent.parent
if not (ROOT / "data" / "training" / "manifest.jsonl").exists():
    ROOT = Path.cwd()
MANIFEST = ROOT / "data" / "training" / "manifest.jsonl"
CLEAN = ROOT / "data" / "training" / "manifest_train_clean.jsonl"
HOLDOUT = _abs.parent / "results" / "sroie_holdout.txt"
OUT = _abs.parent / "results" / "holdout_clean_report.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--out", type=Path, default=CLEAN)
    ap.add_argument("--holdout", type=Path, default=HOLDOUT)
    ap.add_argument("--report", type=Path, default=OUT)
    args = ap.parse_args()

    holdout_paths = {ln.strip() for ln in args.holdout.read_text().splitlines() if ln.strip()}
    holdout_names = {Path(p).name for p in holdout_paths}
    holdout_stems = {Path(p).stem for p in holdout_paths}

    n_in = 0
    n_out = 0
    dropped_path = 0
    dropped_sroie_name = 0
    src_before: Counter = Counter()
    src_after: Counter = Counter()
    dropped_rows_sample: list = []

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    with args.manifest.open() as fin, tmp.open("w") as fout:
        for line in fin:
            n_in += 1
            row = json.loads(line)
            src = row.get("source", "?")
            src_before[src] += 1
            ip = row.get("image_path", "")
            drop = False
            if ip in holdout_paths:
                drop = True
                dropped_path += 1
            elif src == "sroie" and (
                Path(ip).name in holdout_names or Path(ip).stem in holdout_stems
            ):
                drop = True
                dropped_sroie_name += 1
            if drop:
                if len(dropped_rows_sample) < 5:
                    dropped_rows_sample.append({"image_path": ip, "source": src})
                continue
            fout.write(line if line.endswith("\n") else line + "\n")
            n_out += 1
            src_after[src] += 1

    tmp.replace(args.out)

    report = {
        "holdout_images": len(holdout_paths),
        "rows_in": n_in,
        "rows_out": n_out,
        "rows_dropped": n_in - n_out,
        "dropped_full_path": dropped_path,
        "dropped_sroie_name_only": dropped_sroie_name,
        "sources_before": dict(src_before),
        "sources_after": dict(src_after),
        "sroie_before": src_before.get("sroie", 0),
        "sroie_after": src_after.get("sroie", 0),
        "clean_manifest": str(args.out),
        "holdout_file": str(args.holdout),
        "dropped_sample": dropped_rows_sample,
        "note": (
            "Free-engine Phase-0 numbers unaffected. Future adapter training must use "
            "manifest_train_clean.jsonl; holdout images are excluded from it."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
