#!/usr/bin/env python3
"""Prepare a probe manifest for the product's page-size cap.

`InvoicePipeline` resizes page images so the long side is at most `max_px`
(1400 by default). The evaluation runner sends whatever path the manifest
points at, so this script applies the same cap before inference and records the
transform. It does not call a model.

Usage:
    python evaluation/invoice/prepare_pt_probe.py \
        --manifest evaluation/invoice/manifest_pt_real_probe100.jsonl \
        --out-dir "$TINYDOC_PT_ROOT/prepared_1400" \
        --out-manifest evaluation/invoice/manifest_pt_real_probe100_1400.jsonl \
        --max-px 1400
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_image(source: Path, destination: Path, max_px: int, quality: int) -> dict:
    with Image.open(source) as image:
        image = image.convert("RGB")
        width, height = image.size
        scale = min(1.0, max_px / max(width, height))
        if scale < 1.0:
            size = (max(1, round(width * scale)), max(1, round(height * scale)))
            image = image.resize(size, Image.Resampling.LANCZOS)
            destination.parent.mkdir(parents=True, exist_ok=True)
            image.save(destination, format="JPEG", quality=quality, optimize=True)
            resized = True
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            resized = False
        out_width, out_height = image.size
    return {
        "source_width": width,
        "source_height": height,
        "width": out_width,
        "height": out_height,
        "resized": resized,
        "max_px": max_px,
        "jpeg_quality": quality if resized else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--out-manifest", required=True, type=Path)
    ap.add_argument("--max-px", type=int, default=1400)
    ap.add_argument("--quality", type=int, default=92)
    ap.add_argument("--stats", type=Path, default=None)
    args = ap.parse_args()

    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    out_rows = []
    for index, row in enumerate(rows, 1):
        source = Path(row["image_path"])
        if not source.exists():
            raise SystemExit(f"missing image: {source}")
        destination = args.out_dir / f"{row['id']}.jpg"
        prep = prepare_image(source, destination, args.max_px, args.quality)
        prep["source_image_path"] = str(source)
        prep["source_sha256"] = sha256(source)
        prep["prepared_sha256"] = sha256(destination)
        out = dict(row)
        out["source_image_path"] = row["image_path"]
        out["image_path"] = str(destination)
        out["image_paths"] = [str(destination)]
        out["prep"] = prep
        out_rows.append(out)
        if index % 10 == 0:
            print(f"[{index}/{len(rows)}] {row['id']}", flush=True)

    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.out_manifest.open("w", encoding="utf-8") as handle:
        for row in out_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    sizes = [row["prep"]["width"] * row["prep"]["height"] for row in out_rows]
    stats = {
        "manifest": str(args.manifest),
        "out_manifest": str(args.out_manifest),
        "out_dir": str(args.out_dir),
        "documents": len(out_rows),
        "max_px": args.max_px,
        "jpeg_quality": args.quality,
        "resized": sum(1 for row in out_rows if row["prep"]["resized"]),
        "median_pixels": sorted(sizes)[len(sizes) // 2] if sizes else 0,
        "min_pixels": min(sizes) if sizes else 0,
        "max_pixels": max(sizes) if sizes else 0,
    }
    if args.stats:
        args.stats.parent.mkdir(parents=True, exist_ok=True)
        args.stats.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
