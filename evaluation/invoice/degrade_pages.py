#!/usr/bin/env python3
"""Render a degraded ("photo/scan") copy of an invoice page manifest.

The clean multipage probe is PDF-rendered text. Real uploads are photos,
scans, and re-compressed images, so this probe asks whether the page-wise
pipeline and the OCR total reader survive a realistic degradation. The recipe is
deterministic from a seed; the degraded images are scratch artifacts and can be
rebuilt.

Degradations, all mild on purpose (this is a robustness probe, not an
adversarial benchmark):

- downscale to ``--downscale-px`` then upscale back (lost detail)
- rotate by up to ``--rotation`` degrees, centre-cropped back to page size
- Gaussian blur
- brightness/contrast jitter
- Gaussian sensor noise
- JPEG re-compression at ``--quality``

No model calls. Usage::

    python evaluation/invoice/degrade_pages.py \
        --manifest evaluation/invoice/manifest_multipage.jsonl \
        --out-dir "$SCRATCH/degraded_mp" \
        --out-manifest "$SCRATCH/degraded_mp/manifest.jsonl" \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def _rng(seed: int, doc_id: str, page: int) -> random.Random:
    return random.Random(f"{seed}:{doc_id}:{page}")


def degrade_image(
    image_path: str,
    out_path: str,
    *,
    seed: int = 42,
    doc_id: str = "",
    page: int = 1,
    max_px: int = 1400,
    downscale_px: int = 1100,
    rotation: float = 1.2,
    blur: float = 0.8,
    quality: int = 58,
) -> dict[str, Any]:
    """Degrade one page and return the parameters used (for provenance)."""
    rng = _rng(seed, doc_id, page)
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    width, height = image.size
    if max(width, height) > max_px:
        scale = max_px / max(width, height)
        width, height = max(1, int(width * scale)), max(1, int(height * scale))
        image = image.resize((width, height), Image.Resampling.LANCZOS)

    # Lose effective resolution, then put it back: the model still sees a
    # 1400 px page, but the small print has fewer real pixels.
    small = min(downscale_px, max(width, height))
    if small < max(width, height):
        ratio = small / max(width, height)
        reduced = image.resize(
            (max(1, int(width * ratio)), max(1, int(height * ratio))),
            Image.Resampling.LANCZOS,
        )
        image = reduced.resize((width, height), Image.Resampling.BICUBIC)

    angle = rng.uniform(-rotation, rotation)
    if abs(angle) > 0.01:
        rotated = image.rotate(
            angle,
            resample=Image.Resampling.BICUBIC,
            expand=True,
            fillcolor=(250, 250, 247),
        )
        left = max(0, (rotated.width - width) // 2)
        top = max(0, (rotated.height - height) // 2)
        image = rotated.crop((left, top, left + width, top + height))

    blur_radius = rng.uniform(0.2, blur)
    if blur_radius > 0.01:
        image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    brightness = rng.uniform(0.93, 1.05)
    contrast = rng.uniform(0.90, 1.08)
    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageEnhance.Contrast(image).enhance(contrast)

    noise_sigma = rng.uniform(1.5, 6.0)
    array = np.asarray(image).astype(np.float32)
    noise = np.random.default_rng(rng.getrandbits(32)).normal(
        0.0, noise_sigma, array.shape
    )
    array = np.clip(array + noise, 0, 255).astype(np.uint8)
    image = Image.fromarray(array, mode="RGB")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out, format="JPEG", quality=quality, optimize=True, subsampling=2)
    return {
        "rotation_deg": round(angle, 3),
        "blur_radius": round(blur_radius, 3),
        "brightness": round(brightness, 3),
        "contrast": round(contrast, 3),
        "noise_sigma": round(noise_sigma, 3),
        "jpeg_quality": quality,
        "downscale_px": small,
        "max_px": max_px,
    }


def degrade_manifest(
    manifest_path: str | Path,
    out_dir: str | Path,
    out_manifest: str | Path,
    *,
    seed: int = 42,
    max_px: int = 1400,
    downscale_px: int = 1100,
    rotation: float = 1.2,
    blur: float = 0.8,
    quality: int = 58,
) -> dict[str, Any]:
    """Degrade every page in a manifest and write a parallel manifest."""
    rows = [
        json.loads(line)
        for line in Path(manifest_path).read_text().splitlines()
        if line.strip()
    ]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[dict] = []
    pages = 0
    for row in rows:
        doc_id = row["id"]
        source_paths = row.get("image_paths") or (
            [row["image_path"]] if row.get("image_path") else []
        )
        if not source_paths:
            raise ValueError(f"{doc_id}: no image paths")
        degraded_paths = []
        parameters = []
        for page, source in enumerate(source_paths, start=1):
            out = out_dir / f"{doc_id}_p{page}.jpg"
            parameters.append(
                degrade_image(
                    str(source),
                    str(out),
                    seed=seed,
                    doc_id=doc_id,
                    page=page,
                    max_px=max_px,
                    downscale_px=downscale_px,
                    rotation=rotation,
                    blur=blur,
                    quality=quality,
                )
            )
            degraded_paths.append(str(out))
            pages += 1
        updated = dict(row)
        updated["image_paths"] = degraded_paths
        if "image_path" in row:
            updated["image_path"] = degraded_paths[0]
        updated["degradation"] = {
            "seed": seed,
            "recipe": "downscale+rotate+blur+brightness+noise+jpeg",
            "pages": parameters,
        }
        written.append(updated)

    out_manifest = Path(out_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with out_manifest.open("w") as handle:
        for row in written:
            handle.write(json.dumps(row) + "\n")
    return {
        "documents": len(written),
        "pages": pages,
        "seed": seed,
        "manifest": str(out_manifest),
        "out_dir": str(out_dir),
        "recipe": {
            "max_px": max_px,
            "downscale_px": downscale_px,
            "rotation": rotation,
            "blur": blur,
            "quality": quality,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out-manifest", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-px", type=int, default=1400)
    ap.add_argument("--downscale-px", type=int, default=1100)
    ap.add_argument("--rotation", type=float, default=1.2)
    ap.add_argument("--blur", type=float, default=0.8)
    ap.add_argument("--quality", type=int, default=58)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()
    report = degrade_manifest(
        args.manifest,
        args.out_dir,
        args.out_manifest,
        seed=args.seed,
        max_px=args.max_px,
        downscale_px=args.downscale_px,
        rotation=args.rotation,
        blur=args.blur,
        quality=args.quality,
    )
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
