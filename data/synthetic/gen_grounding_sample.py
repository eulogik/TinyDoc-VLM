#!/usr/bin/env python3
"""
Prototype: synthetic documents WITH grounding annotations.

Generates KIE-heavy doc types (invoice, receipt, id_card, form) rendered with
exact element positions (free supervision from the renderer), and emits
training pairs in the standard manifest schema:

    {image_path, prompt, target, source}

source = "synthetic_grounding". Images are saved UNROTATED (grayscale/JPEG
noise only) because bbox targets are computed at render time; rotation would
invalidate pixel coordinates.

Usage:
    python data/synthetic/gen_grounding_sample.py --num-docs 200 \
        --output-dir data/training/grounding_proto
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from PIL import Image

from data.synthetic.generator import ContentGenerator
from data.synthetic.pil_renderer import render_document_with_annotations, augment_image
from data.synthetic.grounding_pairs import build_grounding_pairs, verify_boxes

logger = logging.getLogger(__name__)

GROUNDING_TYPES = ["invoice", "receipt", "id_card", "form"]


def noise_augment(img: Image.Image) -> Image.Image:
    """Augmentations that preserve geometry (NO rotation for grounding docs)."""
    if random.random() > 0.5:
        img = img.convert("L").convert("RGB")
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-docs", type=int, default=200)
    ap.add_argument("--output-dir", type=str, default="data/training_grounding_proto")
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--verify", action="store_true", default=True)
    ap.add_argument("--no-verify", dest="verify", action="store_false")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    random.seed(args.seed)

    out_dir = Path(args.output_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    total_pairs = 0
    verify_failures = 0

    for i in range(args.num_docs):
        doc_type = GROUNDING_TYPES[i % len(GROUNDING_TYPES)]
        content = ContentGenerator.generate(doc_type)
        img, annotations = render_document_with_annotations(doc_type, content)

        if not annotations:
            logger.warning("doc %d (%s): no annotations, skipping", i, doc_type)
            continue

        if args.verify:
            problems = verify_boxes(img, annotations)
            if problems:
                verify_failures += 1
                for p in problems[:3]:
                    logger.warning("doc %d: %s", i, p)

        # NO rotation: bbox targets must stay pixel-accurate.
        img = noise_augment(img)

        image_filename = f"{doc_type}_g{i:06d}.png"
        image_path = images_dir / image_filename
        img.save(str(image_path), "PNG", optimize=True)

        rng = random.Random(args.seed + i)
        pairs = build_grounding_pairs(annotations, rng)
        rel_path = f"data/training/{out_dir.name}/images/{image_filename}"

        for pair in pairs:
            manifest.append({
                "image_path": rel_path,
                "prompt": pair["prompt"],
                "target": pair["target"],
                "source": "synthetic_grounding",
            })
        total_pairs += len(pairs)

        if (i + 1) % 50 == 0:
            logger.info("%d/%d docs, %d pairs so far", i + 1, args.num_docs, total_pairs)

    manifest_path = out_dir / "manifest.jsonl"
    with open(manifest_path, "w") as f:
        for entry in manifest:
            f.write(json.dumps(entry) + "\n")

    stats = {
        "docs": args.num_docs,
        "pairs": total_pairs,
        "avg_pairs_per_doc": round(total_pairs / max(args.num_docs, 1), 2),
        "verify_failures": verify_failures,
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    logger.info("Done: %s", stats)
    logger.info("Manifest -> %s", manifest_path)


if __name__ == "__main__":
    main()
