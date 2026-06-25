#!/usr/bin/env python3
"""
Downloads and formats public benchmark datasets for TinyDoc-VLM evaluation.
Uses HuggingFace datasets library for reliable access.

Usage:
    python evaluation/download_benchmarks.py --data-dir evaluation/data
"""

import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def download_docvqa(data_dir: Path) -> Path:
    """Download DocVQA from HuggingFace datasets."""
    from datasets import load_dataset
    logger.info("Downloading DocVQA...")
    ds = load_dataset("lmms-lab/docvqa", "DocVQA")
    out_dir = data_dir / "docvqa"
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(exist_ok=True)

    formatted = []
    for split in ["train", "val", "test"]:
        if split not in ds:
            continue
        for item in ds[split]:
            img = item.get("image")
            if img is None:
                continue
            img_name = f"{split}_{item.get('questionId', len(formatted)):06d}.png"
            img_path = images_dir / img_name
            if not img_path.exists():
                img.save(str(img_path))
            formatted.append({
                "image": f"docvqa/images/{img_name}",
                "question": item.get("question", ""),
                "answers": [a for a in item.get("answers", []) if a],
            })

    out_path = out_dir / "docvqa.json"
    with open(out_path, "w") as f:
        json.dump(formatted, f, indent=2)
    logger.info(f"DocVQA: {len(formatted)} samples -> {out_path}")
    return out_dir


def download_funsd(data_dir: Path) -> Path:
    """Download FUNSD from HuggingFace datasets."""
    from datasets import load_dataset
    logger.info("Downloading FUNSD...")
    ds = load_dataset("nielsr/funsd")
    out_dir = data_dir / "funsd"
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(exist_ok=True)

    formatted = []
    for split in ["train", "test"]:
        if split not in ds:
            continue
        for item in ds[split]:
            img = item.get("image")
            if img is None:
                continue
            img_name = f"{split}_{len(formatted):06d}.png"
            img_path = images_dir / img_name
            if not img_path.exists():
                img.save(str(img_path))
            formatted.append({
                "image": f"funsd/images/{img_name}",
                "words": item.get("words", []),
                "bboxes": item.get("bboxes", []),
                "labels": item.get("labels", []),
            })

    out_path = out_dir / "funsd.json"
    with open(out_path, "w") as f:
        json.dump(formatted, f, indent=2)
    logger.info(f"FUNSD: {len(formatted)} samples -> {out_path}")
    return out_dir


def download_cord(data_dir: Path) -> Path:
    """Download CORD from HuggingFace datasets."""
    from datasets import load_dataset
    logger.info("Downloading CORD...")
    ds = load_dataset("naver-clova-ix/cord-v2")
    out_dir = data_dir / "cord"
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(exist_ok=True)

    formatted = []
    for split in ["train", "val", "test"]:
        if split not in ds:
            continue
        for item in ds[split]:
            img = item.get("image")
            if img is None:
                continue
            img_name = f"{split}_{len(formatted):06d}.png"
            img_path = images_dir / img_name
            if not img_path.exists():
                img.save(str(img_path))
            formatted.append({
                "image": f"cord/images/{img_name}",
                "ground_truth": item.get("ground_truth", ""),
            })

    out_path = out_dir / "cord.json"
    with open(out_path, "w") as f:
        json.dump(formatted, f, indent=2)
    logger.info(f"CORD: {len(formatted)} samples -> {out_path}")
    return out_dir


def download_sroie(data_dir: Path) -> Path:
    """Download SROIE from HuggingFace datasets."""
    from datasets import load_dataset
    logger.info("Downloading SROIE...")
    ds = load_dataset("rajistics/sroie")
    out_dir = data_dir / "sroie"
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(exist_ok=True)

    formatted = []
    for split in ["train", "test"]:
        if split not in ds:
            continue
        for item in ds[split]:
            img = item.get("image")
            if img is None:
                continue
            img_name = f"{split}_{len(formatted):06d}.png"
            img_path = images_dir / img_name
            if not img_path.exists():
                img.save(str(img_path))
            formatted.append({
                "image": f"sroie/images/{img_name}",
                "ground_truth": item.get("ground_truth", ""),
            })

    out_path = out_dir / "sroie.json"
    with open(out_path, "w") as f:
        json.dump(formatted, f, indent=2)
    logger.info(f"SROIE: {len(formatted)} samples -> {out_path}")
    return out_dir


def download_ocrbench(data_dir: Path) -> Path:
    """Download OCRBench from HuggingFace datasets."""
    from datasets import load_dataset
    logger.info("Downloading OCRBench...")
    ds = load_dataset("echo840/OCRBench", split="test")
    out_dir = data_dir / "ocrbench"
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(exist_ok=True)

    formatted = []
    for item in ds:
        img = item.get("image")
        if img is None:
            continue
        img_name = f"{len(formatted):06d}.png"
        img_path = images_dir / img_name
        if not img_path.exists():
            img.save(str(img_path))
        answer = item.get("answer", "")
        if isinstance(answer, str):
            answers = [answer]
        elif isinstance(answer, list):
            answers = answer
        else:
            answers = [str(answer)]
        formatted.append({
            "image": f"ocrbench/images/{img_name}",
            "question": item.get("question", ""),
            "answers": answers,
            "type": item.get("question_type", ""),
        })

    out_path = out_dir / "ocrbench.json"
    with open(out_path, "w") as f:
        json.dump(formatted, f, indent=2)
    logger.info(f"OCRBench: {len(formatted)} samples -> {out_path}")
    return out_dir


BENCHMARK_DOWNLOADS = {
    "docvqa": download_docvqa,
    "funsd": download_funsd,
    "cord": download_cord,
    "sroie": download_sroie,
    "ocrbench": download_ocrbench,
}


def main():
    parser = argparse.ArgumentParser(description="Download benchmark datasets")
    parser.add_argument("--data-dir", type=str, default="evaluation/data", help="Data directory")
    parser.add_argument("--benchmarks", type=str, nargs="+", default=list(BENCHMARK_DOWNLOADS.keys()),
                        help="Benchmarks to download")
    parser.add_argument("--force", action="store_true", help="Re-download even if exists")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    for name in args.benchmarks:
        if name not in BENCHMARK_DOWNLOADS:
            logger.warning(f"Unknown benchmark: {name}. Available: {list(BENCHMARK_DOWNLOADS.keys())}")
            continue

        benchmark_dir = data_dir / name
        if benchmark_dir.exists() and not args.force:
            logger.info(f"{name} already exists, skipping. Use --force to re-download.")
            continue

        try:
            BENCHMARK_DOWNLOADS[name](data_dir)
        except Exception as e:
            logger.error(f"Failed to download {name}: {e}")
            logger.info(f"Try: pip install datasets huggingface_hub")

    logger.info("Done.")


if __name__ == "__main__":
    main()
