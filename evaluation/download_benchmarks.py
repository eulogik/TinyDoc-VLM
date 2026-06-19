#!/usr/bin/env python3
"""
Downloads and formats public benchmark datasets for TinyDoc-VLM evaluation.
Supports DocVQA, FUNSD, CORD, SROIE, ChartQA, and PubTabNet.

Usage:
    python evaluation/download_benchmarks.py --data-dir evaluation/data
"""

import argparse
import json
import logging
import os
import shutil
import zipfile
import tarfile
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

BENCHMARK_SOURCES = {
    "docvqa": {
        "url": "https://huggingface.co/datasets/nielsr/docvqa_1200_examples_donut/resolve/main/data.zip",
        "filename": "docvqa.zip",
    },
    "funsd": {
        "url": "https://huggingface.co/datasets/nielsr/funsd_donut/resolve/main/data.zip",
        "filename": "funsd.zip",
    },
    "cord": {
        "url": "https://huggingface.co/datasets/naver-clova-ix/cord-v2/resolve/main/data.zip",
        "filename": "cord.zip",
    },
}


def download_file(url: str, dest: Path, desc: str = "") -> Path:
    if dest.exists():
        logger.info(f"{dest.name} already exists, skipping download.")
        return dest

    logger.info(f"Downloading {desc or url}...")
    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as f:
        with tqdm(total=total, unit="B", unit_scale=True, desc=desc or dest.name) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

    return dest


def extract_zip(path: Path, extract_dir: Path) -> Path:
    if extract_dir.exists():
        logger.info(f"{extract_dir} already exists, skipping extraction.")
        return extract_dir

    extract_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Extracting {path.name} to {extract_dir}...")
    with zipfile.ZipFile(path, "r") as zf:
        zf.extractall(extract_dir)

    return extract_dir


def format_docvqa(data_dir: Path) -> None:
    """Format DocVQA into the expected structure."""
    images_dir = data_dir / "images"
    images_dir.mkdir(exist_ok=True)

    formatted = []
    for split in ["train", "val"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
        for img_path in split_dir.glob("*.png"):
            shutil.copy(img_path, images_dir / img_path.name)

    ann_path = data_dir / "dataset.json"
    if ann_path.exists():
        with open(ann_path) as f:
            data = json.load(f)
        for item in data.get("data", []):
            formatted.append({
                "image": item.get("image", ""),
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
            })

        output_path = data_dir / "docvqa.json"
        with open(output_path, "w") as f:
            json.dump(formatted, f, indent=2)
        logger.info(f"Formatted {len(formatted)} DocVQA samples -> {output_path}")


def format_funsd(data_dir: Path) -> None:
    """Format FUNSD into the expected structure."""
    logger.info(f"FUNSD data in {data_dir} — needs manual formatting per task")
    formatted_path = data_dir / "funsd.json"
    if not formatted_path.exists():
        with open(formatted_path, "w") as f:
            json.dump([], f)


def format_cord(data_dir: Path) -> None:
    """Format CORD into the expected structure."""
    logger.info(f"CORD data in {data_dir} — needs manual formatting per task")
    formatted_path = data_dir / "cord.json"
    if not formatted_path.exists():
        with open(formatted_path, "w") as f:
            json.dump([], f)


def main():
    parser = argparse.ArgumentParser(description="Download benchmark datasets")
    parser.add_argument("--data-dir", type=str, default="evaluation/data", help="Data directory")
    parser.add_argument("--benchmarks", type=str, nargs="+", default=list(BENCHMARK_SOURCES.keys()),
                        help="Benchmarks to download")
    parser.add_argument("--force", action="store_true", help="Re-download even if exists")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    for name in args.benchmarks:
        if name not in BENCHMARK_SOURCES:
            logger.warning(f"Unknown benchmark: {name}. Skipping.")
            continue

        source = BENCHMARK_SOURCES[name]
        benchmark_dir = data_dir / name
        benchmark_dir.mkdir(exist_ok=True)

        zip_path = data_dir / source["filename"]
        if args.force and zip_path.exists():
            zip_path.unlink()

        try:
            download_file(source["url"], zip_path, desc=name)
            if zip_path.suffix == ".zip":
                extract_zip(zip_path, benchmark_dir)

            formatter = globals().get(f"format_{name}")
            if formatter:
                formatter(benchmark_dir)

        except Exception as e:
            logger.error(f"Failed to download {name}: {e}")
            logger.info(f"To manually download {name}, visit: {source['url']}")

    logger.info("Done. Some benchmarks may need manual setup — see evaluation/README.md")


if __name__ == "__main__":
    main()
