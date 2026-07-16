"""
Real-benchmark training pairs for TinyDoc-VLM (Tier-2 retrain, item E).

Converts already-downloaded benchmark data into prompt->target training pairs
in the same format as the synthetic markdown data:

  - OCRBench  -> "Answer the question: ..." -> answer   (1000 VQA pairs)
  - FUNSD      -> "Extract all text:"        -> words joined (OCR)
  - CORD       -> "Extract the document as JSON:" -> gt_parse JSON (KIE)
  - SROIE      -> images only (no GT text on disk) -> skipped for training

Output manifest entries: {image_path, prompt, target, source}

Usage:
    python data/real_benchmarks.py --data-dir evaluation/data --output data/training/real.jsonl
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


def load_ocrbench(data_dir: Path) -> List[Dict]:
    path = data_dir / "ocrbench" / "ocrbench.json"
    if not path.exists():
        logger.warning(f"OCRBench not found: {path}")
        return []
    out = []
    with open(path) as f:
        data = json.load(f)
    for item in data:
        img = (data_dir / item["image"]).resolve()
        if not img.exists():
            continue
        ans = item.get("answers", [""])[0]
        out.append({
            "image_path": str(img),
            "prompt": f"Answer the question: {item.get('question','')}",
            "target": ans,
            "source": "ocrbench",
        })
    return out


def load_funsd(data_dir: Path) -> List[Dict]:
    path = data_dir / "funsd" / "funsd.json"
    if not path.exists():
        logger.warning(f"FUNSD not found: {path}")
        return []
    out = []
    with open(path) as f:
        data = json.load(f)
    for item in data:
        img = (data_dir / item["image"]).resolve()
        if not img.exists():
            continue
        words = " ".join(w for w in item.get("words", []) if isinstance(w, str))
        out.append({
            "image_path": str(img),
            "prompt": "Extract all text:",
            "target": words,
            "source": "funsd",
        })
    return out


def load_cord(data_dir: Path) -> List[Dict]:
    path = data_dir / "cord" / "cord.json"
    if not path.exists():
        logger.warning(f"CORD not found: {path}")
        return []
    out = []
    with open(path) as f:
        data = json.load(f)
    for item in data:
        img = (data_dir / item["image"]).resolve()
        if not img.exists():
            continue
        gt = item.get("ground_truth", "")
        try:
            parsed = json.loads(gt) if isinstance(gt, str) else gt
            target = json.dumps(parsed.get("gt_parse", parsed), ensure_ascii=False)
        except Exception:
            target = gt
        out.append({
            "image_path": str(img),
            "prompt": "Extract the document as JSON:",
            "target": target,
            "source": "cord",
        })
    return out


def load_all(data_dir: Path) -> List[Dict]:
    pairs = []
    pairs += load_ocrbench(data_dir)
    pairs += load_funsd(data_dir)
    pairs += load_cord(data_dir)
    logger.info(f"Loaded {len(pairs)} real-benchmark training pairs")
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Convert real benchmarks to training pairs")
    parser.add_argument("--data-dir", type=str, default="evaluation/data")
    parser.add_argument("--output", type=str, default="data/training/real.jsonl")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    pairs = load_all(Path(args.data_dir))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    logger.info(f"Wrote {len(pairs)} pairs to {out}")


if __name__ == "__main__":
    main()
