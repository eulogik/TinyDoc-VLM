"""
Combined training-dataset builder for TinyDoc-VLM (D + E).

1. Generates synthetic markdown-conversion documents (D).
2. Converts real benchmarks to training pairs (E).
3. Merges into one manifest: data/training/manifest.jsonl

Each line: {image_path, prompt, target, source, doc_type?}

Usage:
    python data/build_training_dataset.py --num-docs 50000
"""

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

sys_path = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(sys_path))

from data.synthetic.markdown_dataset import generate_markdown_documents
from data.real_benchmarks import load_all


def main():
    parser = argparse.ArgumentParser(description="Build combined TinyDoc-VLM training set")
    parser.add_argument("--num-docs", type=int, default=50000,
                        help="Number of synthetic documents to generate (target 50K+)")
    parser.add_argument("--output-dir", type=str, default="data/training")
    parser.add_argument("--data-dir", type=str, default="evaluation/data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Synthetic markdown-conversion data (D)
    syn = generate_markdown_documents(
        num_docs=args.num_docs,
        output_dir=out_dir / "synthetic",
        seed=args.seed,
    )

    # 2. Real benchmarks (E)
    real = load_all(Path(args.data_dir))

    # 3. Merge
    merged = syn + real
    manifest_path = out_dir / "manifest.jsonl"
    with open(manifest_path, "w") as f:
        for e in merged:
            f.write(json.dumps(e) + "\n")

    # Stats
    from collections import Counter
    by_source = Counter(e.get("source") for e in merged)
    by_prompt = Counter(e.get("prompt", "").split(":")[0].split(" ")[0] for e in merged)
    stats = {
        "total_pairs": len(merged),
        "synthetic_pairs": len(syn),
        "real_pairs": len(real),
        "by_source": dict(by_source),
        "samples_per_image": round(len(merged) / max(1, args.num_docs), 2),
    }
    with open(out_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"TOTAL training pairs: {len(merged)}")
    logger.info(f"  synthetic: {len(syn)} | real: {len(real)}")
    logger.info(f"  by_source: {dict(by_source)}")
    logger.info(f"Manifest: {manifest_path}")
    logger.info(f"Stats:    {out_dir / 'stats.json'}")


if __name__ == "__main__":
    main()
