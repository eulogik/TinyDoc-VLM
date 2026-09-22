#!/usr/bin/env python3
"""
Rebuild the TinyDoc-VLM training dataset end-to-end:

  1. Regenerate synthetic markdown documents (fixed generator, seed 42)
  2. Merge real public datasets (data/datasets/) + legacy benchmarks
  3. Write data/training/manifest.jsonl + stats.json
  4. Validate: structural checks on the FULL manifest, QA-consistency on the
     synthetic subset (real QA answers are not expected verbatim in any text)
  5. If validation passes: package training.tar.gz in the kaggle layout
     {manifest.jsonl, data/training/synthetic/, data/training/datasets/}

Fails hard (non-zero exit) on any validation error so an outer `&&` chain
skips the HF upload.

Usage:
    python3 data/rebuild_training_data.py [--num-docs 50000]
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
import tarfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.synthetic.markdown_dataset import generate_markdown_documents  # noqa: E402
from data.real_benchmarks import (  # noqa: E402
    load_cord, load_docmatix, load_docvqa, load_funsd, load_ocrbench,
    load_sroie,
)


def log(msg):
    print(msg, flush=True)
    logging.getLogger("rebuild").info(msg)


def normalize_paths(entries: list, out_dir: Path) -> list:
    """Rewrite absolute/Mac image paths to Kaggle-safe repo-relative paths
    (data/training/...) and copy legacy benchmark images into the bundle."""
    datasets_dir = out_dir / "datasets"
    fixed = []
    for e in entries:
        src = e.get("source", "")
        p = Path(e["image_path"])
        suffix = p.as_posix()
        if "/synthetic/images/" in suffix:
            suffix = suffix.split("/synthetic/images/", 1)[1]
            e["image_path"] = f"data/training/synthetic/images/{suffix}"
        elif "/datasets/" in suffix:
            suffix = suffix.split("/datasets/", 1)[1]
            dst = datasets_dir / suffix
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(p, dst)
            e["image_path"] = f"data/training/datasets/{suffix}"
        elif "/evaluation/data/" in suffix:
            fname = Path(suffix.split("/evaluation/data/", 1)[1]).name
            dst = datasets_dir / src / "images" / fname
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(p, dst)
            e["image_path"] = f"data/training/datasets/{src}/images/{fname}"
        else:
            raise SystemExit(f"cannot normalize image_path: {e['image_path']!r}")
        fixed.append(e)
    return fixed


def shard_images(out_dir: Path, entries: list) -> list:
    """Move files in image dirs with >10000 entries into numbered 1000-file
    subdirs (HF git repos cap 10k files per directory) and rewrite manifest
    image_paths accordingly. Idempotent: already-sharded dirs are untouched."""
    image_dirs = [out_dir / "synthetic" / "images"]
    datasets_dir = out_dir / "datasets"
    if datasets_dir.exists():
        for sub in datasets_dir.iterdir():
            if (sub / "images").is_dir():
                image_dirs.append(sub / "images")
    renames = {}
    for d in image_dirs:
        files = sorted(f for f in d.iterdir() if f.is_file())
        if len(files) <= 10000:
            continue
        log(f"sharding {d} ({len(files)} files)")
        for i in range(0, len(files), 1000):
            shard_dir = d / f"{i // 1000:02d}"
            shard_dir.mkdir(exist_ok=True)
            for f in files[i:i + 1000]:
                dst = shard_dir / f.name
                if not dst.exists():
                    f.rename(dst)
                renames[str(f)] = str(dst)
    for e in entries:
        ip = e.get("image_path")
        if ip in renames:
            e["image_path"] = renames[ip]
    return entries


def run_validator(manifest: Path, extra_args: list) -> int:
    log(f"validator: {' '.join(extra_args)} on {manifest}")
    r = subprocess.run(
        [sys.executable, "data/validate_dataset.py", str(manifest)]
        + extra_args,
        cwd=Path(__file__).resolve().parent.parent,
    )
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-docs", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", type=str, default="data/training")
    ap.add_argument("--skip-synthetic", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parent.parent

    # 1. Synthetic (fixed generator)
    syn = []
    syn_manifest = out_dir / "manifest_synthetic.jsonl"
    if args.skip_synthetic:
        if syn_manifest.exists():
            syn = [json.loads(l) for l in syn_manifest.open() if l.strip()]
            log(f"synthetic reused from {syn_manifest}: {len(syn)} pairs")
        else:
            log("--skip-synthetic but no manifest_synthetic.jsonl found")
            sys.exit(1)
    else:
        old = out_dir / "synthetic"
        if old.exists():
            log(f"removing old synthetic dir ({sum(f.stat().st_size for f in old.rglob('*') if f.is_file()) / 1e6:.0f} MB)")
            shutil.rmtree(old)
        log(f"generating {args.num_docs} synthetic documents (seed {args.seed}) ...")
        syn = generate_markdown_documents(
            num_docs=args.num_docs,
            output_dir=out_dir / "synthetic",
            seed=args.seed,
        )
        log(f"synthetic: {len(syn)} pairs")

    # 2. Real data (public datasets + legacy benchmarks)
    real = []
    real += load_ocrbench(Path("evaluation/data"))
    real += load_funsd(Path("evaluation/data"))
    real += load_cord(Path("evaluation/data"))
    real += load_docvqa()
    real += load_sroie()
    real += load_docmatix()
    log(f"real: {len(real)} pairs")
    by_real = Counter(p.get("source") for p in real)
    log(f"real by_source: {dict(by_real)}")

    # 3. Normalize image paths (Mac-absolute -> repo-relative) + copy legacy
    merged = normalize_paths(syn + real, out_dir)
    merged = shard_images(out_dir, merged)
    log(f"TOTAL pairs: {len(merged)}")
    manifest = out_dir / "manifest.jsonl"
    with open(manifest, "w") as f:
        for e in merged:
            f.write(json.dumps(e) + "\n")
    by_source = Counter(e.get("source") for e in merged)
    stats = {
        "total_pairs": len(merged),
        "synthetic_pairs": len(syn),
        "real_pairs": len(real),
        "by_source": dict(by_source),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))

    # 4. Validation gates
    with open(syn_manifest, "w") as f:
        for e in syn:
            f.write(json.dumps(e) + "\n")

    rc = run_validator(manifest, [])
    rc2 = run_validator(syn_manifest, ["--qa-consistency"])
    if rc != 0 or rc2 != 0:
        log("VALIDATION FAILED - not packaging (see errors above)")
        sys.exit(1)
    log("VALIDATION PASSED")

    # 5. Package training.tar.gz (kaggle layout)
    bundle = out_dir / "training.tar.gz"
    if bundle.exists():
        bundle.unlink()
    log("packaging training.tar.gz ...")
    with tarfile.open(bundle, "w:gz", compresslevel=6) as tf:
        # Bundle is extracted INTO REPO/data/training on Kaggle (cwd=REPO),
        # so tar-root layout must be {manifest.jsonl, synthetic/, datasets/}
        # to resolve image_paths like data/training/synthetic/... from REPO.
        tf.add(str(out_dir / "manifest.jsonl"), arcname="manifest.jsonl")
        tf.add(str(out_dir / "synthetic"), arcname="synthetic")
        datasets = out_dir / "datasets"
        if datasets.exists():
            tf.add(str(datasets), arcname="datasets")
    size_mb = bundle.stat().st_size / 1e6
    log(f"DONE: {bundle} ({size_mb:.0f} MB) | pairs: {len(merged)}")
    log(f"next: HF_TOKEN=$(cat training/.hf_token) python3 training/upload_data_to_hf.py --data-dir {out_dir}")


if __name__ == "__main__":
    main()
