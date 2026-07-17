#!/usr/bin/env python3
"""
Headless entry point for the TinyDoc-VLM 768 retrain, designed for:

  * Google Colab CLI:  colab run --gpu T4 training/colab_train.py
  * Colab notebook:    %run training/colab_train.py
  * Any Linux/Mac box with a GPU.

It is idempotent and resumable: every stage checks for its output and skips
if present, and full_train.py resumes from the latest step_* checkpoint. All
large artifacts live under WORK, which is placed on Google Drive when mounted
so a VM restart / Colab disconnect does NOT lose progress.

Usage:
    python training/colab_train.py \
        --steps 8000 --batch-size 8 --grad-accum 1 --max-seq-length 512
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("colab_train")

REPO_URL = "https://github.com/eulogik/TinyDoc-VLM"
BRANCH = "main"


def run(cmd, cwd=None, stream=True):
    logger.info("$ %s", " ".join(str(c) for c in cmd))
    if stream:
        p = subprocess.Popen(
            cmd, cwd=cwd, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        for line in p.stdout:
            print(line, end="")
        p.wait()
        return p.returncode
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.stdout:
        logger.info(r.stdout[-3000:])
    if r.returncode != 0 and r.stderr:
        logger.error(r.stderr[-3000:])
    return r.returncode


def mount_drive():
    """Mount Google Drive if available. Returns the WORK base dir."""
    try:
        from google.colab import drive  # type: ignore
        drive.mount("/content/drive")
        work = "/content/drive/MyDrive/tinydoc-vlm"
        logger.info("Google Drive mounted; WORK=%s (survives restarts)", work)
    except Exception as e:  # not in Colab, or auth skipped
        logger.warning("Drive mount failed (%s); using local dir (wiped on restart)", e)
        work = "/content/tinydoc-vlm" if os.path.exists("/content") else str(Path.cwd())
    Path(work).mkdir(parents=True, exist_ok=True)
    return work


def clone_repo(work):
    repo = Path(work) / "tinydoc-vlm"
    if not (repo / "data/synthetic/markdown_dataset.py").exists():
        if repo.exists():
            import shutil
            shutil.rmtree(repo)
        logger.info("Cloning %s@%s ...", REPO_URL, BRANCH)
        run(["git", "clone", "--depth", "1", "-b", BRANCH, REPO_URL, str(repo)])
    else:
        # Refresh in case fixes landed; resume is safe because outputs are guarded.
        run(["git", "-C", str(repo), "pull", "--ff-only"])
    return repo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-seq-length", type=int, default=512)
    ap.add_argument("--num-docs", type=int, default=50000)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--base", default="eulogik/TinyDoc-VLM-256M")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    work = mount_drive()
    repo = clone_repo(work)
    os.chdir(repo)

    # 1. Install deps (idempotent; pip skips already-installed)
    run([sys.executable, "-m", "pip", "install", "-q",
         "torch", "torchvision", "--index-url",
         "https://download.pytorch.org/whl/cu124"], cwd=str(repo))
    run([sys.executable, "-m", "pip", "install", "-q",
         "transformers", "sentencepiece", "tokenizers", "pillow", "numpy",
         "pandas", "tqdm", "pyyaml", "einops", "faker", "jinja2", "pydantic",
         "datasets", "accelerate"], cwd=str(repo))

    # 2. Data generation (skips if manifest already has enough pairs)
    manifest = repo / "data/training/manifest.jsonl"
    need_data = True
    if manifest.exists():
        try:
            n = sum(1 for _ in open(manifest))
            need_data = n < 10000
        except Exception:
            need_data = True
    if need_data:
        run([sys.executable, "evaluation/download_benchmarks.py",
             "--data-dir", "evaluation/data",
             "--benchmarks", "ocrbench", "funsd", "cord"], cwd=str(repo))
        run([sys.executable, "data/build_training_dataset.py",
             "--num-docs", str(args.num_docs),
             "--output-dir", "data/training",
             "--data-dir", "evaluation/data"], cwd=str(repo))
    else:
        logger.info("Manifest exists — skipping data generation.")

    # 3. Init 768 model (skips if already present)
    init_dir = repo / "checkpoints/init_768"
    if (init_dir / "config.json").exists():
        logger.info("init_768 exists — skipping.")
    else:
        rc = run([sys.executable, "training/init_768_model.py",
                  "--base", args.base, "--out", "checkpoints/init_768"],
                 cwd=str(repo))
        if rc != 0:
            logger.error("init_768 failed; aborting.")
            return

    # 4. Full fine-tune (resumes from latest step_* automatically)
    out = repo / "checkpoints/full768"
    final = out / "final"
    if final.exists():
        logger.info("Final checkpoint exists — training already complete.")
    else:
        rc = run([sys.executable, "training/full_train.py",
                  "--model-id", "checkpoints/init_768",
                  "--manifest", "data/training/manifest.jsonl",
                  "--steps", str(args.steps),
                  "--batch-size", str(args.batch_size),
                  "--grad-accum", str(args.grad_accum),
                  "--warmup", str(args.warmup),
                  "--lr", str(args.lr),
                  "--max-seq-length", str(args.max_seq_length),
                  "--save-every", str(args.save_every),
                  "--device", args.device,
                  "--bf16", "--grad-checkpoint",
                  "--output-dir", "checkpoints/full768",
                  "--max-samples", "2000000"], cwd=str(repo))
        if rc != 0:
            logger.error("Training exited with %s (re-run to resume).", rc)

    # 5. Sync final checkpoint to Drive-backed WORK for safekeeping
    if final.exists():
        dst = Path(work) / "checkpoints/full768_final"
        if dst.exists():
            import shutil
            shutil.rmtree(dst)
        shutil.copytree(final, dst)
        logger.info("Synced final checkpoint to %s", dst)
        logger.info("DONE. Push with: training/push_to_hf.py (set HF_TOKEN)")


if __name__ == "__main__":
    main()
