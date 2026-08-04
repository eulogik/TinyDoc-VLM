#!/usr/bin/env python3
"""
Headless TinyDoc-VLM 768 retrain runner for Kaggle (T4/P100, 30h/week free).

Persistence strategy (Kaggle VMs are ephemeral; no Drive):
  * data/training/   -> HF dataset repo  (uploaded once from your Mac)
  * checkpoints/     -> HF model repo    (uploaded live during training)

Flow per run:
  1. Clone the TinyDoc-VLM repo (fresh VM every time).
  2. Download data/training from the HF dataset repo.
  3. Download the latest checkpoint from the HF model repo (resume).
  4. Run full_train.py; a background thread uploads checkpoints/full768/latest/
     to the HF model repo every time it changes (throttled ~12 min).
  5. On finish, push the final checkpoint and exit.

Usage (inside the Kaggle notebook, or any box with a GPU):
    HF_TOKEN=<token> DATA_REPO=eulogik/TinyDoc-VLM-training-data \
    CKPT_REPO=eulogik/TinyDoc-VLM-768-checkpoints \
    python training/kaggle_train.py --steps 8000
"""

import argparse
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("kaggle_train")

REPO_URL = "https://github.com/eulogik/TinyDoc-VLM"
BRANCH = "main"
WORK = Path("/kaggle/working")
REPO = WORK / "tinydoc-vlm"
OUT_DIR = "checkpoints/full768"
SYNC_THROTTLE_SEC = 12 * 60


def run(cmd, cwd=None, logfile=None):
    """Run a subprocess, tee output to the papermill log AND a file.

    Chunked read: tqdm progress bars write \r (no newline), which would
    deadlock a line-oriented reader once the pipe buffer fills.
    The logfile makes failures visible even when papermill's log stream
    swallows the child's output (as observed with long-running steps).
    """
    logger.info("$ %s", " ".join(str(c) for c in cmd))
    if logfile is not None:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
    p = subprocess.Popen(
        cmd, cwd=cwd, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    out_f = open(logfile, "w") if logfile else None
    while True:
        chunk = p.stdout.read(4096)
        if not chunk:
            break
        sys.stdout.write(chunk)
        sys.stdout.flush()
        if out_f:
            out_f.write(chunk)
            out_f.flush()
    if out_f:
        out_f.close()
    p.wait()
    return p.returncode


def pip_install(pkgs, extra_index=None, force=False):
    cmd = [sys.executable, "-m", "pip", "install", "-q"] + pkgs
    if force:
        cmd.append("--force-reinstall")
    if extra_index:
        cmd += ["--index-url", extra_index]
    rc = run(cmd)
    if rc != 0:
        logger.error("pip install FAILED (%s): %s", rc, " ".join(cmd))
        sys.exit(rc)


class CkptSyncer:
    """Watches checkpoints/full768/latest/ and uploads to the HF model repo."""

    def __init__(self, repo_id, watch_dir: Path):
        self.repo_id = repo_id
        self.watch_dir = watch_dir
        self.last_mtime = 0.0
        self.last_upload = 0.0
        self.stop = threading.Event()

    def _changed(self):
        step_file = self.watch_dir / "step.txt"
        if not step_file.exists():
            return False
        return step_file.stat().st_mtime != self.last_mtime

    def upload_once(self):
        from huggingface_hub import upload_folder
        if not (self.watch_dir / "step.txt").exists():
            logger.info("No latest/ checkpoint yet; skipping upload.")
            return
        logger.info("Uploading latest/ checkpoint to %s ...", self.repo_id)
        upload_folder(
            repo_id=self.repo_id,
            folder_path=str(self.watch_dir),
            repo_type="model",
            commit_message="latest/ checkpoint sync",
        )
        self.last_upload = time.time()
        self.last_mtime = (self.watch_dir / "step.txt").stat().st_mtime
        logger.info("Upload complete.")

    def loop(self):
        while not self.stop.is_set():
            try:
                if self._changed() and time.time() - self.last_upload > SYNC_THROTTLE_SEC:
                    self.upload_once()
            except Exception as e:
                logger.warning("Upload failed (will retry): %s", e)
            self.stop.wait(60)

    def final(self):
        self.stop.set()
        try:
            self.upload_once()
        except Exception as e:
            logger.error("Final upload failed: %s", e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-seq-length", type=int, default=512)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--save-latest-every", type=int, default=50)
    ap.add_argument("--data-repo", default=os.environ.get("DATA_REPO",
                     "eulogik/TinyDoc-VLM-training-data"))
    ap.add_argument("--ckpt-repo", default=os.environ.get("CKPT_REPO",
                     "eulogik/TinyDoc-VLM-768-checkpoints"))
    ap.add_argument("--no-sync", action="store_true",
                    help="Disable live checkpoint upload (still pushes final).")
    args = ap.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.error("HF_TOKEN env var required (set as a Kaggle secret).")
        sys.exit(1)
    # tqdm writes \r-only lines that bloat the log and can stall subprocess
    # pipes; suppress it everywhere downstream.
    os.environ["TQDM_DISABLE"] = "1"

    # 1. Clone repo
    WORK.mkdir(parents=True, exist_ok=True)
    os.chdir(WORK)  # never sit inside the dir we are about to delete
    if REPO.exists():
        run(["rm", "-rf", str(REPO)])
    run(["git", "clone", "--depth", "1", "-b", BRANCH, REPO_URL, str(REPO)])
    os.chdir(REPO)

    # 2. Deps + torch fix for old GPUs. Kaggle may provision a P100 (sm_60),
    #    whose kernels are missing from the preinstalled cu12x torch (sm_70+).
    #    cu118 wheels cover both P100 (sm_60) and T4 (sm_75).
    try:
        import torch
        cap = torch.cuda.get_device_capability(0)
        logger.info("GPU capability: %s", cap)
        need_cu118 = cap[0] < 7
    except Exception:
        need_cu118 = False
    if need_cu118:
        logger.warning("P100/old GPU detected; installing torch cu118 (sm_50+).")
        pip_install(["torch==2.6.0+cu118", "torchvision==0.21.0+cu118"],
                    extra_index="https://download.pytorch.org/whl/cu118", force=True)
        run([sys.executable, "-c",
             "import torch; print('torch', torch.__version__, 'cap', torch.cuda.get_device_capability(0))"])
    pip_install(["transformers", "sentencepiece", "tokenizers", "pillow", "numpy",
                 "pandas", "tqdm", "pyyaml", "einops", "faker", "jinja2", "pydantic",
                 "datasets", "accelerate", "huggingface_hub"])

    # 3. Download training data from HF dataset repo
    data_dir = REPO / "data/training"
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest = data_dir / "manifest.jsonl"
    if manifest.exists():
        logger.info("Manifest already present; skipping data download.")
    else:
        from huggingface_hub import snapshot_download
        logger.info("Downloading data from %s ...", args.data_repo)
        snapshot_download(
            repo_id=args.data_repo, repo_type="dataset",
            local_dir=str(data_dir), token=hf_token,
        )
        if not manifest.exists():
            logger.error("No manifest.jsonl in dataset repo %s", args.data_repo)
            sys.exit(1)

    # 4. Get the pre-built 768 model (built on a dev box; building it on the
    #    Kaggle VM OOMs). Falls back to building locally if download fails.
    ckpt_root = REPO / "checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)
    init_dir = ckpt_root / "init_768"
    if (init_dir / "config.json").exists():
        logger.info("init_768 exists; skipping.")
    else:
        from huggingface_hub import snapshot_download
        try:
            logger.info("Downloading pre-built 768 model ...")
            snapshot_download(
                repo_id="eulogik/TinyDoc-VLM-768-init", repo_type="model",
                local_dir=str(init_dir), token=hf_token,
            )
        except Exception as e:
            logger.warning("init_768 download failed (%s); building instead.", e)
            rc = run([sys.executable, "training/init_768_model.py",
                      "--base", "eulogik/TinyDoc-VLM-256M",
                      "--out", "checkpoints/init_768"])
            if rc != 0:
                logger.error("init_768 build failed; aborting.")
                sys.exit(rc)
        if not (init_dir / "config.json").exists():
            logger.error("init_768 missing; aborting.")
            sys.exit(1)

    # 5. Download latest checkpoint from HF model repo (resume)
    latest = ckpt_root / "full768" / "latest"
    if (ckpt_root / "full768" / "final").exists():
        logger.info("final checkpoint already present; nothing to do.")
        return
    from huggingface_hub import snapshot_download
    try:
        logger.info("Downloading checkpoint from %s ...", args.ckpt_repo)
        snapshot_download(
            repo_id=args.ckpt_repo, repo_type="model",
            local_dir=str(ckpt_root / "full768"), token=hf_token,
        )
        if latest.exists() and (latest / "step.txt").exists():
            step = (latest / "step.txt").read_text().strip()
            logger.info("Resuming from step %s", step)
    except Exception as e:
        logger.warning("No checkpoint found in %s (fresh start): %s", args.ckpt_repo, e)

    # 6. Train with live sync
    syncer = CkptSyncer(args.ckpt_repo, latest) if not args.no_sync else None
    if syncer:
        t = threading.Thread(target=syncer.loop, daemon=True)
        t.start()

    train_log = WORK / "full_train.log"
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
              "--save-latest-every", str(args.save_latest_every),
              "--device", "cuda", "--bf16", "--grad-checkpoint",
              "--output-dir", OUT_DIR,
              "--max-samples", "2000000"],
             logfile=str(train_log))

    if syncer:
        syncer.final()
    if rc != 0:
        logger.error("Training exited %s; latest/ checkpoint synced for next run.", rc)
        # Dump the tail of the captured log: papermill's stream sometimes
        # swallows the child's output, so this is the only reliable view.
        try:
            lines = train_log.read_text().splitlines()
            logger.error("--- full_train.log tail (%d lines) ---", len(lines))
            for l in lines[-60:]:
                logger.error("%s", l)
        except Exception as e:
            logger.error("Could not read %s: %s", train_log, e)
        sys.exit(rc)
    logger.info("DONE. Final checkpoint is in %s and in %s", OUT_DIR, args.ckpt_repo)


if __name__ == "__main__":
    main()
