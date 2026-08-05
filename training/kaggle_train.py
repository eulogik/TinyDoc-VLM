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
import select
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_URL = "https://github.com/eulogik/TinyDoc-VLM"
BRANCH = "main"
WORK = Path("/kaggle/working")
REPO = WORK / "tinydoc-vlm"
OUT_DIR = "checkpoints/full768"
SYNC_THROTTLE_SEC = 12 * 60
LOG_REPO = "eulogik/TinyDoc-VLM-runtime"
LOG_DIR = WORK / "logs"

LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.FileHandler(str(LOG_DIR / "kaggle_train.log")),
    ],
)
logger = logging.getLogger("kaggle_train")


class LogUploader:
    """Ships logs/* to the HF runtime repo so run progress and failures are
    inspectable from anywhere (papermill's log stream can swallow output)."""

    def __init__(self, repo_id, logs_dir, token, interval=45):
        self.repo_id = repo_id
        self.logs_dir = Path(logs_dir)
        self.token = token
        self.interval = interval
        self.stop = threading.Event()
        self._mtime = {}

    def files(self):
        return [f for f in sorted(self.logs_dir.iterdir())
                if f.suffix in (".log", ".txt")]

    def upload_changed(self):
        from huggingface_hub import upload_file
        for f in self.files():
            mtime = f.stat().st_mtime
            if self._mtime.get(f.name) == mtime:
                continue
            try:
                upload_file(path_or_fileobj=str(f),
                            path_in_repo=f"logs/{f.name}",
                            repo_id=self.repo_id, token=self.token)
                self._mtime[f.name] = mtime
            except Exception as e:
                logger.warning("log upload failed (%s): %s", f.name, e)

    def upload_file(self, path_in_repo, local_path):
        from huggingface_hub import upload_file
        try:
            upload_file(path_or_fileobj=str(local_path),
                        path_in_repo=path_in_repo,
                        repo_id=self.repo_id, token=self.token)
        except Exception as e:
            logger.warning("log upload failed (%s): %s", path_in_repo, e)

    def loop(self):
        while not self.stop.is_set():
            try:
                self.upload_changed()
            except Exception:
                pass
            self.stop.wait(self.interval)

    def final(self):
        self.stop.set()
        try:
            self.upload_changed()
        except Exception:
            pass


def run(cmd, cwd=None, logfile=None):
    """Run a subprocess, tee output to the papermill log AND a file.

    Chunked read: tqdm progress bars write \r (no newline), which would
    deadlock a line-oriented reader once the pipe buffer fills.
    select() with a timeout: if the child exits but its stdout pipe is
    still held open by a lingering grandchild, a blocking read() would
    hang the whole run (and the Kaggle session) forever.
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
    idle = 0
    last_output = time.time()
    while True:
        if select.select([p.stdout], [], [], 30)[0]:
            chunk = p.stdout.read(4096)
            if not chunk:
                break
            idle = 0
            last_output = time.time()
            sys.stdout.write(chunk)
            sys.stdout.flush()
            if out_f:
                out_f.write(chunk)
                out_f.flush()
        elif p.poll() is not None:
            idle += 1
            if idle >= 2:
                break
        else:
            idle = 0
            # Stall watchdog: model loading legitimately takes a few minutes,
            # but hours of silence means the child is hung (e.g. a network
            # call without timeout). Kill it instead of burning the session.
            # faulthandler in full_train.py dumps the stack to this log first.
            if time.time() - last_output > 600:
                logger.error("STALLED: no output for 10 min; killing child (see faulthandler stack above)")
                p.kill()
                break
    if p.poll() is None:
        logger.warning("child alive after stdout EOF; terminating")
        p.terminate()
    if out_f:
        out_f.close()
    return p.wait()


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
            path_in_repo="latest",  # keep the repo structured: latest/step.txt etc.
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
        cap = None
    if need_cu118:
        logger.warning("P100/old GPU detected; installing torch cu118 (sm_50+).")
        pip_install(["torch==2.6.0+cu118", "torchvision==0.21.0+cu118"],
                    extra_index="https://download.pytorch.org/whl/cu118", force=True)
        # torchaudio is only a soft dependency of transformers
        # (is_torchaudio_available-guarded) and its cu12x-built .so crashes
        # after the torch downgrade with a missing aoti_torch_abi_version
        # symbol. Removing it lets transformers import cleanly.
        run([sys.executable, "-m", "pip", "uninstall", "-y", "torchaudio"])
    env_script = (
        "import sys, torch\n"
        "import importlib.metadata as md\n"
        "print('python', sys.version.split()[0])\n"
        "print('torch', torch.__version__)\n"
        "print('cuda_available', torch.cuda.is_available())\n"
        "print('device', torch.cuda.get_device_name(0))\n"
        "print('cap', torch.cuda.get_device_capability(0))\n"
        "print('torchvision', md.version('torchvision'))\n"
        "try:\n"
        "    print('torchaudio', md.version('torchaudio'))\n"
        "except Exception:\n"
        "    print('torchaudio', 'NOT INSTALLED')\n"
    )
    rc = run([sys.executable, "-c", env_script],
             logfile=str(LOG_DIR / "env.txt"))
    if rc != 0:
        logger.error("torch import failed (rc=%s); aborting.", rc)
        sys.exit(rc)
    # The init checkpoint is saved in bf16 (built on a dev box). Under the
    # fp16 AMP path in full_train.py, bf16 weights + fp32 vision features
    # crash in the visual-token index-put (modeling.py:126). Convert init
    # to fp32 unconditionally; convert_init_dtype.py skips if already fp32.
    convert_dtype = "float32"
    if convert_dtype:
        logger.info("init checkpoint will be converted to %s before training.", convert_dtype)
    pip_install(["transformers==5.12.1", "sentencepiece", "tokenizers", "pillow", "numpy",
                 "pandas", "tqdm", "pyyaml", "einops", "faker", "jinja2", "pydantic",
                 "datasets", "accelerate", "huggingface_hub"])
    # transformers version must match the version the init checkpoint was
    # saved with (5.12.1): newer versions restructured SiglipVisionModel
    # (extra vision_model nesting) and silently random-init the vision tower.
    from huggingface_hub import hf_hub_download
    try:
        import transformers as _tf
        _tf_ver = _tf.__version__.split("+")[0]
        if _tf_ver != "5.12.1":
            logger.error("transformers %s installed but 5.12.1 required; reinstall failed?", _tf_ver)
            sys.exit(1)
    except Exception as e:
        logger.error("transformers check failed: %s", e)
        sys.exit(1)

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
        if convert_dtype:
            rc = run([sys.executable, "training/convert_init_dtype.py",
                      "--model", str(init_dir), "--dtype", convert_dtype],
                     logfile=str(LOG_DIR / "convert.log"))
            if rc != 0:
                logger.error("init_768 dtype conversion failed; aborting.")
                sys.exit(rc)

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
        # Legacy layout: older syncs uploaded the folder CONTENTS flat
        # (step.txt at full768 root). Normalize into latest/ so full_train's
        # resume lookup works.
        flat_step = ckpt_root / "full768" / "step.txt"
        if flat_step.exists() and not (latest / "step.txt").exists():
            logger.info("Legacy flat checkpoint layout detected; moving into latest/")
            latest.mkdir(parents=True, exist_ok=True)
            for f in (ckpt_root / "full768").iterdir():
                if f.is_file() and f.suffix in (".safetensors", ".bin", ".json", ".txt"):
                    shutil.move(str(f), str(latest / f.name))
        if latest.exists() and (latest / "step.txt").exists():
            step = (latest / "step.txt").read_text().strip()
            logger.info("Resuming from step %s", step)
    except Exception as e:
        logger.warning("No checkpoint found in %s (fresh start): %s", args.ckpt_repo, e)

    # 6. Train with live sync
    log_up = LogUploader(LOG_REPO, LOG_DIR, hf_token)
    threading.Thread(target=log_up.loop, daemon=True).start()
    syncer = CkptSyncer(args.ckpt_repo, latest) if not args.no_sync else None
    if syncer:
        t = threading.Thread(target=syncer.loop, daemon=True)
        t.start()

    train_log = LOG_DIR / "full_train.log"
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
    log_up.final()
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


def record_failure(hf_token):
    """Write + upload a consolidated failure report (runs on any non-zero exit)."""
    report = ["=== run failed ===\n"]
    for name in ("kaggle_train.log", "env.txt", "convert.log", "full_train.log"):
        f = LOG_DIR / name
        if f.exists():
            lines = f.read_text(errors="replace").splitlines()
            report.append(f"--- {name} (last {40 if name != 'full_train.log' else 120} of {len(lines)}) ---")
            report.extend(lines[-40 if name != "full_train.log" else 120:])
            report.append("")
    err = LOG_DIR / "last_error.txt"
    err.write_text("\n".join(report))
    try:
        LogUploader(LOG_REPO, LOG_DIR, hf_token).upload_file("logs/last_error.txt", err)
    except Exception as e:
        logger.error("could not upload failure report: %s", e)


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        if e.code not in (None, 0):
            record_failure(os.environ.get("HF_TOKEN", ""))
        raise
    except Exception:
        record_failure(os.environ.get("HF_TOKEN", ""))
        raise
