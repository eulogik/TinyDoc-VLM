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
import json
import logging
import os
import select
import shutil
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


def _extract_bundle(bundle: Path, data_dir: Path, manifest: Path):
    """Extract training.tar.gz (repo-root layout: synthetic/, manifest.jsonl)
    into data_dir, then sanity-check the manifest appears."""
    import tarfile
    import sys as _sys
    start = time.time()
    with tarfile.open(bundle, "r:gz") as tf:
        kwargs = {"filter": "data"} if _sys.version_info >= (3, 12) else {}
        tf.extractall(str(data_dir), **kwargs)
    logger.info("Extracted %s in %.0fs (%s)", bundle.name,
                time.time() - start, "OK" if manifest.exists() else "MISSING manifest")


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
    """Watches checkpoints/full768/latest/ and uploads to the HF model repo.

    Never overwrites a step already on the hub that is >= the local step, so
    Colab and Kaggle can train the same repo in parallel without regressing
    each other's progress (each worker just advances the hub).
    """

    def __init__(self, repo_id, watch_dir: Path, token: str = ""):
        self.repo_id = repo_id
        self.watch_dir = watch_dir
        self.token = token or None
        self.last_mtime = 0.0
        self.last_upload = 0.0
        self.stop = threading.Event()

    def _changed(self):
        step_file = self.watch_dir / "step.txt"
        if not step_file.exists():
            return False
        return step_file.stat().st_mtime != self.last_mtime

    def _remote_step(self):
        try:
            from huggingface_hub import hf_hub_download
            p = hf_hub_download(self.repo_id, "latest/step.txt", token=self.token)
            return int(Path(p).read_text().strip())
        except Exception:
            return -1

    def upload_once(self):
        from huggingface_hub import upload_folder
        if not (self.watch_dir / "step.txt").exists():
            logger.info("No latest/ checkpoint yet; skipping upload.")
            return
        local_step = int((self.watch_dir / "step.txt").read_text().strip())
        remote_step = self._remote_step()
        if local_step <= remote_step:
            logger.info("Hub already at step %s; local %s skipped.", remote_step, local_step)
            return
        logger.info("Uploading latest/ checkpoint (step %s) to %s ...", local_step, self.repo_id)
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

    # 0. Ship logs from the very start (clone/pip/downloads included), so
    #    progress is visible in the HF runtime repo even if papermill stalls.
    log_up = LogUploader(LOG_REPO, LOG_DIR, hf_token)
    threading.Thread(target=log_up.loop, daemon=True).start()
    logger.info("LogUploader started (repo=%s)", LOG_REPO)

    # 1. Clone repo
    WORK.mkdir(parents=True, exist_ok=True)
    os.chdir(WORK)  # never sit inside the dir we are about to delete
    if REPO.exists():
        run(["rm", "-rf", str(REPO)])
    run(["git", "clone", "--depth", "1", "-b", BRANCH, REPO_URL, str(REPO)])
    os.chdir(REPO)

    # 2. Deps + torch pin. Kaggle may provision a P100 (sm_60), whose kernels
    #    are missing from the preinstalled cu12x torch (sm_70+). cu118 wheels
    #    cover both P100 (sm_60) and T4 (sm_75). Separately, torch 2.10.0+cu128
    #    (the image default) native-SIGSEGV'd FOUR times on T4 today across
    #    every config (DDP, grad-ckpt, single-GPU), while Colab's proven stack
    #    (2.6.0+cu124) and every P100 run (2.6.0+cu118) never crashed. Pin the
    #    proven stack unless KAGGLE_NO_TORCH_PIN=1.
    try:
        import torch
        cap = torch.cuda.get_device_capability(0)
        logger.info("GPU capability: %s", cap)
        torch_ver = torch.__version__.split("+")[0]
        need_reinstall = False
        reason = ""
        if cap[0] < 7:
            need_reinstall = True
            reason = "P100/old GPU: preinstalled cu12x torch lacks sm_50+ kernels"
        elif torch_ver != "2.6.0" and os.environ.get("KAGGLE_NO_TORCH_PIN", "0") != "1":
            need_reinstall = True
            reason = (f"T4: torch {torch_ver} native-SIGSEGV'd 4x today; "
                      "pinning Colab-proven 2.6.0")
    except Exception:
        need_reinstall = False
        cap = None
    if need_reinstall:
        extra = "cu118" if (cap or (0,))[0] < 7 else "cu124"
        logger.warning("%s; installing torch==2.6.0+%s ...", reason, extra)
        pip_install(["torch==2.6.0+" + extra, "torchvision==0.21.0+" + extra],
                    extra_index=f"https://download.pytorch.org/whl/{extra}",
                    force=True)
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
    # fp16 AMP path in full_train.py, mixed dtypes crash in the visual-token
    # index-put (modeling.py:126, now cast-safe). Convert init to fp16: half
    # the memory traffic of fp32 on the P100 and natively supported (the
    # P100 has no bf16 kernels). convert_init_dtype.py skips if already fp16.
    convert_dtype = "float16"
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

    # 3. Download training data from HF dataset repo. Fetch a single tar.gz
    #    bundle (51k+ small PNGs through snapshot_download take hours due to
    #    per-file request overhead; one tarball downloads in ~a minute).
    data_dir = REPO / "data/training"
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest = data_dir / "manifest.jsonl"
    bundle = data_dir / "training.tar.gz"
    if manifest.exists():
        logger.info("Manifest already present; skipping data download.")
    elif bundle.exists():
        logger.info("training.tar.gz present; extracting ...")
        _extract_bundle(bundle, data_dir, manifest)
    else:
        from huggingface_hub import hf_hub_download
        logger.info("Downloading training.tar.gz from %s ...", args.data_repo)
        start = time.time()
        hf_hub_download(
            repo_id=args.data_repo, repo_type="dataset",
            filename="training.tar.gz", local_dir=str(data_dir),
            token=hf_token,
        )
        bundle = data_dir / "training.tar.gz"
        logger.info("tar.gz downloaded in %.0fs (%.1f MiB); extracting ...",
                    time.time() - start, bundle.stat().st_size / 1e6)
        _extract_bundle(bundle, data_dir, manifest)
        bundle.unlink(missing_ok=True)
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
        # NOTE: no dtype-transition guard here anymore. Training always runs
        # from fp32 master weights (AdamW-state NaN fix), so fp16/fp32/bf16
        # checkpoints all resume identically (params are promoted to fp32 at
        # load). Checkpoints are now saved fp32; never discard them.
    except Exception as e:
        logger.warning("No checkpoint found in %s (fresh start): %s", args.ckpt_repo, e)

    # 6. Train with live sync
    syncer = CkptSyncer(args.ckpt_repo, latest, token=hf_token) if not args.no_sync else None
    if syncer:
        t = threading.Thread(target=syncer.loop, daemon=True)
        t.start()

    # ---- Multi-GPU (T4 x2): DDP on torchrun, else single-GPU ----
    # NOTE: DDP/NCCL has SIGSEGV'd on Kaggle T4 x2 three times today (always
    # rank 1, 20-30 min in, regardless of grad-ckpt), while single-GPU has
    # never crashed. So single-GPU is now the default and DDP is opt-in via
    # KAGGLE_DDP=1 (still selectable for experiments).
    gpu_count = torch.cuda.device_count()
    use_ddp = gpu_count >= 2 and os.environ.get("KAGGLE_DDP", "0") == "1"
    if use_ddp:
        logger.info("Multi-GPU (%d GPUs) detected -> DDP via torchrun "
                    "(single-GPU kept as last-resort fallback)", gpu_count)
    else:
        logger.info("Single-GPU mode (%d GPU(s)); DDP opt-in via KAGGLE_DDP=1",
                    max(gpu_count, 1))

    train_log = LOG_DIR / "full_train.log"
    # Memory plan: fp32 master weights (NaN fix) + fp16 autocast needs
    # grad-checkpointing to fit the 14.5 GiB T4 with 5-tile 768px pages
    # (Colab's proven recipe). If that fails (OOM/SIGSEGV), fall back to
    # batch-2 without grad-checkpointing, then batch-1, so the kernel
    # always keeps training (every failure returns nonzero -> retry here).
    attempts = [
        {"grad_ckpt": True, "batch": args.batch_size, "ddp": use_ddp},
        {"grad_ckpt": False,
         "batch": (args.batch_size if not use_ddp else max(1, args.batch_size // 2)),
         "ddp": use_ddp},
        {"grad_ckpt": False, "batch": 1, "ddp": False},
    ]
    rc = None
    # The VM kills the trainer with a silent native SIGSEGV every ~10-40 min
    # (5x today; config-independent). The parent launcher always survives and
    # resume-from-checkpoint is cheap, so on signal deaths we relaunch the SAME
    # config instead of letting the kernel die: each relaunch adds another
    # ~30-40 min of training before the next kill. Python-level errors (rc>0)
    # still advance to the next config.
    max_retries = int(os.environ.get("KAGGLE_TRAIN_RETRIES", "3"))
    for i, attempt in enumerate(attempts):
        if i > 0:
            logger.warning("Training attempt %d failed (rc=%s); trying next config "
                           "grad-ckpt=%s batch-size=%s ddp=%s", i, rc,
                           attempt["grad_ckpt"], attempt["batch"], attempt["ddp"])
        if attempt["ddp"]:
            script = "training/full_train_ddp.py"
            launcher = [sys.executable, "-m", "torch.distributed.run",
                        "--nproc_per_node", str(gpu_count)]
            # Split grad-accum across ranks so the effective batch is
            # preserved: effective = per_gpu_batch * world_size * per_rank_accum
            # The notebook's single-GPU default is BATCH=2, accum=4 (eff 8).
            # With 2 ranks: 2 * 2 * per_rank_accum = 8 -> per_rank_accum = 2.
            per_rank_accum = max(1, args.grad_accum // gpu_count)
            ddp_flag = ["--ddp"]
        else:
            script = "training/full_train.py"
            launcher = [sys.executable]
            per_rank_accum = args.grad_accum
            ddp_flag = []
        for try_num in range(max_retries):
            rc = run(launcher + [script,
                      "--model-id", "checkpoints/init_768",
                      "--manifest", "data/training/manifest.jsonl",
                      "--steps", str(args.steps),
                      "--batch-size", str(attempt["batch"]),
                      "--grad-accum", str(per_rank_accum),
                      "--warmup", str(args.warmup),
                      "--lr", str(args.lr),
                      "--max-seq-length", str(args.max_seq_length),
                      "--save-every", str(args.save_every),
                      "--save-latest-every", str(args.save_latest_every),
                      # T4 (sm_75)/P100 have no bf16 tensor cores: bf16
                      # autocast silently produces NaN losses. Default
                      # fp16+grad-scaler path is correct on both.
                      *(["--grad-checkpoint"] if attempt["grad_ckpt"] else []),
                      "--device", "cuda", "--num-workers", "0",
                      *ddp_flag,
                      "--output-dir", OUT_DIR,
                      "--max-samples", "2000000"],
                     logfile=str(train_log))
            if rc == 0:
                break
            if rc < 0 and try_num < max_retries - 1:
                logger.warning("Config %d try %d killed by signal %d; waiting "
                               "90s then relaunching (resume continues from "
                               "last checkpoint)", i, try_num + 1, -rc)
                time.sleep(90)
                continue
            break
        if rc == 0:
            break

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
