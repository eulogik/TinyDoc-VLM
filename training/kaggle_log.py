#!/usr/bin/env python3
"""Follow a Kaggle TinyDoc-VLM run from your Mac.

The kernel uploads its own + full_train.py output to the private HF repo
eulogik/TinyDoc-VLM-runtime (logs/*). This script tails those files so you
never have to copy-paste the Kaggle browser log again.

Usage:
    python training/kaggle_log.py              # print current logs once
    python training/kaggle_log.py --tail 60    # last 60 lines of full_train.log
    python training/kaggle_log.py --poll       # follow until DONE or failure
    python training/kaggle_log.py --poll --poll-file full_train.log

Token: $HF_TOKEN, or training/.hf_token (gitignored).
Exit: 0 on success (DONE), 1 on failure, 2 on timeout/no log yet.
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

LOG_REPO = "eulogik/TinyDoc-VLM-runtime"
KERNEL = "eulogikdevelopers/tinydoc-vlm-768-retrain"
FILES = ["kaggle_train.log", "env.txt", "convert.log", "full_train.log", "last_error.txt"]
HERE = Path(__file__).resolve().parent

FAIL_MARKERS = ["Training exited", "pip install FAILED", "aborting",
                "torch import failed", "init_768 build failed", "Run was terminated"]
DONE_MARKERS = ["DONE. Final checkpoint"]
STEP_RE = re.compile(r"Step (\d+)/(\d+) \| loss=([\d.e+-]+)")
PROG_RE = re.compile(r"\b(\d+)%\|")


def get_token():
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    f = HERE / ".hf_token"
    if f.exists():
        return f.read_text().strip()
    print("No HF token: set HF_TOKEN env var or write training/.hf_token", file=sys.stderr)
    sys.exit(2)


def fetch(repo, name, token, force=False):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
    try:
        return hf_hub_download(repo_id=repo, filename=f"logs/{name}",
                               token=token, force_download=force)
    except (EntryNotFoundError, RepositoryNotFoundError):
        return None


def tail_file(name, n, token, force=False):
    p = fetch(LOG_REPO, name, token, force=force)
    if not p:
        print(f"[no {name} yet]")
        return
    lines = Path(p).read_text(errors="replace").splitlines()
    for l in lines[-n:]:
        print(l)


def poll(args, token):
    seen = {}  # name -> bytes already printed
    deadline = time.time() + args.timeout
    while time.time() < deadline:
        for name in FILES:
            p = fetch(LOG_REPO, name, token)
            if not p:
                continue
            data = Path(p).read_bytes()
            offset = seen.get(name, 0)
            if len(data) < offset:
                offset = 0  # run restarted, file replaced
            if len(data) > offset:
                tail = data[offset:].decode(errors="replace")
                if tail.strip():
                    print(f"\n[{name}]")
                    sys.stdout.write(tail)
                    sys.stdout.flush()
                seen[name] = len(data)
        # Decide based on kaggle_train.log content
        p = fetch(LOG_REPO, "kaggle_train.log", token)
        if p:
            text = Path(p).read_text(errors="replace")
            for m in FAIL_MARKERS:
                if m in text:
                    print(f"\n*** FAILURE DETECTED: {m} ***", file=sys.stderr)
                    tail_file("last_error.txt", 120, token, force=True)
                    return 1
            for m in DONE_MARKERS:
                if m in text:
                    print(f"\n*** SUCCESS: {m} ***")
                    return 0
        pf = fetch(LOG_REPO, "full_train.log", token)
        if pf:
            steps = STEP_RE.findall(Path(pf).read_text(errors="replace"))
            if steps:
                s = steps[-1]
                print(f"[step {s[0]}/{s[1]} loss={s[2]}]")
        else:
            print("[kernel hasn't started logging yet]")
        time.sleep(args.interval)
    print(f"\n*** TIMEOUT after {args.timeout}s (kernel likely killed) ***", file=sys.stderr)
    tail_file("full_train.log", 60, token, force=True)
    return 2


def kernels_status():
    import shutil
    import subprocess
    kaggle = shutil.which("kaggle")
    if not kaggle:
        alt = Path.home() / ".kaggle-venv" / "bin" / "kaggle"
        if alt.exists():
            kaggle = str(alt)
    if not kaggle:
        return "unknown"
    r = subprocess.run([kaggle, "kernels", "status", KERNEL],
                       capture_output=True, text=True, timeout=60)
    out = r.stdout or r.stderr or ""
    for s in ("RUNNING", "QUEUED", "ERROR", "CANCELED", "COMPLETE"):
        if s in out:
            return s
    return "unknown"


def drive(args, token):
    """Autonomous loop: watch the run; when a session ends (12h kill, crash),
    re-push the kernel to resume; stop on DONE or a hard failure."""
    import subprocess
    import shutil
    HERE = Path(__file__).resolve().parent
    kaggle = shutil.which("kaggle") or str(Path.home() / ".kaggle-venv" / "bin" / "kaggle")
    env = dict(os.environ, PATH=f"{Path(kaggle).parent}:{os.environ.get('PATH', '')}")
    seen = {}
    last_progress = time.time()
    pushes = 0
    print(f"[drive] watching {KERNEL}; auto-repush enabled")
    while True:
        for name in FILES:
            p = fetch(LOG_REPO, name, token)
            if not p:
                continue
            data = Path(p).read_bytes()
            offset = seen.get(name, 0)
            if len(data) < offset:
                offset = 0
            if len(data) > offset:
                tail = data[offset:].decode(errors="replace")
                if tail.strip():
                    print(f"\n[{name}]")
                    sys.stdout.write(tail)
                    sys.stdout.flush()
                seen[name] = len(data)
                if name in ("full_train.log", "kaggle_train.log"):
                    last_progress = time.time()
        p = fetch(LOG_REPO, "kaggle_train.log", token)
        if p:
            text = Path(p).read_text(errors="replace")
            for m in FAIL_MARKERS:
                if m in text:
                    print(f"\n*** FAILURE DETECTED: {m} ***", file=sys.stderr)
                    tail_file("last_error.txt", 120, token, force=True)
                    return 1
            for m in DONE_MARKERS:
                if m in text:
                    print(f"\n*** SUCCESS: {m} ***")
                    return 0
        pf = fetch(LOG_REPO, "full_train.log", token)
        if pf:
            steps = STEP_RE.findall(Path(pf).read_text(errors="replace"))
            if steps:
                print(f"[step {steps[-1][0]}/{steps[-1][1]} loss={steps[-1][2]}]")
        status = kernels_status()
        print(f"[status {status}, {time.time() - last_progress:.0f}s since last log]")
        if status in ("ERROR", "CANCELED", "COMPLETE", "unknown"):
            idle = time.time() - last_progress
            if idle > 120:
                print(f"[drive] session ended ({status}, {idle:.0f}s idle); re-pushing...")
                r = subprocess.run(["bash", str(HERE / "run_kaggle.sh")],
                                   cwd=HERE.parent, env=env, timeout=300)
                if r.returncode != 0:
                    print(f"[drive] run_kaggle.sh failed rc={r.returncode}; "
                          "waiting for a free slot...", file=sys.stderr)
                pushes += 1
                last_progress = time.time()
                time.sleep(60)
        time.sleep(args.interval)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tail", type=int, help="print last N lines of full_train.log")
    ap.add_argument("--poll", action="store_true", help="follow until DONE/failure")
    ap.add_argument("--drive", action="store_true",
                    help="watch + auto re-push the kernel until training is DONE")
    ap.add_argument("--interval", type=int, default=20)
    ap.add_argument("--timeout", type=int, default=9 * 3600)
    args = ap.parse_args()

    token = get_token()
    if args.tail:
        tail_file("full_train.log", args.tail, token, force=True)
        return
    if args.drive:
        sys.exit(drive(args, token))
    if args.poll:
        sys.exit(poll(args, token))
    for name in FILES:
        print(f"\n===== {name} =====")
        tail_file(name, 25, token, force=True)


if __name__ == "__main__":
    main()
