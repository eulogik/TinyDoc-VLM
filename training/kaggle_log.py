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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tail", type=int, help="print last N lines of full_train.log")
    ap.add_argument("--poll", action="store_true", help="follow until DONE/failure")
    ap.add_argument("--interval", type=int, default=20)
    ap.add_argument("--timeout", type=int, default=9 * 3600)
    args = ap.parse_args()

    token = get_token()
    if args.tail:
        tail_file("full_train.log", args.tail, token, force=True)
        return
    if args.poll:
        sys.exit(poll(args, token))
    for name in FILES:
        print(f"\n===== {name} =====")
        tail_file(name, 25, token, force=True)


if __name__ == "__main__":
    main()
