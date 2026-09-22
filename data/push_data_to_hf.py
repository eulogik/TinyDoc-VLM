#!/usr/bin/env python3
"""
Deterministic data-repo push for the training bundle:
  1. delete + recreate eulogik/TinyDoc-VLM-training-data (private)
  2. upload data/training/ (manifest, stats, synthetic/, datasets/)
  3. verify training.tar.gz exists on HF with the same byte size

Usage:
    HF_TOKEN=hf_xxx python3 data/push_data_to_hf.py
"""

import os
import sys

from huggingface_hub import HfApi, upload_folder

REPO = "eulogik/TinyDoc-VLM-training-data"
DATA_DIR = "data/training"
BUNDLE = f"{DATA_DIR}/training.tar.gz"


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Set HF_TOKEN env var first.", file=sys.stderr)
        sys.exit(1)
    api = HfApi(token=token)

    print(f"wiping {REPO} ...")
    api.delete_repo(repo_id=REPO, repo_type="dataset", missing_ok=True)
    api.create_repo(repo_id=REPO, repo_type="dataset", private=True, exist_ok=False)

    local_size = os.path.getsize(BUNDLE)
    print(f"uploading {DATA_DIR} ({local_size / 1e9:.2f} GB bundle) ...")
    upload_folder(repo_id=REPO, folder_path=DATA_DIR, repo_type="dataset",
                  commit_message="gold-class training data (fixed synthetic + public)")

    files = api.list_repo_files(REPO, repo_type="dataset")
    if "training.tar.gz" not in files:
        print("FAIL: training.tar.gz missing from repo", file=sys.stderr)
        sys.exit(1)
    info = api.get_paths_info(REPO, paths=["training.tar.gz"],
                              repo_type="dataset")
    remote_size = info[0].size if info else -1
    print(f"repo files: {len(files)} | training.tar.gz remote size: {remote_size / 1e9:.2f} GB")
    if remote_size != local_size:
        print("FAIL: remote size mismatch", file=sys.stderr)
        sys.exit(1)
    print("PUSH VERIFIED OK")


if __name__ == "__main__":
    main()