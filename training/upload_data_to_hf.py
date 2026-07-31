#!/usr/bin/env python3
"""
Upload data/training/ (manifest + images) to the HF dataset repo used by
kaggle_train.py. Run once after generating the dataset; re-run to update.

Usage:
    HF_TOKEN=hf_xxx python training/upload_data_to_hf.py
    HF_TOKEN=hf_xxx python training/upload_data_to_hf.py --data-dir data/training
"""

import argparse
import os
import sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/training")
    ap.add_argument("--repo", default="eulogik/TinyDoc-VLM-training-data")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Set HF_TOKEN env var first.", file=sys.stderr)
        sys.exit(1)

    from huggingface_hub import HfApi, upload_folder
    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo, repo_type="dataset", private=True,
                    exist_ok=True)
    upload_folder(
        repo_id=args.repo, folder_path=args.data_dir, repo_type="dataset",
        commit_message="training data sync",
    )
    print(f"Uploaded {args.data_dir} -> https://huggingface.co/datasets/{args.repo}")

if __name__ == "__main__":
    main()
