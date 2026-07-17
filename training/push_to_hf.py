#!/usr/bin/env python3
"""
Push the trained 768 checkpoint to Hugging Face.

Run AFTER colab_train.py completes (final checkpoint at
checkpoints/full768/final). Set your token via env (do NOT hard-code it):

    HF_TOKEN=hf_xxx python training/push_to_hf.py \
        --repo eulogik/TinyDoc-VLM-768 \
        --checkpoint checkpoints/full768/final

Pushes to a NEW repo so the legacy eulogik/TinyDoc-VLM-256M base is untouched.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="eulogik/TinyDoc-VLM-768")
    ap.add_argument("--checkpoint", default="checkpoints/full768/final")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.is_dir():
        raise SystemExit(f"Checkpoint not found: {ckpt}")

    token = args.token
    if not token:
        raise SystemExit("Set HF_TOKEN env var (or --token). Never hard-code tokens in files.")

    from huggingface_hub import login, HfApi
    login(token=token)
    api = HfApi()
    # Create repo if it doesn't exist (private=False by default is public).
    api.create_repo(repo_id=args.repo, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=str(ckpt), repo_id=args.repo, repo_type="model")
    print(f"Pushed {ckpt} -> https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
