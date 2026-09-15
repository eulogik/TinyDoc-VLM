#!/usr/bin/env python3
"""
QLoRA fine-tune of SmolVLM2-2.2B-Instruct on REAL document pairs only.

Why this exists: TinyDoc-VLM-768's vision tower died in training (outputs
constant features for any input) and the decoder memorized text templates.
Instead of reviving it, we fine-tune a proven grounded VLM on the 46k real
pairs (docvqa/sroie/docmatix/ocrbench/funsd) — no synthetic templates.

Data bundle (built on Mac, uploaded to HF dataset repo):
    real_only.tar.gz -> manifest.jsonl + datasets/{docvqa,sroie,docmatix,ocrbench,funsd}/
    manifest rows: {image_path (bundle-relative), prompt, target, source}

Dataset format for TRL: {messages: [{role, content:[{type:image},{type:text}]}], images: [PIL]}

Usage (Kaggle T4 16GB):
    pip install -U transformers trl peft bitsandbytes pillow num2words
    HF_TOKEN=... DATA_REPO=eulogik/TinyDoc-VLM-real-data \
      python training/smolvlm2_qlora.py --max-steps 3000 --output hub

Memory: fp16 base (~4.4GB) on ONE T4 + LoRA r16 + grad-ckpt + batch2/accum8 fits 15GB.
Time: ~2.9k steps/epoch, ~4-7s/step -> 4-6h for 1 epoch. Fits one Kaggle session.
"""

import argparse
import io
import json
import logging
import os
import random
import tarfile
import tempfile
import time
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"  # fallback only
MAX_TARGET_CHARS = 1500  # drop pathological long targets (table-JSON dumps)
EVAL_SPLIT = 500
SEED = 42


def fetch_bundle(data_repo: str, token: str, workdir: Path) -> Path:
    from huggingface_hub import snapshot_download
    logger.info("Downloading dataset repo %s ...", data_repo)
    local = snapshot_download(repo_id=data_repo, repo_type="dataset",
                              token=token, local_dir=str(workdir / "data"))
    root = Path(local)
    manifest = root / "manifest.jsonl"
    if not manifest.exists():
        # bundle layout: single tarball
        balls = list(root.glob("*.tar.gz"))
        assert balls, f"no manifest.jsonl or tarball in {data_repo}"
        logger.info("Extracting %s ...", balls[0].name)
        with tarfile.open(balls[0]) as tar:
            tar.extractall(root)
    assert (root / "manifest.jsonl").exists(), "manifest.jsonl missing after fetch"
    return root


def load_rows(bundle: Path):
    rows = []
    with open(bundle / "manifest.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if len(d.get("target", "")) > MAX_TARGET_CHARS:
                continue
            img = bundle / d["image_path"]
            if not img.exists():
                continue
            rows.append(d)
    rng = random.Random(SEED)
    rng.shuffle(rows)
    logger.info("Loaded %d usable real pairs", len(rows))
    return rows


def to_messages(row: dict) -> dict:
    return {
        "messages": [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": row["prompt"]},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": row["target"]},
            ]},
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-repo", default=os.environ.get("DATA_REPO",
                        "eulogik/TinyDoc-VLM-real-data"))
    ap.add_argument("--hub-id", default=os.environ.get("HUB_ID",
                        "eulogik/SmolVLM2-TinyDoc-real"))
    ap.add_argument("--output-dir", default="/tmp/smolvlm2-tinydoc")
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--dry-run", action="store_true",
                    help="Build dataset + print stats, skip training")
    ap.add_argument("--model-path", default=None,
                    help="Local path to SmolVLM2 model (Kaggle dataset). "
                         "If None, downloads from HuggingFace Hub.")
    args = ap.parse_args()
    model_path = args.model_path or MODEL_ID

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    token = os.environ.get("HF_TOKEN")
    assert token, "HF_TOKEN env required"

    import torch
    from PIL import Image

    workdir = Path(tempfile.mkdtemp(prefix="smolvlm2-data-"))
    bundle = fetch_bundle(args.data_repo, token, workdir)
    rows = load_rows(bundle)
    if args.dry_run:
        from collections import Counter
        print("rows:", len(rows))
        print(Counter(r["source"] for r in rows))
        print("example:", json.dumps(rows[0], indent=2)[:400])
        return

    from datasets import Dataset
    # Hold out eval split BEFORE mapping (keep raw rows for image loading)
    eval_rows, train_rows = rows[:EVAL_SPLIT], rows[EVAL_SPLIT:]

    def gen(split_rows):
        for r in split_rows:
            img = Image.open(bundle / r["image_path"]).convert("RGB")
            m = to_messages(r)
            yield {"messages": m["messages"], "images": [img]}

    train_ds = Dataset.from_generator(lambda: gen(train_rows))
    eval_ds = Dataset.from_generator(lambda: gen(eval_rows))

    from transformers import AutoProcessor, AutoModelForImageTextToText
    import os as _os

    logger.info("Loading model from: %s", model_path)
    _local = _os.path.isdir(model_path)
    _hf_kwargs = dict(local_files_only=_local, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path, **_hf_kwargs)

    # Load model in float16 directly (no 4-bit quantization).
    # SmolVLM2-2.2B is 4.4 GB in fp16, fits on one T4 15 GB with LoRA + grad-ckpt.
    # fp16 (not bf16): T4 is Turing (sm_75) with no bfloat16 support.
    # BitsAndBytes + device_map="auto" causes a dtype mismatch in the vision
    # encoder (float32 vs bfloat16 in inputs_merger) because accelerate's hooks
    # convert the encoder to fp32.  The official HuggingFace tutorial loads
    # SmolVLM2 without quantization for LoRA fine-tuning.
    # Single GPU only: Trainer wraps multi-GPU models in DataParallel, which is
    # broken with PEFT (replica has no fp params -> StopIteration in self.dtype).
    # The notebook forces CUDA_VISIBLE_DEVICES=0.
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, torch_dtype=torch.float16, **_hf_kwargs)
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable()
    except Exception as e:
        logger.warning("grad-ckpt not enabled: %s", e)

    from peft import LoraConfig
    peft_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    logger.info("LoRA r=%d on %s", args.lora_r, peft_cfg.target_modules)
    from trl import SFTConfig, SFTTrainer
    sft_args = SFTConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        gradient_checkpointing=True,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=90,
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=args.eval_every,
        save_strategy="steps",
        save_steps=args.eval_every,
        save_total_limit=3,
        fp16=True,  # T4 (Turing) has no bfloat16 support; use fp16, not bf16
        optim="adamw_torch_fused",
        loss_type="nll",  # TRL>=0.15 defaults to chunked_nll, whose forward-patch
                          # crashes on SmolVLM2 (lm_head.forward is a partial)
        dataset_text_field="messages",  # TRL applies chat template + masks user turns
        max_length=2560,  # SmolVLM2 tiles docs to ~17 crops; rows run 1600-2400 toks
        push_to_hub=False,
        report_to="none",
        seed=SEED,
    )
    trainer = SFTTrainer(
        model=model,  # base (quantized) model; TRL applies peft_config itself
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=processor,
        peft_config=peft_cfg,
    )
    trainer.model.print_trainable_parameters()
    logger.info("Starting SFT: max_steps=%d eff_batch=%d", args.max_steps,
                args.batch * args.accum)
    t0 = time.time()
    trainer.train()
    logger.info("Train done in %.1fh", (time.time() - t0) / 3600)

    adapter_dir = Path(args.output_dir) / "adapter_final"
    trainer.save_model(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))

    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(args.hub_id, exist_ok=True)
    api.upload_folder(repo_id=args.hub_id, folder_path=str(adapter_dir))
    logger.info("Adapter pushed -> %s", args.hub_id)


if __name__ == "__main__":
    main()
