#!/usr/bin/env python3
"""
Full-model training for TinyDoc-VLM (Tier-2 retrain, item E).

Trains ALL model parameters (no LoRA) on the markdown-conversion + real-benchmark
manifest produced by `data/build_training_dataset.py`. Runs at 768x768.

This is the "good margin" jump: the architecture now (heads removed, 768 res,
ngram penalty) is exercised by full fine-tuning on prompt-routed
markdown/text/JSON/VQA targets.

Usage (M4 Mac, pilot):
    python training/full_train.py --steps 500 --batch-size 1 --device mps

Usage (Colab T4, full run):
    python training/full_train.py --steps 30000 --batch-size 4 --device cuda
"""

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

logger = logging.getLogger(__name__)


class MarkdownTrainDataset(Dataset):
    """Reads combined manifest: {image_path, prompt, target, source}."""

    def __init__(self, manifest_path: str, max_samples: int = 1_000_000):
        self.data = []
        with open(manifest_path) as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                item = json.loads(line)
                if not item.get("image_path") or not os.path.exists(item["image_path"]):
                    continue
                if not item.get("prompt") or not item.get("target"):
                    continue
                self.data.append(item)
        logger.info(f"Loaded {len(self.data)} training pairs from {manifest_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = Image.open(item["image_path"]).convert("RGB")
        return {
            "prompt": item["prompt"],
            "target": item["target"],
            "image": img,
            "source": item.get("source", "unknown"),
        }


def collate_fn(batch, processor, max_seq_length=256):
    prompts = [f"<image>\n{item['prompt']}" for item in batch]
    answers = [item["target"] for item in batch]
    images = [item["image"] for item in batch]

    sz = processor.image_processor.image_size  # now 768
    full_texts = [f"{p}\n{a}" for p, a in zip(prompts, answers)]

    enc = processor.tokenizer(
        full_texts, padding=True, truncation=True,
        max_length=max_seq_length, return_tensors="pt",
    )
    prompt_enc = processor.tokenizer(
        prompts, padding=True, truncation=True,
        max_length=max_seq_length, return_tensors="pt",
    )

    pv = []
    for img in images:
        t = processor.image_processor.preprocess(img)
        pv.append(t)
    max_tiles = max(t.shape[0] for t in pv)
    padded = []
    for t in pv:
        T = t.shape[0]
        if T < max_tiles:
            t = torch.cat([t, torch.zeros((max_tiles - T, 3, sz, sz), dtype=t.dtype)], dim=0)
        padded.append(t)
    pixel_values = torch.stack(padded, dim=0)

    input_ids = enc["input_ids"]
    labels = input_ids.clone()
    plen = prompt_enc["attention_mask"].sum(dim=1)
    for i in range(len(batch)):
        labels[i, :plen[i].item()] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": enc["attention_mask"],
        "labels": labels,
        "pixel_values": pixel_values,
    }


def train(
    model, processor, train_dataset,
    steps=500, batch_size=1, learning_rate=1e-4,
    warmup_steps=50, grad_accum=4, log_every=10, save_every=200,
    output_dir="checkpoints/full", device="auto",
    bf16=False, grad_checkpoint=False,
):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu")

    logger.info(f"Full fine-tune on {device} for {steps} steps (all params)")
    model = model.to(device)
    if grad_checkpoint:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
    use_bf16 = bf16 and device in ("mps", "cuda")
    if use_bf16:
        logger.info("Using bf16 autocast")
    model.train()

    loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b, processor),
        num_workers=0 if device == "mps" else 2,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate, weight_decay=0.01, betas=(0.9, 0.999),
    )

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        prog = (step - warmup_steps) / max(steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    step = 0
    micro = 0
    running_loss = 0.0
    start = time.time()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    while step < steps:
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            pixel_values = batch["pixel_values"].to(device)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    loss = model(input_ids=input_ids, attention_mask=attention_mask,
                                 pixel_values=pixel_values, labels=labels).loss / grad_accum
                scaler.scale(loss).backward()
            elif use_bf16:
                with torch.amp.autocast(device, dtype=torch.bfloat16):
                    loss = model(input_ids=input_ids, attention_mask=attention_mask,
                                 pixel_values=pixel_values, labels=labels).loss / grad_accum
                loss.backward()
            else:
                loss = model(input_ids=input_ids, attention_mask=attention_mask,
                             pixel_values=pixel_values, labels=labels).loss / grad_accum
                loss.backward()

            running_loss += loss.item() * grad_accum
            micro += 1

            if micro % grad_accum == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                if device == "mps":
                    torch.mps.empty_cache()
                step += 1

                if step % log_every == 0:
                    avg = running_loss / log_every
                    logger.info(f"Step {step}/{steps} | loss={avg:.4f} | "
                                f"{step / (time.time() - start):.1f} steps/s")
                    running_loss = 0.0
                if step % save_every == 0 and step < steps:
                    sp = out / f"step_{step}"
                    sp.mkdir(exist_ok=True)
                    model.save_pretrained(str(sp))
                    logger.info(f"Saved {sp}")
                if step >= steps:
                    break
        if step >= steps:
            break

    final = out / "final"
    final.mkdir(exist_ok=True)
    model.save_pretrained(str(final))
    logger.info(f"Done. {steps} steps in {time.time() - start:.0f}s. Final: {final}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/training/manifest.jsonl")
    ap.add_argument("--model-id", default="eulogik/TinyDoc-VLM-256M")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-samples", type=int, default=1_000_000)
    ap.add_argument("--output-dir", default="checkpoints/full")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--bf16", action="store_true", help="bf16 autocast (use on MPS to save memory)")
    ap.add_argument("--grad-checkpoint", action="store_true", help="gradient checkpointing (saves activation memory)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    from tinydoc_vlm import TinyDocVLMForConditionalGeneration, TinyDocVLMProcessor

    logger.info(f"Loading model {args.model_id}")
    model = TinyDocVLMForConditionalGeneration.from_pretrained(args.model_id, trust_remote_code=True)
    processor = TinyDocVLMProcessor()
    # Keep image processor resolution in sync with the loaded model config
    processor.image_processor.image_size = model.config.image_size

    ds = MarkdownTrainDataset(args.manifest, max_samples=args.max_samples)
    if len(ds) == 0:
        logger.error("No data. Run: python data/build_training_dataset.py --num-docs 50000")
        return

    train(model, processor, ds,
          steps=args.steps, batch_size=args.batch_size, learning_rate=args.lr,
          warmup_steps=args.warmup, grad_accum=args.grad_accum,
          output_dir=args.output_dir, device=args.device,
          bf16=args.bf16, grad_checkpoint=args.grad_checkpoint)


if __name__ == "__main__":
    main()
