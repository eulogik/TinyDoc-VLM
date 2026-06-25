#!/usr/bin/env python3
"""
Fast LoRA training for TinyDoc-VLM — runs on Colab free T4 or M4 Mac.

This script fine-tunes TinyDoc-VLM using LoRA (Low-Rank Adaptation) which:
- Reduces trainable params from 290M → ~5M (58x less memory)
- Trains 20-50x faster than full fine-tuning
- Fits in 4GB VRAM (Colab T4) or 8GB unified memory (M4 Mac)

Usage (Colab):
    !python training/fast_train.py --steps 5000 --batch-size 4

Usage (M4 Mac):
    python training/fast_train.py --steps 5000 --batch-size 2 --device mps
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

logger = logging.getLogger(__name__)


class SyntheticDocDataset(Dataset):
    """Dataset from synthetic document manifest + images."""
    
    def __init__(self, manifest_path: str, data_root: str, processor, max_samples: int = 50000):
        self.processor = processor
        self.data = []
        
        with open(manifest_path) as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                item = json.loads(line)
                img_path = os.path.join(data_root, item.get("image_path", ""))
                if not img_path or not os.path.exists(img_path):
                    continue
                    
                doc_type = item.get("doc_type", "unknown")
                qa_pairs = item.get("qa_pairs", [])
                
                for qa in qa_pairs:
                    self.data.append({
                        "image_path": img_path,
                        "question": qa.get("question", ""),
                        "answer": qa.get("answer", ""),
                        "doc_type": doc_type,
                    })
        
        logger.info(f"Loaded {len(self.data)} QA pairs from {manifest_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        img = Image.open(item["image_path"]).convert("RGB")
        
        prompt = f"<image>\n{item['question']}"
        answer = item["answer"]
        
        return {
            "prompt": prompt,
            "answer": answer,
            "image": img,
            "doc_type": item["doc_type"],
        }


def collate_fn(batch, processor, max_seq_length=512):
    """Collate batch into model inputs."""
    prompts = [item["prompt"] for item in batch]
    answers = [item["answer"] for item in batch]
    images = [item["image"] for item in batch]
    
    full_texts = [f"{p}\n{a}" for p, a in zip(prompts, answers)]
    
    encodings = processor.tokenizer(
        full_texts,
        padding=True,
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )
    
    pixel_values_list = []
    for img in images:
        tile_tensor = processor.image_processor.preprocess(img)
        pixel_values_list.append(tile_tensor)
    
    max_tiles = max(tv.shape[0] for tv in pixel_values_list)
    padded = []
    for tv in pixel_values_list:
        T = tv.shape[0]
        if T < max_tiles:
            pad = torch.zeros((max_tiles - T, 3, 384, 384), dtype=tv.dtype)
            tv = torch.cat([tv, pad], dim=0)
        padded.append(tv)
    pixel_values = torch.stack(padded, dim=0)
    
    input_ids = encodings["input_ids"]
    labels = input_ids.clone()
    
    prompt_only = processor.tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )
    prompt_lengths = prompt_only["attention_mask"].sum(dim=1)
    
    for i in range(len(batch)):
        prompt_len = prompt_lengths[i].item()
        labels[i, :prompt_len] = -100
    
    return {
        "input_ids": input_ids,
        "attention_mask": encodings["attention_mask"],
        "labels": labels,
        "pixel_values": pixel_values,
    }


def apply_lora(model, rank=16, alpha=32):
    """Apply LoRA to the decoder's attention layers."""
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
        logger.error("peft not installed. Run: pip install peft")
        raise
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def train(
    model,
    processor,
    train_dataset,
    steps: int = 5000,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    warmup_steps: int = 200,
    grad_accum: int = 4,
    log_every: int = 10,
    save_every: int = 500,
    output_dir: str = "checkpoints/lora",
    device: str = "auto",
):
    """Fast LoRA training loop."""
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    logger.info(f"Training on {device} for {steps} steps")
    
    model = model.to(device)
    model.train()
    
    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, processor),
        num_workers=0 if device == "mps" else 2,
        pin_memory=device == "cuda",
    )
    
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.999),
    )
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    import math
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    
    step = 0
    epoch = 0
    running_loss = 0.0
    start_time = time.time()
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    while step < steps:
        epoch += 1
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            
            if use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        labels=labels,
                    )
                    loss = outputs.loss / grad_accum
                scaler.scale(loss).backward()
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    labels=labels,
                )
                loss = outputs.loss / grad_accum
                loss.backward()
            
            running_loss += loss.item() * grad_accum
            
            if (batch_idx + 1) % grad_accum == 0:
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
                step += 1
                
                if step % log_every == 0:
                    avg_loss = running_loss / log_every
                    elapsed = time.time() - start_time
                    steps_per_sec = step / elapsed
                    lr = scheduler.get_last_lr()[0]
                    logger.info(
                        f"Step {step}/{steps} | loss={avg_loss:.4f} | "
                        f"lr={lr:.2e} | {steps_per_sec:.1f} steps/s | "
                        f"elapsed={elapsed:.0f}s"
                    )
                    running_loss = 0.0
                
                if step % save_every == 0:
                    save_path = output_path / f"step_{step}"
                    save_path.mkdir(exist_ok=True)
                    model.save_pretrained(str(save_path))
                    logger.info(f"Saved checkpoint: {save_path}")
                
                if step >= steps:
                    break
        
        logger.info(f"Epoch {epoch} complete. Step {step}/{steps}")
    
    final_path = output_path / "final"
    final_path.mkdir(exist_ok=True)
    model.save_pretrained(str(final_path))
    
    total_time = time.time() - start_time
    logger.info(f"Training complete! {steps} steps in {total_time:.0f}s ({steps/total_time:.1f} steps/s)")
    logger.info(f"Final checkpoint: {final_path}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Fast LoRA training for TinyDoc-VLM")
    parser.add_argument("--manifest", type=str, default="data/synthetic/output/manifest.jsonl")
    parser.add_argument("--data-root", type=str, default="data/synthetic")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=50000)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--output-dir", type=str, default="checkpoints/lora")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--model-id", type=str, default="eulogik/TinyDoc-VLM-256M")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    from tinydoc_vlm import TinyDocVLMForConditionalGeneration, TinyDocVLMProcessor
    
    logger.info(f"Loading model: {args.model_id}")
    model = TinyDocVLMForConditionalGeneration.from_pretrained(args.model_id, trust_remote_code=True)
    processor = TinyDocVLMProcessor()
    
    model = apply_lora(model, rank=args.lora_rank)
    
    logger.info(f"Loading dataset: {args.manifest}")
    dataset = SyntheticDocDataset(
        manifest_path=args.manifest,
        data_root=args.data_root,
        processor=processor,
        max_samples=args.max_samples,
    )
    
    if len(dataset) == 0:
        logger.error("No training data found! Generate data first: python data/synthetic/generator.py --num-docs 10000")
        return
    
    logger.info(f"Training data: {len(dataset)} QA pairs")
    
    model = train(
        model=model,
        processor=processor,
        train_dataset=dataset,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=args.warmup,
        grad_accum=args.grad_accum,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
