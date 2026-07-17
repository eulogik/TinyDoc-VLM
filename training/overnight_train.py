#!/usr/bin/env python3
"""
Full overnight training run for TinyDoc-VLM on M4 Mac.

1. Generate 3K synthetic documents (~200MB)
2. Train LoRA for ~17K steps with cosine annealing
3. Save best checkpoints, eval periodically

Target: 15-16 hours on M4 MPS at 0.3 steps/s

Usage:
    python training/overnight_train.py
"""

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

logger = logging.getLogger(__name__)


class SyntheticDocDataset(Dataset):
    """Dataset from synthetic document manifest + images."""
    
    def __init__(self, manifest_path: str, data_root: str, processor, max_seq_length: int = 512):
        self.processor = processor
        self.max_seq_length = max_seq_length
        self.data = []
        
        with open(manifest_path) as f:
            for line in f:
                item = json.loads(line)
                img_path = os.path.join(data_root, item.get("image_path", ""))
                if not img_path or not os.path.exists(img_path):
                    continue
                    
                for qa in item.get("qa_pairs", []):
                    self.data.append({
                        "image_path": img_path,
                        "question": qa.get("question", ""),
                        "answer": qa.get("answer", ""),
                    })
        
        logger.info(f"Loaded {len(self.data)} QA pairs")
    
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
    from peft import LoraConfig, get_peft_model, TaskType
    
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


def train_overnight(
    model,
    processor,
    train_dataset,
    output_dir: str = "checkpoints/overnight",
    total_steps: int = 17000,
    batch_size: int = 2,
    grad_accum: int = 4,
    learning_rate: float = 2e-4,
    warmup_steps: int = 500,
    save_every: int = 1000,
    eval_every: int = 500,
    log_every: int = 25,
    max_checkpoints: int = 3,
):
    """Full overnight training with cosine annealing and best-checkpoint tracking."""
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Training on {device} for {total_steps} steps")
    logger.info(f"Batch size: {batch_size} × {grad_accum} grad_accum = {batch_size * grad_accum} effective")
    logger.info(f"LR: {learning_rate}, Warmup: {warmup_steps} steps")
    logger.info(f"Saving every {save_every} steps, keeping top {max_checkpoints}")
    
    model = model.to(device)
    model.train()
    
    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, processor),
        num_workers=0,
        pin_memory=False,
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
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Track best checkpoints by loss
    best_checkpoints = []  # [(loss, step, path)]
    
    step = 0
    epoch = 0
    running_loss = 0.0
    start_time = time.time()
    best_loss = float("inf")
    
    # Resume from latest checkpoint if exists
    resume_path = output_path / "latest"
    if resume_path.exists():
        adapter_path = resume_path / "adapter_model.safetensors"
        if adapter_path.exists():
            from peft import PeftModel
            logger.info(f"Resuming from {resume_path}")
            model = PeftModel.from_pretrained(model, str(resume_path))
            # Try to read step from metadata
            meta_path = resume_path / "training_meta.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                    step = meta.get("step", 0)
                    best_loss = meta.get("best_loss", float("inf"))
                    best_checkpoints = meta.get("best_checkpoints", [])
                    logger.info(f"Resumed from step {step}, best_loss={best_loss:.4f}")
    
    while step < total_steps:
        epoch += 1
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            
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
                    eta = (total_steps - step) / steps_per_sec if steps_per_sec > 0 else 0
                    logger.info(
                        f"Step {step}/{total_steps} | loss={avg_loss:.4f} | "
                        f"lr={lr:.2e} | {steps_per_sec:.2f} steps/s | "
                        f"elapsed={elapsed/3600:.1f}h | ETA={eta/3600:.1f}h"
                    )
                    running_loss = 0.0
                
                if step % save_every == 0:
                    save_path = output_path / f"step_{step}"
                    save_path.mkdir(exist_ok=True)
                    model.save_pretrained(str(save_path))
                    
                    # Track best
                    avg_loss = running_loss / max(log_every, 1) if running_loss > 0 else avg_loss
                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        best_link = output_path / "best"
                        if best_link.exists():
                            best_link.unlink()
                        best_link.symlink_to(save_path.name)
                        logger.info(f"  ★ New best: loss={avg_loss:.4f}")
                    
                    # Manage checkpoint rotation
                    best_checkpoints.append((avg_loss, step, str(save_path)))
                    best_checkpoints.sort(key=lambda x: x[0])
                    if len(best_checkpoints) > max_checkpoints:
                        worst = best_checkpoints.pop()
                        worst_path = Path(worst[2])
                        if worst_path.exists():
                            import shutil
                            shutil.rmtree(worst_path)
                            logger.info(f"  Removed old checkpoint: {worst_path.name}")
                
                if step % eval_every == 0 and step > 0:
                    # Save latest for resume
                    latest_path = output_path / "latest"
                    latest_path.mkdir(exist_ok=True)
                    model.save_pretrained(str(latest_path))
                    
                    # Save training metadata
                    meta = {
                        "step": step,
                        "best_loss": best_loss,
                        "best_checkpoints": best_checkpoints,
                        "total_steps": total_steps,
                        "elapsed_hours": (time.time() - start_time) / 3600,
                    }
                    with open(latest_path / "training_meta.json", "w") as f:
                        json.dump(meta, f, indent=2)
                
                if step >= total_steps:
                    break
        
        logger.info(f"Epoch {epoch} complete. Step {step}/{total_steps}")
    
    # Save final
    final_path = output_path / "final"
    final_path.mkdir(exist_ok=True)
    model.save_pretrained(str(final_path))
    
    total_time = time.time() - start_time
    logger.info(f"Training complete! {step} steps in {total_time/3600:.1f}h")
    logger.info(f"Best loss: {best_loss:.4f}")
    logger.info(f"Best checkpoints: {[(f'{l:.2f}', s) for l, s, _ in best_checkpoints]}")
    logger.info(f"Final checkpoint: {final_path}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Overnight training for TinyDoc-VLM")
    parser.add_argument("--steps", type=int, default=17000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--num-docs", type=int, default=3000)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--output-dir", type=str, default="checkpoints/overnight")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("training/overnight_log.txt"),
        ]
    )
    
    # Step 1: Generate synthetic data
    data_dir = "data/synthetic/overnight_output"
    manifest_path = f"{data_dir}/manifest.jsonl"
    
    if not os.path.exists(manifest_path):
        logger.info(f"Generating {args.num_docs} synthetic documents...")
        os.system(f"python data/synthetic/generator.py --num-docs {args.num_docs} --output-dir {data_dir}")
        logger.info(f"Data generated: {data_dir}")
    else:
        logger.info(f"Using existing data: {manifest_path}")
    
    # Step 2: Load model and apply LoRA
    from tinydoc_vlm import TinyDocVLMForConditionalGeneration, TinyDocVLMProcessor
    
    model_id = "eulogik/TinyDoc-VLM-256M"
    logger.info(f"Loading model: {model_id}")
    model = TinyDocVLMForConditionalGeneration.from_pretrained(model_id, trust_remote_code=True)
    processor = TinyDocVLMProcessor()
    
    model = apply_lora(model, rank=args.lora_rank)
    
    # Step 3: Load dataset
    logger.info("Loading dataset...")
    dataset = SyntheticDocDataset(
        manifest_path=manifest_path,
        data_root="data/synthetic",
        processor=processor,
    )
    logger.info(f"Dataset: {len(dataset)} QA pairs")
    
    # Step 4: Train
    model = train_overnight(
        model=model,
        processor=processor,
        train_dataset=dataset,
        output_dir=args.output_dir,
        total_steps=args.steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup,
    )


if __name__ == "__main__":
    main()
