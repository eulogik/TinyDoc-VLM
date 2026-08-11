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
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from tinydoc_vlm import TinyDocVLMForConditionalGeneration, TinyDocVLMProcessor

logger = logging.getLogger(__name__)


class MarkdownTrainDataset(Dataset):
    """Reads combined manifest: {image_path, prompt, target, source}."""

    def __init__(self, manifest_path: str, max_samples: int = 1_000_000):
        self.data = []
        # Manifest may store absolute paths from another machine; fall back to
        # resolving relative to the manifest's own directory.
        manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
        with open(manifest_path) as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                item = json.loads(line)
                if not item.get("image_path") or not item.get("prompt") or not item.get("target"):
                    continue
                img_path = item["image_path"]
                if not os.path.exists(img_path):
                    # Strip machine-specific prefix: take everything from the
                    # last 'data/training/' marker, resolve against manifest dir.
                    marker = "data/training/"
                    idx = img_path.rfind(marker)
                    rel = img_path[idx + len(marker):] if idx >= 0 else os.path.basename(img_path)
                    alt = os.path.join(manifest_dir, rel)
                    alt2 = os.path.join(manifest_dir, "synthetic", "images", os.path.basename(img_path))
                    if os.path.exists(alt):
                        img_path = alt
                    elif os.path.exists(alt2):
                        img_path = alt2
                    else:
                        continue
                item["image_path"] = img_path
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
    save_latest_every=0,
    output_dir="checkpoints/full", device="auto",
    bf16=False, grad_checkpoint=False, max_seq_length=512, resume=True,
    resume_step=None,
):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu")
    elif device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda was set but PyTorch was not compiled with CUDA support "
            "(or no GPU is available). Try '--device auto' or "
            "'pip install torch --index-url https://download.pytorch.org/whl/cu124'."
        )

    logger.info(f"Full fine-tune on {device} for {steps} steps (all params)")
    model = model.to(device)
    if grad_checkpoint:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
    use_bf16 = bf16 and device in ("mps", "cuda")
    if use_bf16:
        logger.info("Using bf16 autocast")
    model.train()

    nw = 0 if device == "mps" else 2
    loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b, processor, max_seq_length),
        num_workers=nw,
        prefetch_factor=4 if nw > 0 else None,
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
    # Autocast keeps softmax/layernorm in fp32 for numerical stability
    # even when params are fp16. GradScaler requires fp32 grads (from fp32
    # params) to unscale; skip it for fp16/bf16 params where it crashes.
    first_dtype = next(model.parameters()).dtype
    use_autocast = device == "cuda" and not use_bf16
    use_scaler = use_autocast and first_dtype == torch.float32
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None

    step = 0
    micro = 0
    running_loss = 0.0
    start = time.time()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Resume from the latest valid checkpoint. Priority:
    # 1. latest/ (frequent overwriting checkpoint, minimal loss on disconnect)
    # 2. step_* dirs (landmark checkpoints)
    # The model was already loaded from the checkpoint in main() so we only
    # need to read the step number to resume counting from there. The step
    # comes from main()'s _find_resume_checkpoint resolution (single source
    # of truth): a stale latest/step.txt must not drift the counter away
    # from the checkpoint that was actually loaded.
    if resume:
        if resume_step and resume_step > 0:
            step = resume_step
            start = time.time()
            if step >= steps:
                logger.info(f"Already at step {step} >= target {steps}; skipping training.")
        elif (out / "latest").exists() and ((out / "latest") / "step.txt").exists():
            resume_step = int(((out / "latest") / "step.txt").read_text().strip())
            if resume_step > 0:
                logger.info(f"Resuming from {(out / 'latest')} (step {resume_step})")
                step = resume_step
                start = time.time()
                if step >= steps:
                    logger.info(f"Already at step {step} >= target {steps}; skipping training.")

        if step == 0:
            step_dirs = sorted(out.glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
            for candidate in reversed(step_dirs):
                ckpt_files = [f for f in candidate.iterdir()
                              if f.name.endswith((".safetensors", ".bin"))
                              and f.stat().st_size > 0]
                if not ckpt_files:
                    logger.warning(f"Incomplete checkpoint {candidate} (no weight files), skipping")
                    continue
                resume_step = int(candidate.name.split("_")[1])
                logger.info(f"Resuming from {candidate} (step {resume_step})")
                step = resume_step
                start = time.time()
                if step >= steps:
                    logger.info(f"Already at step {step} >= target {steps}; skipping training.")
                break

    while step < steps:
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            pixel_values = batch["pixel_values"].to(device)

            if use_autocast:
                with torch.amp.autocast("cuda"):
                    loss = model(input_ids=input_ids, attention_mask=attention_mask,
                                 pixel_values=pixel_values, labels=labels).loss / grad_accum
            elif use_bf16:
                with torch.amp.autocast(device, dtype=torch.bfloat16):
                    loss = model(input_ids=input_ids, attention_mask=attention_mask,
                                 pixel_values=pixel_values, labels=labels).loss / grad_accum
            else:
                loss = model(input_ids=input_ids, attention_mask=attention_mask,
                             pixel_values=pixel_values, labels=labels).loss / grad_accum
            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += loss.item() * grad_accum
            micro += 1

            if micro % grad_accum == 0:
                if use_scaler:
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
                    tmp = out / f".step_{step}_tmp"
                    tmp.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(str(tmp))
                    if sp.exists():
                        import shutil
                        shutil.rmtree(str(sp))
                    tmp.rename(str(sp))
                    logger.info(f"Saved {sp}")
                if save_latest_every > 0 and step % save_latest_every == 0 and step > 0:
                    sp = out / "latest"
                    tmp = out / ".latest_tmp"
                    tmp.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(str(tmp))
                    (tmp / "step.txt").write_text(str(step))
                    if sp.exists():
                        import shutil
                        shutil.rmtree(str(sp))
                    tmp.rename(str(sp))
                    logger.info(f"Saved latest (step {step})")
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
    ap.add_argument("--grad-accum", type=int, default=1, help="Micro-batches per optimizer step. Keep low on Colab to limit wall-clock per step.")
    ap.add_argument("--max-samples", type=int, default=1_000_000)
    ap.add_argument("--max-seq-length", type=int, default=512, help="Max token length for prompt+target (markdown targets need >256).")
    ap.add_argument("--output-dir", default="checkpoints/full")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--bf16", action="store_true", help="bf16 autocast (use on MPS to save memory)")
    ap.add_argument("--grad-checkpoint", action="store_true", help="gradient checkpointing (saves activation memory)")
    ap.add_argument("--save-every", type=int, default=200, help="Save an intermediate checkpoint every N steps (throttled to avoid filling disk)")
    ap.add_argument("--save-latest-every", type=int, default=0, help="Overwrite latest/ checkpoint every N steps for fine-grained resume. 0 = disabled.")
    ap.add_argument("--log-every", type=int, default=10, help="Log training loss every N steps")
    ap.add_argument("--no-resume", action="store_true", help="Start from scratch instead of resuming latest step_* checkpoint.")
    ap.add_argument("--resume-from", default=None,
                    help="Explicit checkpoint dir to resume from (overrides --output-dir scan). Used "
                         "by the Colab notebook to resume from a hub-downloaded checkpoint.")
    args = ap.parse_args()

    import os as _os
    _os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    # Dump the Python stack to a SEPARATE file every 60s so a silent hang
    # (e.g. a CUDA kernel that never returns) is diagnosable in logs/
    # faulthandler.log WITHOUT polluting the captured stdout/stderr that
    # kaggle_train.py's stall watchdog measures silence by. A hung GPU shows
    # up as "File ... in train" in this file; the watchdog then kills and
    # the drive loop resumes from the last checkpoint.
    import faulthandler
    fa_path = Path("logs") / "faulthandler.log"
    fa_path.parent.mkdir(parents=True, exist_ok=True)
    faulthandler.dump_traceback_later(60, repeat=True, file=open(fa_path, "w"))
    from tinydoc_vlm import TinyDocVLMForConditionalGeneration, TinyDocVLMProcessor

    # If resuming, load from checkpoint directly (avoids OOM from loading
    # the base model AND the checkpoint on a 14.5 GiB T4 simultaneously).
    # Resolve the authoritative resume step from the checkpoint that main()
    # actually loaded (if any): max(latest, step_*), validated.
    resume_step = None
    if args.resume_from and Path(args.resume_from).exists():
        resume_ckpt = Path(args.resume_from)
        logger.info(f"Using explicit resume checkpoint {resume_ckpt}")
    else:
        resume_ckpt = _find_resume_checkpoint(Path(args.output_dir)) if not args.no_resume else None
    if resume_ckpt:
        logger.info(f"Loading model from resume checkpoint {resume_ckpt}")
        model = TinyDocVLMForConditionalGeneration.from_pretrained(str(resume_ckpt), trust_remote_code=True)
        try:
            resume_step = int((resume_ckpt / "step.txt").read_text().strip())
        except Exception:
            if resume_ckpt.name.startswith("step_"):
                resume_step = int(resume_ckpt.name.split("_")[1])
    else:
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
          bf16=args.bf16, grad_checkpoint=args.grad_checkpoint,
          save_every=args.save_every, log_every=args.log_every,
          save_latest_every=args.save_latest_every,
          max_seq_length=args.max_seq_length, resume=not args.no_resume,
          resume_step=resume_step)


def _find_resume_checkpoint(out: Path) -> Path | None:
    """Find the most valid, HIGHEST-step checkpoint for resume.

    Latest/ is the frequent save but can be stale (a crash mid-atomic-rename
    leaves it at an older step, or step.txt written without weights).
    step_* dirs are atomic (saved via tmp+rename) so always valid. Pick
    whichever has the highest step, validating weights in both.
    """
    def _valid_step(p: Path) -> int | None:
        if not p.exists():
            return None
        ckpt_files = [f for f in p.iterdir()
                      if f.name.endswith((".safetensors", ".bin"))
                      and f.stat().st_size > 0]
        if not ckpt_files:
            return None
        if (p / "step.txt").exists():
            try:
                s = int((p / "step.txt").read_text().strip())
            except ValueError:
                return None
            if s > 0:
                return s
        if p.name.startswith("step_"):
            return int(p.name.split("_")[1])
        return None

    best, best_step = None, -1
    for p in [out / "latest"] + sorted(out.glob("step_*"),
                                       key=lambda x: int(x.name.split("_")[1])):
        s = _valid_step(p)
        if s is not None and s > best_step:
            best, best_step = p, s
    if best is not None:
        logger.info(f"Found resume checkpoint {best} (step {best_step})")
    return best


if __name__ == "__main__":
    main()
