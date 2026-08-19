#!/usr/bin/env python3
"""
Quality-check TinyDoc-VLM-768 full fine-tune (final checkpoint, step 8000).

Loads the hub checkpoint (eulogik/TinyDoc-VLM-768-checkpoints), renders a
FRESH synthetic eval set (new seed, unseen pages) with ground truth, and
scores:
  1. VQA exact-match (normalized) — the quantitative signal
  2. Markdown / text-extraction outputs — printed for eyeball review

Usage:
    python training/eval_768.py                          # local eval, auto device
    python training/eval_768.py --device mps --pages 15  # explicit device
    python training/eval_768.py --ckpt-dir /path/local   # skip hub download
"""

import argparse
import json
import logging
import random
import re
import sys
import time
from pathlib import Path

# Allow running as `python training/eval_768.py` from a fresh repo clone
# (no pip install): make the repo root importable for tinydoc_vlm.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

logger = logging.getLogger(__name__)


def normalize(s: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation for loose matching."""
    s = re.sub(r"[\$£€]|\s+", "", s.lower())
    return s.strip(" .,:;\"'`!?")


def loose_match(pred: str, truth: str) -> bool:
    p, t = normalize(pred), normalize(truth)
    if not t:
        return False
    return p == t or t in p or p in t


def load_model(ckpt_dir: Path, device: str):
    from tinydoc_vlm import TinyDocVLMForConditionalGeneration, TinyDocVLMProcessor

    logger.info("Loading model from %s", ckpt_dir)
    model = TinyDocVLMForConditionalGeneration.from_pretrained(
        str(ckpt_dir), trust_remote_code=True)
    processor = TinyDocVLMProcessor()
    processor.image_processor.image_size = model.config.image_size
    logger.info("image_size=%d (tiles: %d tokens each)",
                model.config.image_size,
                (model.config.image_size // model.config.patch_size // model.config.pixel_shuffle_scale) ** 2)
    model = model.to(device)
    model.eval()
    return model, processor


def generate_sample(model, processor, img, question: str, device: str, max_new_tokens: int = 256) -> str:
    prompt = f"<image>\n{question}"
    inputs = processor(text=prompt, images=[img], return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        with torch.inference_mode():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.pad_token_id,
                # use_cache=True hangs on MPS via the custom
                # prepare_inputs_for_generation KV-cache path (the GPU command
                # buffer never completes -> waitUntilCompleted deadlock).
                # Disabling cache is slower but works on all devices and is
                # fine for a one-shot quality check.
                use_cache=False,
            )
    return processor.tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()


def build_eval_set(eval_dir: Path, pages: int, seed: int):
    """Render fresh synthetic pages with ground truth (unseen seed)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from data.synthetic.generator import ContentGenerator, DOCUMENT_TYPES
    from data.synthetic.pil_renderer import render_document, augment_image

    rng = random.Random(seed)
    images_dir = eval_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    items = []
    for i in range(pages):
        doc_type = DOCUMENT_TYPES[i % len(DOCUMENT_TYPES)]
        meta = ContentGenerator.generate(doc_type)
        img = render_document(doc_type, meta)
        img = augment_image(img)
        path = images_dir / f"page_{i:03d}_{doc_type}.png"
        img.save(path)
        qas = ContentGenerator.generate_qa(meta, doc_type)
        items.append({"page": str(path), "doc_type": doc_type,
                      "metadata": meta, "qa": qas})
        rng.seed(seed + i)
    manifest = eval_dir / "manifest.json"
    manifest.write_text(json.dumps(items, indent=2, default=str))
    logger.info("Rendered %d eval pages -> %s", pages, images_dir)
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default=None,
                    help="Local checkpoint dir. Default: download hub latest into checkpoints/eval_768")
    ap.add_argument("--ckpt-repo", default="eulogik/TinyDoc-VLM-768-checkpoints")
    ap.add_argument("--pages", type=int, default=12, help="Fresh eval pages to render")
    ap.add_argument("--seed", type=int, default=20260814, help="Eval-set seed (must differ from training seed 42)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--output", default="eval_768_results.json")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--no-qa", action="store_true", help="Skip VQA scoring; only print markdown/text outputs")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    for _noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(_noisy).setLevel(logging.WARNING)

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    logger.info("Device: %s", device)

    # ---- 1. Model ----
    if args.ckpt_dir:
        ckpt = Path(args.ckpt_dir)
    else:
        ckpt = Path("checkpoints") / "eval_768"
        if not (ckpt / "latest" / "model.safetensors").exists():
            from huggingface_hub import snapshot_download
            ckpt.mkdir(parents=True, exist_ok=True)
            logger.info("Downloading %s (latest = step 8000) ...", args.ckpt_repo)
            snapshot_download(repo_id=args.ckpt_repo, local_dir=str(ckpt))
        # The hub repo keeps the live checkpoint under latest/ (uploaded by
        # CkptSyncer). The ROOT-level model.safetensors/step.txt is a stale
        # legacy flat upload (step 50) and MUST NOT be used.
        if (ckpt / "latest" / "model.safetensors").exists():
            ckpt = ckpt / "latest"
        step_txt = ckpt / "step.txt"
        if step_txt.exists():
            logger.info("Checkpoint step: %s", step_txt.read_text().strip())
    model, processor = load_model(ckpt, device)

    # ---- 2. Fresh eval pages ----
    eval_dir = Path("data") / "eval_768"
    items = build_eval_set(eval_dir, args.pages, args.seed)
    if not items:
        logger.error("No eval pages built")
        return

    # ---- 3. Run ----
    results = []
    qa_correct = qa_total = 0
    for item in items:
        img = __import__("PIL").Image.open(item["page"]).convert("RGB")
        entry = {"page": item["page"], "doc_type": item["doc_type"],
                 "markdown": None, "extract_text": None, "qa": []}

        for label, question, tokens in [
            ("markdown", "Convert the document to markdown:", args.max_new_tokens),
            ("extract_text", "Extract all text:", min(args.max_new_tokens, 512)),
        ]:
            t0 = time.time()
            entry[label] = generate_sample(model, processor, img, question, device, tokens)
            logger.info("[%s] %s (%s, %.1fs)", item["doc_type"], label, item["page"].split("/")[-1], time.time() - t0)

        if not args.no_qa:
            for qa in item["qa"]:
                # Training used the "Answer the question: " prefix on QA
                # prompts (see data/training/manifest.jsonl); match it or the
                # model emits off-distribution garbage.
                q_prompt = f"Answer the question: {qa['question']}"
                t0 = time.time()
                pred = generate_sample(model, processor, img, q_prompt, device, min(args.max_new_tokens, 128))
                match = loose_match(pred, qa["answer"])
                qa_total += 1
                qa_correct += int(match)
                entry["qa"].append({"question": qa["question"], "answer": qa["answer"],
                                    "predicted": pred, "match": match,
                                    "time_s": round(time.time() - t0, 2)})
        results.append(entry)

    # ---- 4. Report ----
    qa_acc = qa_correct / qa_total if qa_total else None
    logger.info("=" * 70)
    if qa_acc is not None:
        logger.info("VQA exact/loose-match accuracy: %d/%d = %.1f%%",
                    qa_correct, qa_total, 100 * qa_acc)
    logger.info("=" * 70)
    for r in results:
        logger.info("--- %s (%s) ---", r["doc_type"], Path(r["page"]).name)
        if r["markdown"]:
            logger.info("MARKDOWN:\n%s", r["markdown"][:600])
        if r["extract_text"]:
            logger.info("EXTRACT:\n%s", r["extract_text"][:400])
        if r["qa"]:
            for q in r["qa"]:
                mark = "OK " if q["match"] else "XX "
                logger.info("  [%s] %s -> %s | expected: %s",
                            mark, q["question"], q["predicted"][:60], q["answer"][:40])

    out = {"device": device, "checkpoint_step": step_txt.read_text().strip() if (ckpt / "step.txt").exists() else "unknown",
           "pages": len(results), "qa_accuracy": qa_acc, "qa_correct": qa_correct,
           "qa_total": qa_total, "results": results}
    Path(args.output).write_text(json.dumps(out, indent=2, default=str))
    logger.info("Results -> %s", args.output)


if __name__ == "__main__":
    main()
