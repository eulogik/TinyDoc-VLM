#!/usr/bin/env python3
"""
Evaluate LoRA-adapted TinyDoc-VLM.

Usage:
    python training/eval_lora.py --checkpoint checkpoints/lora_v2/final --benchmark ocrbench
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image

logger = logging.getLogger(__name__)


def load_lora_model(checkpoint_path: str, base_model_id: str = "eulogik/TinyDoc-VLM-256M"):
    """Load base model + LoRA adapter."""
    from peft import PeftModel
    from tinydoc_vlm import TinyDocVLMForConditionalGeneration, TinyDocVLMProcessor
    
    logger.info(f"Loading base model: {base_model_id}")
    model = TinyDocVLMForConditionalGeneration.from_pretrained(base_model_id, trust_remote_code=True)
    
    logger.info(f"Loading LoRA adapter: {checkpoint_path}")
    model = PeftModel.from_pretrained(model, checkpoint_path)
    model = model.merge_and_unload()
    
    processor = TinyDocVLMProcessor()
    return model, processor


def evaluate_sample(model, processor, image_path: str, question: str, device: str = "cpu") -> str:
    """Generate answer for a single sample."""
    img = Image.open(image_path).convert("RGB")
    
    prompt = f"<image>\n{question}"
    
    tile_tensor = processor.image_processor.preprocess(img)
    pixel_values = torch.stack([tile_tensor], dim=0).to(device)
    
    encoding = processor.tokenizer(prompt, return_tensors="pt")
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            max_new_tokens=128,
            do_sample=False,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    
    answer = processor.tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
    return answer.strip()


def evaluate_benchmark(model, processor, benchmark_path: str, max_samples: int = 50, device: str = "cpu"):
    """Evaluate on a benchmark dataset."""
    with open(benchmark_path) as f:
        content = f.read()
    
    if content.strip().startswith("["):
        data = json.loads(content)
    else:
        data = [json.loads(line) for line in content.split("\n") if line.strip()]
    
    correct = 0
    total = 0
    results = []
    
    for item in data[:max_samples]:
        img_path = item.get("image_path") or item.get("image", "")
        if img_path and not os.path.isabs(img_path):
            img_dir = os.path.dirname(benchmark_path)
            img_path = os.path.join(img_dir, "images", os.path.basename(img_path))
        
        question = item.get("question", "")
        answers = item.get("answers", item.get("answer", ""))
        expected = answers[0] if isinstance(answers, list) and answers else str(answers)
        
        if not img_path or not os.path.exists(img_path):
            continue
        
        start = time.time()
        predicted = evaluate_sample(model, processor, img_path, question, device)
        elapsed = time.time() - start
        
        match = predicted.lower().strip() == expected.lower().strip()
        if match:
            correct += 1
        total += 1
        
        results.append({
            "question": question,
            "expected": expected,
            "predicted": predicted,
            "match": match,
            "time": elapsed,
        })
        
        if total % 10 == 0:
            logger.info(f"Progress: {total}/{max_samples} | Accuracy: {correct/total:.1%}")
    
    accuracy = correct / total if total > 0 else 0
    logger.info(f"Final: {correct}/{total} = {accuracy:.1%}")
    
    return accuracy, results


def main():
    parser = argparse.ArgumentParser(description="Evaluate LoRA-adapted TinyDoc-VLM")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to LoRA checkpoint")
    parser.add_argument("--benchmark", type=str, default="ocrbench", choices=["ocrbench", "funsd", "cord"])
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=str, default="eval_results.json")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    
    model, processor = load_lora_model(args.checkpoint)
    model = model.to(device)
    model.eval()
    
    benchmark_dir = f"evaluation/data/{args.benchmark}"
    if not os.path.exists(benchmark_dir):
        logger.error(f"Benchmark dir not found: {benchmark_dir}")
        return
    
    # Look for JSON or JSONL files
    for ext in [".json", ".jsonl"]:
        candidate = os.path.join(benchmark_dir, f"{args.benchmark}{ext}")
        if os.path.exists(candidate):
            benchmark_path = candidate
            break
        candidate = os.path.join(benchmark_dir, f"test{ext}")
        if os.path.exists(candidate):
            benchmark_path = candidate
            break
    else:
        # Use first JSON/JSONL file found
        import glob
        files = glob.glob(os.path.join(benchmark_dir, "*.json*"))
        if not files:
            logger.error(f"No data files found in {benchmark_dir}")
            return
        benchmark_path = files[0]
    
    logger.info(f"Using benchmark: {benchmark_path}")
    
    accuracy, results = evaluate_benchmark(model, processor, benchmark_path, args.max_samples, device)
    
    output = {
        "benchmark": args.benchmark,
        "accuracy": accuracy,
        "total": len(results),
        "correct": sum(1 for r in results if r["match"]),
        "results": results,
    }
    
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
