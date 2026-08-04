#!/usr/bin/env python3
"""Convert a saved TinyDoc-VLM checkpoint to another dtype, in place.

P100/T4 (sm_60/sm_75) have no bf16 hardware, so a bf16-saved init
checkpoint cannot be moved to CUDA there. Loading with fp16 AMP in
full_train.py then works because the weights are fp32 (AMP casts on
the fly) -- the standard recipe.

Usage:
    python training/convert_init_dtype.py --model checkpoints/init_768 --dtype float32
Skips (exit 0) if the checkpoint is already the target dtype.
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from tinydoc_vlm import TinyDocVLMForConditionalGeneration

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("convert_init_dtype")

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--dtype", required=True, choices=list(DTYPES))
    args = ap.parse_args()

    cfg_path = args.model / "config.json"
    if not cfg_path.exists():
        logger.error("no config.json at %s", args.model)
        return 1
    cfg = json.loads(cfg_path.read_text())
    cur = cfg.get("torch_dtype")
    target = args.dtype
    if cur == target:
        logger.info("already %s; nothing to do.", target)
        return 0

    logger.info("loading %s (torch_dtype=%s) ...", args.model, cur)
    with torch.no_grad():
        model = TinyDocVLMForConditionalGeneration.from_pretrained(
            str(args.model), trust_remote_code=True, low_cpu_mem_usage=True)
        logger.info("converting to %s ...", target)
        model = model.to(DTYPES[target])
        logger.info("saving ...")
        model.save_pretrained(str(args.model))
    logger.info("done: %s is now torch_dtype=%s", args.model, target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
