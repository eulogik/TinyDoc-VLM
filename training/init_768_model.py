#!/usr/bin/env python3
"""
Initialize a 768x768 TinyDoc-VLM for full retraining (Tier-2, item B+E).

The published base checkpoint is 384x384. To retrain at the new default 768
resolution we must:
  1. Copy the trained decoder + compressor from the base checkpoint.
  2. Take the base's SigLIP vision encoder (384 -> 24x24 = 576 pos emb) and
     bilinearly interpolate its positional embeddings to the 768 grid
     (48x48 = 2304), so the vision backbone keeps its pretrained features.
  3. Keep the 256-token learnable visual_pos_embed random (it is learned).

Output: a directory with a 768 TinyDocVLMForConditionalGeneration ready for
`training/full_train.py --model-id <out>`.

Usage:
    python training/init_768_model.py --base eulogik/TinyDoc-VLM-256M --out checkpoints/init_768
"""

import argparse
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="eulogik/TinyDoc-VLM-256M")
    ap.add_argument("--out", default="checkpoints/init_768")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from tinydoc_vlm import TinyDocVLMConfig, TinyDocVLMForConditionalGeneration

    # 1. Build 768 model from the base config (preserves vocab size / special tokens)
    base = TinyDocVLMForConditionalGeneration.from_pretrained(args.base, trust_remote_code=True)
    cfg = base.config
    cfg.image_size = 768
    cfg.vision_config.image_size = 768
    assert cfg.image_size == 768, f"expected 768, got {cfg.image_size}"
    model = TinyDocVLMForConditionalGeneration(cfg)

    # 2. Copy compatible weights from base (decoder + compressor)
    logger.info("Copying decoder + compressor from base")
    model.decoder.load_state_dict(base.decoder.state_dict())
    model.compressor.load_state_dict(base.compressor.state_dict())

    # 3. Interpolate vision-encoder position embeddings 384 -> 768 grid
    logger.info("Interpolating vision pos emb 384 -> 768")
    base.vision_encoder.resize_pos_embeddings(48)  # 48x48 = 768/16
    model.vision_encoder.encoder.load_state_dict(base.vision_encoder.encoder.state_dict())

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    cfg.save_pretrained(str(out))
    logger.info(f"Saved 768-initialized model to {out}")


if __name__ == "__main__":
    main()
