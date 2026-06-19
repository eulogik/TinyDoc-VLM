#!/usr/bin/env python3
"""
Training launcher for TinyDoc-VLM.
Loads a YAML config, initializes the model/dataset/trainer, and runs training.

Usage:
    python training/run.py --config training/stage1_layout_pretrain.yaml
"""

import argparse
import yaml
import logging
import os
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import random_split

from tinydoc_vlm import (
    TinyDocVLMConfig,
    TinyDocVLMForConditionalGeneration,
    TinyDocVLMProcessor,
    TinyDocVLMTrainer,
    TrainerConfig,
    DocumentDataset,
)

logger = logging.getLogger(__name__)


def load_config(path: str) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f)


def create_model(cfg: Dict) -> TinyDocVLMForConditionalGeneration:
    model_cfg = cfg.get("model", {})
    config = TinyDocVLMConfig(
        vision_config=model_cfg.get("vision_config"),
        decoder_config=model_cfg.get("decoder_config"),
        pixel_shuffle_scale=model_cfg.get("pixel_shuffle_scale", 3),
        image_size=model_cfg.get("image_size", 384),
        patch_size=model_cfg.get("patch_size", 16),
    )
    model = TinyDocVLMForConditionalGeneration(config)

    pretrained = model_cfg.get("pretrained_checkpoint")
    if pretrained and os.path.exists(pretrained):
        logger.info(f"Loading pretrained checkpoint: {pretrained}")
        model = TinyDocVLMForConditionalGeneration.from_pretrained(pretrained)

    return model


def create_dataset(cfg: Dict) -> DocumentDataset:
    data_cfg = cfg.get("data", {})
    return DocumentDataset(
        data_root=data_cfg.get("data_root", "data/synthetic"),
        manifest_path=data_cfg.get("manifest_path"),
        max_seq_length=data_cfg.get("max_seq_length", 2048),
        stage=data_cfg.get("stage", 1),
    )


def main():
    parser = argparse.ArgumentParser(description="Train TinyDoc-VLM")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML training config")
    parser.add_argument("--device", type=str, default=None, help="Device override")
    parser.add_argument("--override", type=str, nargs="*", default=[], help="Override config values: key=value")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    cfg = load_config(args.config)
    for override in args.override:
        key, value = override.split("=", 1)
        keys = key.split(".")
        d = cfg
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = yaml.safe_load(value)

    model = create_model(cfg)
    processor = TinyDocVLMProcessor()
    dataset = create_dataset(cfg)

    train_size = int(0.9 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

    train_cfg_data = cfg.get("training", {})
    trainer_cfg = TrainerConfig(
        output_dir=train_cfg_data.get("output_dir", "checkpoints"),
        num_epochs=train_cfg_data.get("num_epochs", 3),
        batch_size=train_cfg_data.get("batch_size", 8),
        gradient_accumulation_steps=train_cfg_data.get("gradient_accumulation_steps", 4),
        learning_rate=train_cfg_data.get("learning_rate", 1e-4),
        min_learning_rate=train_cfg_data.get("min_learning_rate", 1e-5),
        warmup_steps=train_cfg_data.get("warmup_steps", 500),
        weight_decay=train_cfg_data.get("weight_decay", 0.01),
        max_grad_norm=train_cfg_data.get("max_grad_norm", 1.0),
        max_seq_length=train_cfg_data.get("max_seq_length", 2048),
        stage=train_cfg_data.get("stage", 1),
        use_fp16=train_cfg_data.get("use_fp16", True),
        save_every_steps=train_cfg_data.get("save_every_steps", 1000),
        eval_every_steps=train_cfg_data.get("eval_every_steps", 500),
        log_every_steps=train_cfg_data.get("log_every_steps", 10),
        gradient_checkpointing=train_cfg_data.get("gradient_checkpointing", True),
    )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    trainer = TinyDocVLMTrainer(
        model=model,
        processor=processor,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=trainer_cfg,
        device=device,
    )

    trainer.train()


if __name__ == "__main__":
    main()
