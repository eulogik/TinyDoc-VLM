#!/usr/bin/env python3
"""
Export TinyDoc-VLM to ONNX format with dynamic axes for production inference.

Usage:
    python export/export_onnx.py --model-path checkpoints/best --output model.onnx
"""

import argparse
import logging
from pathlib import Path

import torch
from tinydoc_vlm import TinyDocVLMForConditionalGeneration

logger = logging.getLogger(__name__)


def export_to_onnx(model_path: str, output_path: str, opset: int = 17):
    logger.info(f"Loading model from {model_path}...")
    model = TinyDocVLMForConditionalGeneration.from_pretrained(model_path)
    model.eval()

    device = torch.device("cpu")
    model.to(device)

    # Create dummy inputs for tracing
    B, N, C, H, W = 1, 1, 3, 384, 384
    pixel_values = torch.randn(B, N, C, H, W)
    seq_len = 128
    input_ids = torch.randint(0, 100, (B, seq_len))
    attention_mask = torch.ones(B, seq_len, dtype=torch.long)

    output_path = Path(output_path)

    logger.info(f"Exporting to ONNX (opset {opset})...")
    torch.onnx.export(
        model,
        args=(input_ids, pixel_values, attention_mask),
        f=str(output_path),
        opset_version=opset,
        input_names=["input_ids", "pixel_values", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "pixel_values": {0: "batch_size", 1: "num_tiles"},
            "attention_mask": {0: "batch_size", 1: "seq_len"},
            "logits": {0: "batch_size", 1: "seq_len"},
        },
    )

    logger.info(f"ONNX model saved to {output_path}")
    logger.info(f"Model size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Export TinyDoc-VLM to ONNX")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="tinydoc-vlm.onnx", help="Output ONNX path")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    args = parser.parse_args()
    export_to_onnx(args.model_path, args.output, args.opset)
