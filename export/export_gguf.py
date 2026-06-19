#!/usr/bin/env python3
"""
Export TinyDoc-VLM decoder to GGUF format for llama.cpp / ollama inference.

Note: Only the text decoder portion can be exported this way.
The vision encoder requires a separate pipeline.

Usage:
    python export/export_gguf.py --model-path checkpoints/best --output tinydoc-vlm.gguf
"""

import argparse
import logging
from pathlib import Path

import torch
from tinydoc_vlm import TinyDocVLMForConditionalGeneration

logger = logging.getLogger(__name__)


def export_decoder_to_gguf(model_path: str, output_path: str):
    logger.info(f"Loading model from {model_path}...")
    model = TinyDocVLMForConditionalGeneration.from_pretrained(model_path)
    model.eval()

    logger.info("Extracting decoder weights...")
    decoder = model.decoder.lm
    state_dict = decoder.state_dict()

    output_path = Path(output_path)

    logger.info(f"Saving decoder weights to {output_path}...")
    torch.save(state_dict, output_path.with_suffix(".pt"))
    logger.info(f"Decoder weights saved. Use llama.cpp convert.py to create GGUF.")

    instructions = f"""
=== GGUF Export Instructions ===

1. Install llama.cpp:
   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp && make

2. Convert decoder weights:
   python llama.cpp/convert.py {output_path.with_suffix('.pt')} \\
       --outfile {output_path} \\
       --model-name tinydoc-vlm

3. For quantized versions:
   ./llama.cpp/quantize {output_path} {output_path.with_stem(output_path.stem + '_q4_0').with_suffix('.gguf')} q4_0

4. Run with llama.cpp:
   ./llama.cpp/main -m {output_path} -p "Extract: ..."

Note: For full vision-language inference, use the Python SDK with ONNX runtime.
"""
    print(instructions)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Export TinyDoc-VLM decoder to GGUF")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="tinydoc-vlm.gguf", help="Output GGUF path")
    args = parser.parse_args()
    export_decoder_to_gguf(args.model_path, args.output)
