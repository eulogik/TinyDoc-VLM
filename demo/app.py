#!/usr/bin/env python3
"""
Gradio demo for TinyDoc-VLM.
Interactive document understanding with structured output extraction.

Usage:
    python demo/app.py --model-path checkpoints/best
"""

import argparse
import json
import tempfile
from pathlib import Path
from typing import Optional

import gradio as gr
import torch
from PIL import Image

from tinydoc_vlm import TinyDocVLMForConditionalGeneration, TinyDocVLMProcessor

EXAMPLE_DIR = Path(__file__).parent / "examples"
EXAMPLE_DIR.mkdir(exist_ok=True)


def load_model(model_path: str, device: Optional[str] = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyDocVLMForConditionalGeneration.from_pretrained(model_path)
    model.to(device)
    model.eval()
    processor = TinyDocVLMProcessor()
    return model, processor, device


def extract_document(image, question: str, model, processor, device) -> str:
    if image is None:
        return "Please upload a document image."

    text = f"{question} <image>"
    inputs = processor(text=text, images=image)

    with torch.no_grad():
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            num_beams=1,
        )

    result = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result


def extract_json(image, model, processor, device) -> str:
    return extract_document(
        image,
        "Extract all structured information from this document as JSON:",
        model, processor, device,
    )


def extract_kv(image, model, processor, device) -> str:
    return extract_document(
        image,
        "Extract all key-value pairs from this document:",
        model, processor, device,
    )


def extract_table(image, model, processor, device) -> str:
    return extract_document(
        image,
        "Extract any tables in this document as Markdown:",
        model, processor, device,
    )


def build_interface(model, processor, device):
    with gr.Blocks(title="TinyDoc-VLM Document Understanding", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 📄 TinyDoc-VLM
            ### The World's Smallest Document Understanding Model
            Upload a document image to extract structured information.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Document Image", height=400)
                question_input = gr.Textbox(
                    label="Question / Prompt",
                    value="Extract all structured information from this document as JSON:",
                    lines=2,
                )
                with gr.Row():
                    submit_btn = gr.Button("🔍 Extract", variant="primary", scale=2)
                    clear_btn = gr.Button("🗑️ Clear", scale=1)

            with gr.Column(scale=1):
                output = gr.Textbox(label="Extracted Result", lines=20, max_lines=40)

        gr.Examples(
            examples=[
                [str(EXAMPLE_DIR / "invoice.png"), "Extract invoice details as JSON:"],
                [str(EXAMPLE_DIR / "receipt.png"), "What is the total amount?"],
                [str(EXAMPLE_DIR / "table.png"), "Convert this table to Markdown:"],
            ] if any(EXAMPLE_DIR.iterdir()) else [],
            inputs=[image_input, question_input],
        )

        with gr.Accordion("⚡ Quick Actions", open=False):
            with gr.Row():
                json_btn = gr.Button("📋 Extract as JSON")
                kv_btn = gr.Button("🔑 Extract Key-Value")
                table_btn = gr.Button("📊 Extract Table")
                ocr_btn = gr.Button("📝 OCR Text")

        def process(image, question):
            return extract_document(image, question, model, processor, device)

        def process_json(image):
            return extract_json(image, model, processor, device)

        def process_kv(image):
            return extract_kv(image, model, processor, device)

        def process_table(image):
            return extract_table(image, model, processor, device)

        submit_btn.click(fn=process, inputs=[image_input, question_input], outputs=output)
        json_btn.click(fn=process_json, inputs=[image_input], outputs=output)
        kv_btn.click(fn=process_kv, inputs=[image_input], outputs=output)
        table_btn.click(fn=process_table, inputs=[image_input], outputs=output)

        def clear():
            return None, "", ""

        clear_btn.click(fn=clear, outputs=[image_input, question_input, output])

        gr.Markdown(
            """
            ---
            ### 💡 Tips
            - Upload scanned documents, photos of documents, or screenshots
            - For best results, ensure the document is well-lit and in focus
            - Use specific questions for better extraction accuracy
            - The model runs locally — no data leaves your device
            """
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="TinyDoc-VLM Demo")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to model checkpoint (default: creates a new model)")
    parser.add_argument("--device", type=str, default=None, help="Device to use")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on")
    args = parser.parse_args()

    if args.model_path:
        logger.info(f"Loading model from {args.model_path}")
        model, processor, device = load_model(args.model_path)
    else:
        logger.info("No model path provided. Creating fresh model (weights will be untrained).")
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        from tinydoc_vlm import TinyDocVLMConfig
        config = TinyDocVLMConfig()
        model = TinyDocVLMForConditionalGeneration(config)
        model.to(device)
        processor = TinyDocVLMProcessor()

    logger.info(f"Running on {device}")
    demo = build_interface(model, processor, device)
    demo.launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    main()
