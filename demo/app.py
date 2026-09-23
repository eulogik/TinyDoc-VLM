#!/usr/bin/env python3
"""Local Gradio demo — ReceiptPipeline (free Ollama engine), evidence + overlay.

Usage:
  export PATH="/opt/homebrew/bin:$PATH"
  PYTHONPATH=sdk python demo/app.py --share --port 7860
  # or after pip install ./sdk:
  python demo/app.py --share

Requires local Ollama with qwen2.5vl:3b (`ollama pull qwen2.5vl:3b`).
No API key. Evidence boxes are OCR-span (Tesseract), not VLM boxes.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).absolute().parent.parent
SDK = ROOT / "sdk"
if str(SDK) not in sys.path:
    sys.path.insert(0, str(SDK))

EXAMPLE_DIR = Path(__file__).parent / "examples"


def build_demo():
    import gradio as gr

    from tinydoc import ReceiptPipeline, draw_overlay

    pipe = ReceiptPipeline("auto")

    def extract(image, with_evidence: bool, overlay: bool):
        if image is None:
            return (
                "Upload a receipt image.",
                "{}",
                None,
                "—",
                "—",
            )
        # gradio may give PIL or path
        if hasattr(image, "save"):
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            image.save(tmp.name)
            path = tmp.name
        else:
            path = str(image)

        try:
            doc = pipe.extract(path, with_evidence=with_evidence)
        except Exception as e:
            traceback.print_exc()
            return (f"Error: {e}", "{}", None, "—", "—")

        fields_json = json.dumps(doc.fields, indent=2, ensure_ascii=False)
        summary = (
            f"engine=`{doc.engine}` · schema_valid=`{doc.schema_valid}` · "
            f"confidence=`{doc.confidence:.3f}` · evidence_coverage=`{doc.evidence_coverage:.3f}` · "
            f"latency=`{doc.latency_ms:.0f} ms`"
        )
        detail_rows = [
            [
                fr.name,
                fr.value,
                f"{fr.confidence:.3f}",
                (fr.evidence or {}).get("kind", ""),
                ((fr.evidence or {}).get("quote") or "")[:120],
            ]
            for fr in doc.field_results
        ]
        overlay_path = None
        if overlay:
            try:
                out = Path(tempfile.mkstemp(suffix="_overlay.png")[1])
                draw_overlay(path, doc.field_results, out)
                overlay_path = str(out)
            except Exception as e:
                traceback.print_exc()
                overlay_path = None
        return summary, fields_json, overlay_path, detail_rows, doc.raw[:2000]

    examples = []
    if EXAMPLE_DIR.exists():
        for p in sorted(EXAMPLE_DIR.glob("*.png")):
            examples.append([str(p), True, True])

    with gr.Blocks(title="TinyDoc — local grounded extraction") as demo:
        gr.Markdown(
            """
# TinyDoc — local grounded receipt extraction
Schema-validated JSON · per-field OCR-span evidence · confidence · **no API key**.
Engine: free `ollama:qwen2.5vl:3b` (or OCR floor if Ollama is down).
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.Image(type="pil", label="Receipt image", height=420)
                with_evidence = gr.Checkbox(value=True, label="Attach evidence (Tesseract)")
                do_overlay = gr.Checkbox(value=True, label="Draw overlay PNG")
                btn = gr.Button("Extract", variant="primary")
                ex = gr.Examples(
                    examples=examples,
                    inputs=[image_in, with_evidence, do_overlay],
                    label="Examples",
                )
            with gr.Column(scale=1):
                summary = gr.Textbox(label="Summary", lines=2)
                fields_json = gr.Code(label="Fields (JSON)", language="json")
                overlay_out = gr.Image(label="Overlay", type="pil")
                detail = gr.Dataframe(
                    headers=["field", "value", "confidence", "evidence", "quote"],
                    label="Field detail",
                    wrap=True,
                )
                raw = gr.Textbox(label="Raw", lines=4)

        btn.click(
            fn=extract,
            inputs=[image_in, with_evidence, do_overlay],
            outputs=[summary, fields_json, overlay_out, detail, raw],
        )
        gr.Markdown(
            """
---
**Honesty notes**
- Evidence kind is always `ocr_span` (Tesseract word boxes), not VLM-predicted boxes.
- Schema-valid rate and field F1 are measured on a 100-doc SROIE holdout — see `evaluation/phase0/README.md`.
- Address field is the hardest (lookalike OCR confusions); confidence + evidence coverage surface HITL cases.
            """
        )
    return demo


def main() -> None:
    import gradio as gr

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--share", action="store_true", help="Create public Gradio link")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    demo = build_demo()
    demo.launch(
        share=args.share,
        server_port=args.port,
        server_name=args.host,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
