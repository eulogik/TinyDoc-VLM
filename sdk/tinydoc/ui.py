"""Local Gradio UI for TinyDoc.

The UI is deliberately local: it talks to the same Ollama server and the same
SDK the CLI uses. It is not a hosted document service, and it does not upload
anything to a third party. Start it with::

    pip install 'tinydoc[ui]'
    tinydoc ui

Receipt tab: single image, four fields, OCR-span evidence, optional overlay.
Invoice tab: one PDF (all pages) or a list of page images, page-wise extraction,
deterministic merge, arithmetic review flags, and optional OCR totals.

Only the receipt field-confidence scores are calibrated (and only for review
routing). Invoice confidence is intentionally not shown as a probability; the
invoice tab shows evidence match quality and review flags instead.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, List, Optional, Tuple

RECEIPT_ENGINES = [
    "ollama:qwen2.5vl:3b",
    "ollama:qwen3-vl:2b-instruct",
]
INVOICE_ENGINES = [
    "ollama:qwen3-vl:2b-instruct",
    "ollama:qwen2.5vl:3b",
]

HEADER = """
# TinyDoc — cited document extraction that runs locally

Schema-valid JSON · OCR-span evidence · arithmetic review flags · Apache 2.0

This UI runs on **your machine**. It calls a local Ollama server and writes
nothing to a cloud service. If Ollama is not running, extraction will fail with
a clear error; it will not silently upload the document.

**Measured, not implied.** Receipt field F1 is 0.9475 (Qwen2.5-VL-3B) and
0.9200 (Qwen3-VL-2B) on SROIE clean-100, a public-train-derived set that is not
an external benchmark. The invoice multipage probe is 10 synthetic documents:
page-wise VLM gets 6/10 fully correct; adding deterministic OCR totals reaches
10/10 on nine key-header fields, but that is a **pipeline result, not a pure VLM
score**, and line items remain weak. Real supplier invoices are unmeasured.
"""

CSS = """
html, body { background: #F4F6F2 !important; color-scheme: light !important; }
body.dark, .dark {
  --body-background-fill: #F4F6F2 !important;
  --background-fill-primary: #F4F6F2 !important;
  --background-fill-secondary: #E7ECE7 !important;
  --block-background-fill: #FBFCFA !important;
  --block-border-color: #C9D2CC !important;
  --block-label-background-fill: #E7ECE7 !important;
  --block-label-text-color: #4A5A64 !important;
  --block-title-text-color: #4A5A64 !important;
  --block-info-text-color: #4A5A64 !important;
  --body-text-color: #0E1A24 !important;
  --body-text-color-subdued: #4A5A64 !important;
  --border-color-primary: #C9D2CC !important;
  --border-color-accent: #1E6F5C !important;
  --input-background-fill: #FFFFFF !important;
  --input-border-color: #C9D2CC !important;
  --table-odd-background-fill: #FBFCFA !important;
  --table-even-background-fill: #F4F6F2 !important;
  --color-accent: #1E6F5C !important;
  --color-accent-soft: #DCE8E2 !important;
  --button-secondary-background-fill: #E7ECE7 !important;
  --button-secondary-text-color: #0E1A24 !important;
}
:root {
  --ink: #0E1A24;
  --paper: #F4F6F2;
  --rule: #C9D2CC;
  --green: #1E6F5C;
  --oxide: #B23A2E;
  --muted: #4A5A64;
}
gradio-app { width: 100% !important; }
.gradio-container {
  width: 100% !important;
  background: var(--paper) !important;
  color: var(--ink) !important;
  max-width: 1180px !important;
  margin: 0 auto !important;
  padding: 0 22px 44px !important;
  font-family: "Avenir Next", "Helvetica Neue", Arial, sans-serif !important;
}
.gradio-container::before {
  content: "";
  display: block;
  height: 6px;
  background: var(--ink);
  margin: 0 -22px 30px;
}
.gradio-container main.contain {
  width: 100% !important;
  max-width: 1180px !important;
  margin: 0 auto !important;
  padding: 0 !important;
}
.gradio-container main.contain > .wrap { padding: 0 !important; }
.gradio-container .prose { max-width: 80ch; }
.gradio-container .prose h1 {
  font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif !important;
  font-size: clamp(1.95rem, 3.6vw, 2.9rem) !important;
  line-height: 1.04 !important;
  letter-spacing: -0.035em !important;
  font-weight: 600 !important;
  margin: 0 0 0.4rem !important;
}
.gradio-container .prose h1 + p { color: var(--muted); font-size: 1.02rem; }
.gradio-container .prose h2,
.gradio-container .prose h3 {
  font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif !important;
  letter-spacing: -0.02em;
}
.gradio-container .prose hr { border: 0; border-top: 1px solid var(--rule); margin: 2rem 0; }
.gradio-container .tabs { border-bottom: 1px solid var(--rule) !important; }
.gradio-container .tab-nav button {
  border-radius: 0 !important;
  font-family: "Avenir Next", "Helvetica Neue", Arial, sans-serif !important;
  font-size: 0.92rem !important;
  color: var(--muted) !important;
}
.gradio-container .tab-nav button.selected {
  color: var(--ink) !important;
  border-bottom: 3px solid var(--green) !important;
}
.gradio-container button {
  border-radius: 2px !important;
  box-shadow: none !important;
  font-family: "Avenir Next", "Helvetica Neue", Arial, sans-serif !important;
}
.gradio-container button.primary,
.gradio-container .primary {
  background: var(--ink) !important;
  color: var(--paper) !important;
  border: 1px solid var(--ink) !important;
}
.gradio-container button.primary:hover {
  background: var(--green) !important;
  border-color: var(--green) !important;
}
.gradio-container input,
.gradio-container textarea,
.gradio-container .wrap,
.gradio-container .file-preview {
  border-radius: 2px !important;
}
.gradio-container table {
  border-collapse: collapse !important;
  font-size: 0.84rem !important;
}
.gradio-container table th {
  background: #E7ECE7 !important;
  color: var(--ink) !important;
  font-weight: 600 !important;
  border-bottom: 1px solid var(--rule) !important;
}
.gradio-container table td { border-bottom: 1px solid #DDE3DD !important; }
.gradio-container .label-wrap span { color: var(--muted) !important; font-size: 0.78rem !important; }
.gradio-container label,
.gradio-container .label-wrap,
.gradio-container .block-label { color: #4A5A64 !important; opacity: 1 !important; }
.gradio-container .label-wrap span,
.gradio-container label span { color: #4A5A64 !important; opacity: 1 !important; }
.gradio-container footer { display: none !important; }
"""

FOOTER = """
---
**What this is not:** a straight-through accounts-payable system, a hosted
service, or a TinyDoc-trained checkpoint. The reader is a third-party Qwen base
model; the product is the trust layer around it. Invoice line items and flags
need human review. See the repository's `docs/BENCHMARKS.md` for the full
evidence and caveats.
"""


def _error(message: str) -> dict:
    return {"ok": False, "error": message}


def _receipt_rows(result) -> List[List[Any]]:
    rows = []
    for field in result.field_results:
        evidence = field.evidence or {}
        rows.append(
            [
                field.name,
                field.value,
                round(float(field.confidence), 3),
                evidence.get("quote", ""),
                evidence.get("page", ""),
            ]
        )
    return rows


def _receipt_overlay(result, image_path: str) -> Optional[str]:
    try:
        from tinydoc.overlay import draw_overlay

        out_dir = Path(tempfile.mkdtemp(prefix="tinydoc-overlay-"))
        out = out_dir / (Path(image_path).stem + "-overlay.png")
        draw_overlay(image_path, result.field_results, out)
        return str(out)
    except Exception:  # noqa: BLE001 - the overlay is optional
        return None


def _invoice_flags(result) -> str:
    if not result.review_flags:
        return "No arithmetic review flags."
    lines = ["**Review flags (arithmetic, no gold answers):**", ""]
    for issue in result.arithmetic_issues:
        lines.append(
            f"- `{issue['check']}`: expected {issue['expected']}, "
            f"computed {issue['computed']}, difference {issue['difference']}"
        )
    for flag in result.review_flags:
        if flag not in {issue["check"] for issue in result.arithmetic_issues}:
            lines.append(f"- `{flag}`")
    if result.corrections:
        lines.extend(["", "**Recorded corrections:**", ""])
        lines.extend(f"- `{correction}`" for correction in result.corrections)
    return "\n".join(lines)


def _gradio_major(gr) -> int:
    try:
        return int(str(gr.__version__).split(".")[0])
    except (AttributeError, ValueError):  # pragma: no cover - defensive
        return 4


def build_app():
    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise SystemExit(
            "Gradio is not installed. Install the UI extra: pip install 'tinydoc[ui]'"
        ) from exc

    from tinydoc.invoice import InvoicePipeline
    from tinydoc.pipeline import ReceiptPipeline

    def run_receipt(image_path, engine, evidence):
        if not image_path:
            return _error("Choose a receipt image first."), [], None
        try:
            pipeline = ReceiptPipeline(engine)
            result = pipeline.extract(image_path, with_evidence=bool(evidence))
            return result.to_dict(), _receipt_rows(result), _receipt_overlay(
                result, image_path
            )
        except Exception as exc:  # noqa: BLE001 - show the failure in the UI
            return _error(f"{type(exc).__name__}: {exc}"), [], None

    def run_invoice(files, engine, ocr_totals):
        if not files:
            return _error("Choose a PDF or page images first."), [], ""
        if isinstance(files, (str, Path)):
            files = [files]
        paths = [str(path) for path in files if path]
        if not paths:
            return _error("No readable file paths."), [], ""
        try:
            pipeline = InvoicePipeline(engine)
            # One PDF is handled page-by-page by the pipeline. Multiple images
            # are treated as the ordered pages of one invoice.
            source = paths[0]
            result = pipeline.extract(
                source,
                images=paths if len(paths) > 1 else None,
                with_ocr_totals=bool(ocr_totals),
            )
            rows = [
                [index + 1, item.get("description", ""), item.get("amount")]
                for index, item in enumerate(result.line_items)
            ]
            return result.to_dict(), rows, _invoice_flags(result)
        except Exception as exc:  # noqa: BLE001 - show the failure in the UI
            return _error(f"{type(exc).__name__}: {exc}"), [], ""

    # Gradio 6 moved css/theme from Blocks() to launch(). Keep both paths so
    # the SDK works on the older 4.x/5.x releases too.
    blocks_kwargs = {}
    if _gradio_major(gr) < 6:
        blocks_kwargs = {"css": CSS, "theme": gr.themes.Base()}
    with gr.Blocks(title="TinyDoc", **blocks_kwargs) as demo:
        gr.Markdown(HEADER)

        with gr.Tab("Receipt (single image)"):
            with gr.Row():
                receipt_image = gr.Image(
                    type="filepath", label="Receipt image", height=360
                )
                with gr.Column():
                    receipt_engine = gr.Dropdown(
                        RECEIPT_ENGINES,
                        value=RECEIPT_ENGINES[0],
                        label="Engine",
                    )
                    receipt_evidence = gr.Checkbox(
                        value=True, label="Attach OCR-span evidence"
                    )
                    receipt_button = gr.Button("Extract receipt", variant="primary")
            receipt_json = gr.JSON(label="Result")
            receipt_table = gr.Dataframe(
                headers=["field", "value", "confidence", "quote", "page"],
                datatype=["str", "str", "number", "str", "number"],
                label="Fields and evidence (confidence is for review routing, not a probability)",
                interactive=False,
                wrap=True,
            )
            receipt_overlay = gr.Image(
                label="Evidence overlay", height=420, interactive=False
            )
            receipt_button.click(
                run_receipt,
                inputs=[receipt_image, receipt_engine, receipt_evidence],
                outputs=[receipt_json, receipt_table, receipt_overlay],
            )

        with gr.Tab("Invoice (PDF or page images)"):
            with gr.Row():
                invoice_files = gr.File(
                    file_count="multiple",
                    type="filepath",
                    label="One PDF, or the ordered page images of one invoice",
                )
                with gr.Column():
                    invoice_engine = gr.Dropdown(
                        INVOICE_ENGINES,
                        value=INVOICE_ENGINES[0],
                        label="Engine",
                    )
                    invoice_ocr = gr.Checkbox(
                        value=True,
                        label="Reconcile printed totals with OCR (deterministic)",
                    )
                    invoice_button = gr.Button("Extract invoice", variant="primary")
            invoice_json = gr.JSON(label="Result")
            invoice_lines = gr.Dataframe(
                headers=["#", "description", "amount"],
                datatype=["number", "str", "number"],
                label="Line items (weakest field; review before posting)",
                interactive=False,
                wrap=True,
            )
            invoice_flags = gr.Markdown()
            invoice_button.click(
                run_invoice,
                inputs=[invoice_files, invoice_engine, invoice_ocr],
                outputs=[invoice_json, invoice_lines, invoice_flags],
            )

        gr.Markdown(FOOTER)

    return demo


def launch(host: str = "127.0.0.1", port: int = 7860, share: bool = False) -> None:
    """Launch the local UI. ``share=True`` exposes a public Gradio link."""
    import gradio as gr

    demo = build_app()
    if share:
        print(
            "WARNING: --share exposes this app through a public Gradio link. "
            "Documents leave this machine. Use only with non-sensitive files."
        )
    launch_kwargs = {
        "server_name": host,
        "server_port": port,
        "share": share,
        "show_error": True,
    }
    if _gradio_major(gr) >= 6:
        launch_kwargs.update({"css": CSS, "theme": gr.themes.Base()})
    demo.launch(**launch_kwargs)


__all__ = ["build_app", "launch"]
