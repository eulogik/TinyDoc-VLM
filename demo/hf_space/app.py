"""Static TinyDoc project card for Hugging Face Spaces.

This Space deliberately does NOT load the retired 256M checkpoint. Free CPU
Spaces cannot serve the measured 2B/3B engines at usable latency, so the card
explains how to run them locally and links to the artifact-backed results.
"""

from __future__ import annotations

import os

import gradio as gr

CARD = """
# TinyDoc — cited document extraction that runs locally

**Schema-valid JSON · exact OCR-span evidence · field confidence · Apache 2.0**

TinyDoc is an open-source Python SDK and CLI that turns receipt and invoice
images into schema-validated JSON, with the page, span, and a confidence score
behind every field. It runs local Qwen2.5-VL-3B or Qwen3-VL-2B-Instruct models
through Ollama. It is **not** a fine-tuned TinyDoc checkpoint; the reader is a
third-party base model and the product is the trust layer around it.

![TinyDoc architecture](https://raw.githubusercontent.com/eulogik/TinyDoc-VLM/main/docs/assets/architecture.png)

## Measured results

![Measured results by tier](https://raw.githubusercontent.com/eulogik/TinyDoc-VLM/main/docs/assets/tiers.png)

| Benchmark | 3B | 2B |
|---|---:|---:|
| SROIE clean-100 — field F1 | **0.9475** | **0.9200** |
| SROIE clean-100 — schema valid | 1.00 | 1.00 |
| SROIE clean-100 — p50 latency | 14.1 s | **7.0 s** |
| Invoice v1, page 1 — key header | 91.9% | **93.5%** |
| Invoice v1, page 1 — fully correct | 65.7% | **71.8%** |
| Invoice v1, page 1 — p50 latency | 26.5 s | **16.1 s** |
| Invoice multipage, all pages — fully correct (n=10, 46 pages) | — | **4/10** vs 0/10 page 1 |
| Invoice multipage, all pages — key-header (n=10) | — | **71.1%** vs 63.3% page 1 |
| Invoice multipage, page-wise — fully correct (n=10, 46 pages) | — | **6/10** · **93.3%** key-header |
| Invoice multipage, page-wise — line-item F1 | — | **0.848** · 0 parse errors · ~2× slower |
| DocVQA reading — ANLS / exact (200 public, single-reference) | — | **0.829 / 0.765** |

> **Not an external benchmark.** SROIE clean-100 is a duplicate-filtered split of
> the public SROIE **train** pool; Qwen may have seen it during pretraining.
> The original 100-doc set is 60% leaked. The invoice set is synthetic,
> vendor-made, and page 1 only; real supplier invoices are unmeasured and
> flagging fails (0/7). The multipage probes are 10 synthetic documents at
> 1400 px, 2B only; all-pages gets 4/10, page-wise 6/10 and 93.3% key-header,
> with zero parse errors but ~2× latency. Line items remain weak, three NAD
> invoices are read as ZAR, and the SDK path remains single-image. DocVQA is 200
> public images, one question per image, 768 px mirror, single reference; it is
> not comparable to leaderboard numbers. These are same-document engine
> comparisons, not generalization claims.

## What you get back

![TinyDoc result table with evidence and confidence](https://raw.githubusercontent.com/eulogik/TinyDoc-VLM/main/docs/assets/trust.png)

Every non-empty field carries `{quote, page, bbox, confidence}`. Evidence
coverage is 0.798 of non-empty fields. Confidence is for **review routing**, not
a probability:

- Field ECE is **0.115–0.135** raw and **0.005–0.014** after document-grouped
  isotonic out-of-fold calibration.
- Document-level confidence is **not** calibrated (ECE 0.09–0.10).
- Boxes are OCR spans. VLM-predicted boxes measured hit@0.5 of **0.04–0.20**
  against a 0.70 gate, so they are not claimed or shipped.

## Run it locally

```bash
pip install tinydoc==0.2.0

ollama pull qwen2.5vl:3b            # quality tier
ollama pull qwen3-vl:2b-instruct    # fast tier
```

```python
from tinydoc import ReceiptPipeline

pipe = ReceiptPipeline("ollama:qwen2.5vl:3b")
result = pipe.extract("receipt.jpg")

print(result.fields, result.schema_valid)
for field in result.field_results:
    print(field.name, field.value, field.confidence, field.evidence)
```

```bash
tinydoc extract ./receipts/ --engine ollama:qwen2.5vl:3b \\
    --out results.jsonl --overlay overlays/
```

## Why this card, not a live demo?

The previous Space loaded `eulogik/TinyDoc-VLM-256M`, which measured **0.0% on
OCRBench (n=999 scored)** and is retired from all performance claims. Serving it
as a live demo would misrepresent the project. The working 2B/3B engines need
several GB of RAM and a local Ollama runtime; they are not suitable for this
free CPU Space.

## Read more

- [Source and install](https://github.com/eulogik/TinyDoc-VLM)
- [System card](https://github.com/eulogik/TinyDoc-VLM/blob/main/docs/MODEL_CARD.md)
- [Benchmarks and caveats](https://github.com/eulogik/TinyDoc-VLM/blob/main/docs/BENCHMARKS.md)
- [Plan and evidence ledger](https://github.com/eulogik/TinyDoc-VLM/blob/main/docs/v3_plan.md)
- [Live site](https://eulogik.github.io/TinyDoc-VLM/)
"""

with gr.Blocks(title="TinyDoc — cited document extraction") as demo:
    gr.Markdown(CARD)
    gr.Markdown(
        "---\n**By** [eulogik](https://eulogik.com) · "
        "[GitHub](https://github.com/eulogik/TinyDoc-VLM) · "
        "[PyPI](https://pypi.org/project/tinydoc/) · Apache 2.0"
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        show_error=True,
    )
