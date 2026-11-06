"""Static TinyDoc project card for Hugging Face Spaces.

This Space deliberately does NOT load the retired 256M checkpoint. Free CPU
Spaces cannot serve the measured 2B/3B engines at usable latency, so the card
explains how to run them locally and links to the artifact-backed results.
"""

from __future__ import annotations

import os

import gradio as gr

CARD = """
# 📄 TinyDoc — local-first grounded document extraction

**Schema-valid JSON · per-field OCR evidence · confidence · Apache 2.0**

This Space is a static project card. The working engines are local Ollama
models; free CPU hardware cannot serve them at usable latency.

## Measured (SROIE public-train-derived clean-100, n=100)

| Engine | Field F1 | Address | Schema | Median |
|---|---:|---:|---:|---:|
| **Qwen2.5-VL-3B** — quality | **0.9475** | 0.86 | 1.00 | 14.1 s |
| **Qwen3-VL-2B-Instruct** — fast | **0.9200** | 0.84 | 1.00 | **7.0 s** |
| Hybrid v2 — first-party | 0.603 | 0.37 | 0.93 | ~8 s |
| PP-OCR + heuristics | 0.376 | 0.03 | — | — |

> ⚠️ **Not an external benchmark.** The 100-doc set is a duplicate-filtered
> split of the public SROIE **train** pool; Qwen may have seen it during
> pretraining. The original set is 60% leaked. These are same-document engine
> comparisons, not generalization claims.

## Run it locally

```bash
pip install "tinydoc @ git+https://github.com/eulogik/TinyDoc-VLM.git#subdirectory=sdk"
ollama pull qwen2.5vl:3b            # quality
ollama pull qwen3-vl:2b-instruct    # fast tier
```

```python
from tinydoc import ReceiptPipeline

pipe = ReceiptPipeline("ollama:qwen2.5vl:3b")
result = pipe.extract("receipt.jpg")
print(result.fields, result.schema_valid, result.confidence)
for field in result.field_results:
    print(field.name, field.value, field.evidence)
```

## Trust layer status

- **Evidence is OCR-span**, not VLM-predicted boxes. A 100-box probe measured
  hit@0.5 of **0.04–0.20** against a 0.70 gate, so boxes are not claimed yet.
- **Confidence is field-level calibrated only.** Evidence-on field ECE is
  **0.115–0.135** raw and **0.005–0.014** after document-grouped isotonic
  cross-validation. Document-level exact-4 confidence is **not** calibrated
  (ECE 0.09–0.10); do not treat it as a probability.
- Grounding and calibration artifacts are linked from the
  [v3 plan](https://github.com/eulogik/TinyDoc-VLM/blob/main/docs/v3_plan.md).

## Invoice probe (synthetic, page 1)

A public synthetic benchmark (Invoice Extraction Benchmark v1, 181 docs) measures
**90.6% key-header field accuracy** but only **55.8% documents fully correct**.
Flagging is **0/7** on inconsistent/multi-invoice files, and multipage is 0% fully
correct because only page 1 is rendered. The set is synthetic and vendor-made;
this is not real supplier-invoice accuracy.

## Why this card, not a live demo?

The previous Space loaded `eulogik/TinyDoc-VLM-256M`, which measured **0.0% on
OCRBench (n=999 scored)** and is retired from all performance claims. Serving it
as a live demo would misrepresent the project. The working 2B/3B engines need
several GB of RAM and a local Ollama runtime; they are not suitable for this
free CPU Space.

## Links

- [Source + install](https://github.com/eulogik/TinyDoc-VLM)
- [v3 plan and evidence](https://github.com/eulogik/TinyDoc-VLM/blob/main/docs/v3_plan.md)
- [Benchmarks](https://github.com/eulogik/TinyDoc-VLM/blob/main/docs/BENCHMARKS.md)
"""

with gr.Blocks(title="TinyDoc — local-first document extraction") as demo:
    gr.Markdown(CARD)
    gr.Markdown(
        "---\n**By** [eulogik](https://eulogik.com) · "
        "[GitHub](https://github.com/eulogik/TinyDoc-VLM) · "
        "[PyPI](https://pypi.org/project/tinydoc/)"
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        show_error=True,
    )
