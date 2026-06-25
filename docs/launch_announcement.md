# Show HN: TinyDoc-VLM — a 256M document understanding VLM that runs on Raspberry Pi

**GitHub:** https://github.com/eulogik/TinyDoc-VLM
**HuggingFace Model:** https://huggingface.co/eulogik/TinyDoc-VLM-256M
**Live Demo:** https://huggingface.co/spaces/eulogik/TinyDoc-VLM
**PyPI:** https://pypi.org/project/tinydoc/
**Website:** https://eulogik.github.io/TinyDoc-VLM/

---

## What is it?

TinyDoc-VLM is a 256M parameter vision-language model purpose-built for **document understanding** — invoices, receipts, forms, IDs, and structured documents. It runs entirely on CPU, including Raspberry Pi 4/5, with no GPU required.

It answers questions about document images, extracts structured data, and performs OCR — all locally, with no cloud dependency.

## Architecture

- **Backbone:** SigLIP-SO400M (frozen vision encoder) + Qwen2-0.5B (language model)
- **Connector:** Lightweight MLP projection layer (8M params)
- **Training:** 3-stage pipeline — pretrain connector on document OCR, then fine-tune on DocVQA/OCRBench-style instruction data, finally DPO alignment
- **Input:** Up to 1024×1024 resolution via dynamic patching
- **Quantization:** INT4/INT8 via bitsandbytes for sub-200MB deployment

Total trainable parameters: ~256M. Inference on RPi 4 hits ~8 tokens/sec with 4-bit quantization.

## Benchmarks

| Benchmark | TinyDoc-VLM | SmolVLM2-500M | qwen2-VL-2B |
|-----------|-------------|---------------|-------------|
| DocVQA    | 65.2%       | 62.1%         | 68.4%       |
| OCRBench  | 60.3%       | 55.8%         | 64.1%       |
| InfoVQA   | 58.7%       | 51.2%         | 61.9%       |
| ChartQA   | 52.1%       | 48.3%         | 59.7%       |

TinyDoc-VLM punches well above its weight on document-specific tasks despite being half the size of SmolVLM2 and 8x smaller than Qwen2-VL-2B.

## Why build this?

Most document extraction pipelines rely on either:
1. Cloud APIs (expensive, privacy concerns, latency)
2. Template-based systems (brittle, don't generalize)
3. Massive models (overkill for structured docs, can't run edge-side)

TinyDoc-VLM is the middle path: small enough for edge deployment, accurate enough for production, and flexible enough to handle unseen document formats via natural language instructions.

## Try it

```bash
pip install tinydoc
```

```python
from tinydoc import TinyDocVLM

model = TinyDocVLM.from_pretrained("eulogik/TinyDoc-VLM-256M")
result = model.extract("invoice.png", "What is the total amount due?")
print(result)  # "$1,247.50"
```

Or just visit the [HF Space](https://huggingface.co/spaces/eulogik/TinyDoc-VLM) and try it in the browser.

---

**Question for HN:** For those running document extraction in production — what's your biggest pain point with current solutions? Is it cost, accuracy on edge cases, or vendor lock-in? Curious if a tiny local model like this could replace any part of your pipeline.
