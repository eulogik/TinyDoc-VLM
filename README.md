<div align="center">
  <h1>TinyDoc-VLM</h1>
  <p><b>256M-Parameter Document-Specialist Vision-Language Model</b></p>
  <p>SigLIP-B/16 + PixelShuffle 3× + SmolLM2-135M · OCR, VQA, Form Extraction, Table Parsing</p>
  <p>Apache 2.0 · Runs on CPU · <1GB VRAM · LoRA Fine-tuning</p>

[![PyPI](https://img.shields.io/pypi/v/tinydoc?color=blue&label=tinydoc)](https://pypi.org/project/tinydoc/)
[![HF Model](https://img.shields.io/badge/🤗-Model%20Hub-yellow)](https://huggingface.co/eulogik/TinyDoc-VLM-256M)
[![HF LoRA](https://img.shields.io/badge/🤗-LoRA%20Adapter-yellow)](https://huggingface.co/eulogik/TinyDoc-VLM-LoRA)
[![HF Space](https://img.shields.io/badge/🤗-Live%20Demo-yellow)](https://huggingface.co/spaces/eulogik/TinyDoc-VLM)
[![GitHub](https://img.shields.io/badge/GitHub-Source-181717?logo=github)](https://github.com/eulogik/TinyDoc-VLM)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/eulogik/TinyDoc-VLM/actions/workflows/ci.yml/badge.svg)](https://github.com/eulogik/TinyDoc-VLM/actions)

---

**Built by [eulogik](https://eulogik.com)** — AI infrastructure for document intelligence.

[PyPI](https://pypi.org/project/tinydoc/) · [Model Hub](https://huggingface.co/eulogik/TinyDoc-VLM-256M) · [LoRA Checkpoint](https://huggingface.co/eulogik/TinyDoc-VLM-LoRA) · [Live Demo](https://huggingface.co/spaces/eulogik/TinyDoc-VLM)
</div>

## What is TinyDoc-VLM?

TinyDoc-VLM is an open-source **document understanding AI** that reads invoices, receipts, forms, tables, and charts. At just **256M parameters**, it runs on a MacBook, Raspberry Pi 5, or any CPU — no GPU required.

**Use cases:** Invoice processing, receipt scanning, form data extraction, table parsing, document Q&A, OCR, visual question answering.

## Highlights

- **256M params** — SigLIP-B/16 vision encoder (93M) + PixelShuffle 3× compressor + SmolLM2-135M decoder
- **<1GB VRAM** — Runs on MacBook Air, Raspberry Pi 5, or any CPU with ONNX
- **Structured output** — JSON extraction, key-value pairs, table parsing, OCR, VQA
- **LoRA fine-tuning** — Train on your own docs with 2.7M trainable params (0.93% of total)
- **Apache 2.0** — Fully open-source, free for commercial use
- **ONNX export** — Deploy anywhere with ONNX Runtime

## Quick Start

### Install

```bash
pip install tinydoc
```

### Python SDK

```python
from PIL import Image
from tinydoc import TinyDocExtractor

extractor = TinyDocExtractor(device="cpu")

# Ask a question about a document
img = Image.open("invoice.png")
result = extractor.ask(img, "What is the total?")
print(result.answer)  # "$1,234.56"

# Extract structured JSON
result = extractor.extract(img, output_format="json")
print(result.fields)  # {"total": "$1,234.56", "date": "2024-01-15", ...}

# Extract tables
result = extractor.extract_table(img)
print(result.markdown)  # Markdown-formatted table
```

### Direct Model Access

```python
from tinydoc_vlm import TinyDocVLMForConditionalGeneration, TinyDocVLMProcessor

model = TinyDocVLMForConditionalGeneration.from_pretrained("eulogik/TinyDoc-VLM-256M")
processor = TinyDocVLMProcessor()
```

## Model Architecture

```
Image (384×384)
    ↓
SigLIP Vision Encoder (93M)          ← 576 patches × 768 dim
    ↓
Pixel-Shuffle Compressor (scale=3)   ← 9× compression → 64 tokens
    ↓
Visual Position Embeddings
    ↓
SmolLM2 Decoder (135M)               ← 30 layers, GQA (9:3 heads), 8192 ctx
    ↓
Multi-Task Output Heads
    ↓
JSON / KV Extraction / Table / OCR / QA
```

**Total: 256M parameters** | Vision: 93M | Compressor: 3M | Decoder: 135M | Heads: 25M

## LoRA Fine-tuning

Train TinyDoc-VLM on your own documents using LoRA. Only **2.7M params** (0.93%) are trained.

### M4 Mac (overnight run)

```bash
# Generate 3K synthetic documents
python data/synthetic/generator.py --num-docs 3000 --output-dir data/synthetic/output

# Train for 17K steps (~15 hours on M4)
python training/fast_train.py \
    --manifest data/synthetic/output/manifest.jsonl \
    --data-root data/synthetic \
    --steps 17000 --batch-size 1 --grad-accum 4 --device mps

# Or use the one-liner
bash training/m4_train.sh 17000
```

### Colab Free T4

Open [training/colab_train.ipynb](training/colab_train.ipynb) — complete pipeline in one notebook (~1 hour for 5K steps).

### Training Results

| Metric | Value |
|--------|-------|
| Best checkpoint | Step 14,000 (loss: 15.0) |
| Training data | 3,000 synthetic docs (6,815 QA pairs) |
| Training time | 15.1 hours on M4 |
| LoRA rank | 16 (alpha: 32) |

## Deployment

### ONNX Export

```bash
python export/export_onnx.py --model-path eulogik/TinyDoc-VLM-256M --output model.onnx
```

ONNX models on [HF Hub](https://huggingface.co/eulogik/TinyDoc-VLM-256M):
- `tinydoc-vlm-vision.onnx` — Vision encoder (33KB)
- `tinydoc-vlm-compressor.onnx` — Token compressor (31KB)
- `tinydoc-vlm-decoder.onnx` — Language decoder (59MB)

### HuggingFace Spaces

Live demo: [huggingface.co/spaces/eulogik/TinyDoc-VLM](https://huggingface.co/spaces/eulogik/TinyDoc-VLM)

## Benchmarks

| Benchmark | Status | Target |
|-----------|--------|--------|
| OCRBench | In progress | >75% |
| DocVQA | Pending | >85% |
| FUNSD | Pending | >95% |
| CORD | Pending | >95% |

Full analysis in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Package Structure

| Package | Location | Description |
|---------|----------|-------------|
| `tinydoc` | [PyPI](https://pypi.org/project/tinydoc/) | Python SDK — `TinyDocExtractor.ask()`, `.extract()`, `.extract_table()` |
| `tinydoc-vlm` | [GitHub](https://github.com/eulogik/TinyDoc-VLM) | Full model code, training pipeline, synthetic data engine, evaluation suite |
| `TinyDoc-VLM-256M` | [HF Hub](https://huggingface.co/eulogik/TinyDoc-VLM-256M) | Pre-trained weights — 1.1GB, loads via `from_pretrained()` |
| `TinyDoc-VLM-LoRA` | [HF Hub](https://huggingface.co/eulogik/TinyDoc-VLM-LoRA) | LoRA adapter — 10MB, merge with base model |

## Links

| Resource | URL |
|----------|-----|
| GitHub | [github.com/eulogik/TinyDoc-VLM](https://github.com/eulogik/TinyDoc-VLM) |
| PyPI | [pypi.org/project/tinydoc](https://pypi.org/project/tinydoc/) |
| Model Hub | [huggingface.co/eulogik/TinyDoc-VLM-256M](https://huggingface.co/eulogik/TinyDoc-VLM-256M) |
| LoRA Checkpoint | [huggingface.co/eulogik/TinyDoc-VLM-LoRA](https://huggingface.co/eulogik/TinyDoc-VLM-LoRA) |
| Live Demo | [huggingface.co/spaces/eulogik/TinyDoc-VLM](https://huggingface.co/spaces/eulogik/TinyDoc-VLM) |
| Documentation | [eulogik.github.io/TinyDoc-VLM](https://eulogik.github.io/TinyDoc-VLM/) |

## Launch Assets

| Document | Description |
|----------|-------------|
| [HN Post](docs/launch_announcement.md) | Hacker News Show HN draft |
| [Reddit Post](docs/reddit_post.md) | r/LocalLLaMA, r/MachineLearning |
| [Twitter Thread](docs/twitter_thread.md) | 7-tweet launch thread |
| [Pitch Deck](docs/pitch_deck.md) | Enterprise one-pager |

## Citation

```bibtex
@software{eulogik_tinydoc_vlm_2026,
  author = {eulogik},
  title = {TinyDoc-VLM: 256M-Param Document-Specialist Vision-Language Model},
  year = {2026},
  url = {https://github.com/eulogik/TinyDoc-VLM}
}
```

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache 2.0. See [LICENSE](LICENSE) for details.

---

<div align="center">
  <b>Made with by <a href="https://eulogik.com">eulogik</a></b>
</div>
