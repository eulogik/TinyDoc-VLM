<div align="center">
  <h1>📄 TinyDoc-VLM</h1>
  <p><b>The World's Smallest Document-Specialist VLM</b></p>
  <p>256M parameters · Runs on Raspberry Pi · >100 tok/s on CPU</p>

[![PyPI](https://img.shields.io/pypi/v/tinydoc?color=blue)](https://pypi.org/project/tinydoc/)
[![HF Model](https://img.shields.io/badge/🤗-Model-yellow)](https://huggingface.co/eulogik/TinyDoc-VLM-256M)
[![HF Space](https://img.shields.io/badge/🤗-Space-yellow)](https://huggingface.co/spaces/eulogik/TinyDoc-VLM)
[![CI](https://github.com/eulogik/TinyDoc-VLM/actions/workflows/ci.yml/badge.svg)](https://github.com/eulogik/TinyDoc-VLM/actions)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Twitter](https://img.shields.io/twitter/follow/eulogik?style=social)](https://twitter.com/eulogik)

---

**Built by [eulogik](https://eulogik.com)** — AI infrastructure for document intelligence.

[🐍 PyPI](https://pypi.org/project/tinydoc/) · [🤗 Model Hub](https://huggingface.co/eulogik/TinyDoc-VLM-256M) · [🤗 Space Demo](https://huggingface.co/spaces/eulogik/TinyDoc-VLM) · [📖 Website](https://eulogik.github.io/TinyDoc-VLM/) · [🐦 @eulogik](https://twitter.com/eulogik)
</div>

## 🔥 Highlights

- **256M params**: SigLIP-B/16 vision encoder (93M) + Pixel-Shuffle connector + SmolLM2-135M decoder
- **<1GB VRAM**: Runs on Raspberry Pi 5, MacBook Air, or any CPU with ONNX
- **Structured output**: JSON extraction, key-value pairs, table parsing, OCR, VQA — all from one model
- **3-stage training**: Layout pretrain → Document understanding → Instruction tuning on 10K+ synthetic docs
- **Apache 2.0**: Fully open-source, free for commercial use

## 🚀 Quick Start

```bash
pip install tinydoc
```

```python
from PIL import Image
from tinydoc import TinyDocExtractor

extractor = TinyDocExtractor(device="cpu")  # loads from HF Hub

# Ask a question
img = Image.open("invoice.png")
result = extractor.ask(img, "What is the total?")
print(result.answer)  # "$1,234.56"

# Extract JSON fields
result = extractor.extract(img, output_format="json")
print(result.fields)  # {"total": "$1,234.56", "date": "2024-01-15", ...}

# Extract tables
result = extractor.extract_table(img)
print(result.markdown)  # Markdown-formatted table
```

## 📦 Package Structure

| Package | Description |
|---------|-------------|
| `tinydoc` (PyPI) | Lightweight Python SDK — `TinyDocExtractor.ask()`, `.extract()`, `.extract_table()` |
| `tinydoc-vlm` (source) | Full model code, training pipeline, synthetic data engine, evaluation suite |
| `eulogik/TinyDoc-VLM-256M` (HF Hub) | Pre-trained weights — 1.1GB, loads via `from_pretrained()` |

## 🏗️ Model Architecture

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

## 📊 Benchmarks (v0.1.0 — Preliminary)

| Benchmark | TinyDoc-VLM | SmolVLM2-256M | Target |
|-----------|-------------|---------------|--------|
| **DocVQA** | 65.3% | ~58% | >85% |
| **OCRBench** | 60.8% | 52.6% | >75% |
| **FUNSD** | 85.2% | — | >95% |
| **CORD** | 87.6% | — | >95% |
| **SROIE** | 85.9% | — | >95% |
| **ChartQA** | 61.4% | ~55% | >75% |
| **Table Extraction** | 68.7% | ~58% | >85% |

TinyDoc-VLM-256M outperforms SmolVLM2-256M by **+6–13 points** on most benchmarks despite being the same parameter class, confirming that document-specialized architecture provides meaningful gains. Full analysis in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## 🧪 Training

3-stage curriculum on synthetic documents (10K types):

1. **Layout Pretrain** — Learn document structure, region classification, layout detection
2. **Doc Understanding** — QA, extraction, table parsing with real benchmarks
3. **Instruction Tuning** — Multi-turn conversation, structured output formatting

[**Colab Notebook**](training/tinydoc_colab_training.ipynb) — Train your own on a free T4 GPU.

## 🔧 Deployment

```bash
# ONNX export
python export/export_onnx.py --model-path eulogik/TinyDoc-VLM-256M --output model.onnx

# GGUF export (decoder only)
python export/export_gguf.py --model-path eulogik/TinyDoc-VLM-256M --output model.gguf
```

## 📚 Citation

```bibtex
@software{eulogik_tinydoc_vlm_2025,
  author = {eulogik},
  title = {TinyDoc-VLM: The World's Smallest Document-Specialist VLM},
  year = {2025},
  url = {https://github.com/eulogik/TinyDoc-VLM}
}
```

## 🗺️ Launch Assets

| Document | Description |
|----------|-------------|
| [HN Post](docs/launch_announcement.md) | Hacker News Show HN draft |
| [Reddit Post](docs/reddit_post.md) | r/LocalLLaMA / r/MachineLearning |
| [Twitter Thread](docs/twitter_thread.md) | 7-tweet launch thread |
| [Pitch Deck](docs/pitch_deck.md) | Enterprise one-pager |
| [OpenRouter Info](docs/openrouter_submission.md) | Submission-ready model metadata |

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

Apache 2.0. See [LICENSE](LICENSE) for details.

---

<div align="center">
  <b>Made with ❤️ by <a href="https://eulogik.com">eulogik</a></b>
</div>
