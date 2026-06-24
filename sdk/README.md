# 📄 TinyDoc — AI Document Understanding in 3 Lines of Code

> **Extract answers, tables, and structured data from any document image.**

[![PyPI](https://img.shields.io/pypi/v/tinydoc?color=blue&label=pip%20install)](https://pypi.org/project/tinydoc/)
[![GitHub](https://img.shields.io/badge/source-eulogik%2FTinyDoc--VLM-blue)](https://github.com/eulogik/TinyDoc-VLM)
[![HF Space](https://img.shields.io/badge/demo-🤗-live-yellow)](https://huggingface.co/spaces/eulogik/TinyDoc-VLM)
[![License](https://img.shields.io/pypi/l/tinydoc-green)](https://opensource.org/licenses/Apache-2.0)

---

## What is this?

**TinyDoc** is a Python SDK powered by [TinyDoc-VLM](https://huggingface.co/eulogik/TinyDoc-VLM-256M) — a 256M parameter vision-language model trained specifically for document understanding. It runs on **CPU** with no GPU required.

Drop in a document image. Ask a question. Get the answer.

---

## Install

```bash
pip install tinydoc
```

That's it. One package. ~1.1GB model weights auto-download from HuggingFace on first use.

---

## Quick Start

```python
from PIL import Image
from tinydoc import TinyDocExtractor

extractor = TinyDocExtractor()  # auto-detects device, loads from HF Hub
img = Image.open("invoice.png")

# 💬 Ask a question
result = extractor.ask(img, "What is the total amount?")
print(result.answer)  # "$1,234.56"

# 📋 Extract all fields as JSON
result = extractor.extract(img, output_format="json")
print(result.fields)  # {"vendor": "Acme Corp", "total": "$1,234.56", "date": "2024-01-15", ...}

# 📊 Extract tables to Markdown
result = extractor.extract_table(img)
print(result.markdown)
# | Item       | Qty | Price  |
# |------------|-----|--------|
# | Widget A   | 10  | $25.00 |
# | Widget B   | 5   | $50.00 |
```

---

## What can it do?

| Task | How | Example |
|------|-----|---------|
| **VQA** | `extractor.ask(img, "question")` | "What is the invoice date?" |
| **JSON Extraction** | `extractor.extract(img)` | Pulls all key-value pairs |
| **Table Parsing** | `extractor.extract_table(img)` | Converts tables to Markdown |
| **OCR** | `extractor.ask(img, "Transcribe the text")` | Plain text output |
| **Key-Value Pairs** | `extractor.extract(img, output_format="kv")` | Dict of field→value |

---

## Why TinyDoc?

| | GPT-4V | Tesseract | TinyDoc |
|--|--------|-----------|---------|
| **Size** | ~2T params | N/A | **256M** |
| **Cost** | $0.01+/query | Free | **Free** |
| **Runs on** | API only | CPU | **CPU or GPU** |
| **Structured output** | Prompt-dependent | None | **Native** |
| **Latency** | ~2-5s (API) | <100ms | **<500ms** |
| **License** | Proprietary | Apache 2.0 | **Apache 2.0** |

---

## Advanced

```python
extractor = TinyDocExtractor(
    device="cuda",           # or "cpu", "mps"
    model_name_or_id="eulogik/TinyDoc-VLM-256M",  # or local path
)

result = extractor.ask(
    img,
    "What are the line items?",
    max_new_tokens=256,      # override default 512
)
```

---

## Links

| Platform | Link |
|----------|------|
| 🐍 **PyPI** | [pypi.org/project/tinydoc](https://pypi.org/project/tinydoc/) |
| 🤗 **Model Hub** | [eulogik/TinyDoc-VLM-256M](https://huggingface.co/eulogik/TinyDoc-VLM-256M) |
| 🤗 **Live Demo** | [Space: eulogik/TinyDoc-VLM](https://huggingface.co/spaces/eulogik/TinyDoc-VLM) |
| 📖 **GitHub** | [github.com/eulogik/TinyDoc-VLM](https://github.com/eulogik/TinyDoc-VLM) |
| 🌐 **Website** | [eulogik.github.io/TinyDoc-VLM](https://eulogik.github.io/TinyDoc-VLM/) |
| 🐦 **Twitter** | [@eulogik](https://twitter.com/eulogik) |

---

## License

Apache 2.0 — free for commercial use.

---

**Built by [eulogik](https://eulogik.com)** — AI infrastructure for document intelligence.
