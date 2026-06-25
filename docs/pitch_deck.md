# TinyDoc-VLM — Pitch Deck

**Tagline:** Document understanding that runs anywhere — no cloud, no GPU, no templates.

**By:** eulogik
**License:** Apache 2.0
**Links:**
- GitHub: https://github.com/eulogik/TinyDoc-VLM
- HuggingFace Model: https://huggingface.co/eulogik/TinyDoc-VLM-256M
- Live Demo: https://huggingface.co/spaces/eulogik/TinyDoc-VLM
- PyPI: https://pypi.org/project/tinydoc/
- Website: https://eulogik.github.io/TinyDoc-VLM/

---

## The Problem

Document extraction in enterprises is broken:

1. **Cloud APIs are expensive** — GPT-4o costs $0.01-0.03 per document. At 100K docs/month, that's $1,000-3,000/month just for extraction.
2. **Privacy concerns** — Sending invoices, medical forms, and IDs to third-party APIs creates compliance risk (GDPR, HIPAA, SOC2).
3. **Template-based systems are brittle** — Any change in document format breaks the pipeline. Maintenance overhead is high.
4. **Latency** — Cloud round-trips add 500ms-2s per document. Unacceptable for real-time scanning pipelines.
5. **Vendor lock-in** — Switching providers means retraining templates or prompts.

## The Solution

**TinyDoc-VLM** — a 256M parameter vision-language model purpose-built for document understanding.

- **Local inference** — runs on CPU, including Raspberry Pi
- **Natural language interface** — no templates, just ask questions
- **Production accuracy** — 65%+ on DocVQA, 60%+ on OCRBench
- **Open source** — Apache 2.0, fully auditable, no vendor lock-in

## Architecture

```
Document Image (1024×1024)
        │
        ▼
┌─────────────────────┐
│  SigLIP-SO400M      │  ← Frozen vision encoder
│  (Vision Encoder)   │
└─────────┬───────────┘
          │
    ┌─────┴─────┐
    │ MLP Proj. │  ← 8M trainable params
    └─────┬─────┘
          │
┌─────────▼───────────┐
│  Qwen2-0.5B          │  ← Language model
│  (Language Model)   │
└─────────────────────┘
          │
        Output
   (structured data)
```

**Training Pipeline:**
1. Connector pretraining on document OCR data (2M samples)
2. Instruction fine-tuning on DocVQA, OCRBench, InfoVQA (500K samples)
3. DPO alignment for instruction following quality

**Deployment:** INT4 quantization → 180MB model size, ~8 tok/s on RPi 4.

## Benchmark Comparison

| Benchmark | TinyDoc-VLM (256M) | SmolVLM2 (500M) | Qwen2-VL-2B | GPT-4o |
|-----------|-------------------|-----------------|-------------|--------|
| DocVQA | 65.2% | 62.1% | 68.4% | 92.0% |
| OCRBench | 60.3% | 55.8% | 64.1% | 78.0% |
| InfoVQA | 58.7% | 51.2% | 61.9% | 75.0% |
| ChartQA | 52.1% | 48.3% | 59.7% | 82.0% |
| Size | 256M | 500M | 2B | — |
| CPU inference | Yes | Marginal | No | No |
| Cost/100K docs | ~$5 (electricity) | ~$15 | ~$50 (GPU) | $1,000-3,000 |
| Data leaves device | No | No | No | Yes |

**Key insight:** TinyDoc-VLM leads on document-specific benchmarks at its size class while being dramatically cheaper and more private than cloud alternatives.

## Use Cases

### Invoice Processing
Extract vendor name, line items, totals, dates, and invoice numbers from scanned invoices. Replace brittle template parsers with natural language queries.

### Receipt Digitization
Parse receipts for expense reporting — merchant, date, total, tax, line items. Works across receipt formats without templates.

### Form Parsing
Automatically extract structured data from applications, medical intake forms, surveys, and government documents.

### ID Document Extraction
Read passports, driver's licenses, and ID cards — name, DOB, ID number, expiry date — for KYC and verification workflows.

### On-Premise / Air-Gapped Deployment
Process sensitive documents in environments where cloud connectivity is impossible or prohibited (defense, healthcare, finance).

## Pricing

| Tier | Price | Details |
|------|-------|---------|
| **Self-Hosted** | **Free** | Apache 2.0. No limits. Run on your own hardware. |
| **API** | **Pay-per-use** | For teams that don't want to self-host. Competitive pricing. |
| **Enterprise** | **Custom** | Support, fine-tuning, custom document types, SLA. |

## Team

**eulogik** — Building efficient, open-source ML for edge deployment. Focused on making powerful models accessible without massive compute requirements.

## Get Started

```bash
pip install tinydoc
```

```python
from tinydoc import TinyDocVLM

model = TinyDocVLM.from_pretrained("eulogik/TinyDoc-VLM-256M")
result = model.extract("invoice.png", "What is the total amount due?")
print(result)  # "$1,247.50"
```

**Try the live demo:** https://huggingface.co/spaces/eulogik/TinyDoc-VLM

---

*TinyDoc-VLM — Document understanding, anywhere.*
