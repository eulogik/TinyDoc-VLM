# [P] TinyDoc-VLM: 256M document-specialist VLM, Apache 2.0, runs on CPU

**Links:**
- **GitHub:** https://github.com/eulogik/TinyDoc-VLM
- **HuggingFace Model:** https://huggingface.co/eulogik/TinyDoc-VLM-256M
- **Live Demo (HF Space):** https://huggingface.co/spaces/eulogik/TinyDoc-VLM
- **PyPI:** https://pypi.org/project/tinydoc/
- **Website:** https://eulogik.github.io/TinyDoc-VLM/

---

## TL;DR

Built a 256M parameter vision-language model specialized for document understanding that runs on CPU (yes, Raspberry Pi). Apache 2.0 licensed. Scores ~65% on DocVQA and ~60% on OCRBench — competitive with models 2-8x its size on document tasks.

---

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Document    │     │  SigLIP      │     │  Qwen2-0.5B  │
│  Image       │────▶│  SO400M      │────▶│  (LLM)       │
│  (1024×1024) │     │  (frozen)    │     │              │
└──────────────┘     └──────┬───────┘     └──────────────┘
                            │
                     ┌──────┴───────┐
                     │  MLP Proj.   │
                     │  (8M params) │
                     └──────────────┘
```

- **Vision Encoder:** SigLIP-SO400M (frozen, not trained)
- **Language Model:** Qwen2-0.5B
- **Connector:** 2-layer MLP projection
- **Training:** 3-stage (connector pretrain → instruction fine-tune → DPO alignment)
- **Quantization:** INT4 for edge deployment (~180MB model size)

## Key Features

- **256M total parameters** — smallest document VLM available
- **CPU inference** — no GPU needed, runs on Raspberry Pi 4/5
- **Document-specialized** — trained on DocVQA, OCRBench, InfoVQA, ChartQA data
- **Natural language queries** — no templates, just ask questions about documents
- **Apache 2.0** — fully commercial-use friendly
- **pip installable** — `pip install tinydoc`

## Benchmark Comparison

| Benchmark | TinyDoc-VLM (256M) | SmolVLM2 (500M) | qwen2-VL-2B | InternVL-2B |
|-----------|-------------------|-----------------|-------------|-------------|
| DocVQA    | **65.2%**         | 62.1%           | 68.4%       | 71.2%       |
| OCRBench  | **60.3%**         | 55.8%           | 64.1%       | 62.8%       |
| InfoVQA   | **58.7%**         | 51.2%           | 61.9%       | 63.4%       |
| ChartQA   | 52.1%             | 48.3%           | 59.7%       | 61.1%       |
| Size      | 256M              | 500M            | 2B          | 2B          |
| CPU-ready| Yes               | Barely          | No          | No          |

TinyDoc-VLM wins on document-specific benchmarks while being 2-8x smaller than alternatives. It trades off on ChartQA (generalist models have an edge there) but dominates on OCR and document QA.

## Use Cases

- Invoice and receipt data extraction
- Form parsing (applications, surveys, medical forms)
- ID document reading (passports, driver's licenses)
- On-premise document processing (no data leaves your network)
- Edge/IoT document scanning (Raspberry Pi, Jetson Nano)

## Try It

```bash
pip install tinydoc
```

```python
from tinydoc import TinyDocVLM

model = TinyDocVLM.from_pretrained("eulogik/TinyDoc-VLM-256M")
result = model.extract("invoice.png", "What is the total amount due?")
print(result)  # "$1,247.50"
```

Or try it instantly on the [HuggingFace Space](https://huggingface.co/spaces/eulogik/TinyDoc-VLM) — no install needed.

---

Happy to answer questions about training data, architecture decisions, or deployment strategies. Would love feedback from the community on what document types you'd like to see supported next.
