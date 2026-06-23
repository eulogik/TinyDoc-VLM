# TinyDoc-VLM — Living Walkthrough & Handover Doc

> **Last updated**: 2026-06-23
> **Repo**: https://github.com/eulogik/TinyDoc-VLM  
> **HF Model**: https://huggingface.co/eulogik/TinyDoc-VLM-256M  
> **HF Space**: https://huggingface.co/spaces/eulogik/TinyDoc-VLM  
> **PyPI**: https://pypi.org/project/tinydoc/  
> **Website**: https://eulogik.github.io/TinyDoc-VLM/  

---

## 1. Environment & Platform

| Setting | Value |
|---------|-------|
| OS | macOS 12.7.6, Intel x86_64 |
| Python | 3.11.15 |
| PyTorch | 2.2.2 (last version supporting macOS x86_64) |
| Transformers | 4.44.2 (downgraded; 4.45+ requires torch>=2.4) |
| NumPy | 1.26.4 (NumPy 2.x breaks torch 2.2.2 binaries) |
| Virtual env | `venv/` at repo root |

**Constraints**: PyTorch 2.3+ dropped macOS x86_64 support. `nn.RMSNorm` requires PyTorch 2.4+. `ProcessorMixin` in transformers 4.44.2 has strict class-name lookups.

---

## 2. Architecture Overview

```
Input Image -> TinyDocImageProcessor -> [tiles: (B, N, 3, 384, 384)]
                                                  |
                                       SigLIPVisionEncoder (93M params)
                                                  |
                                   PixelShuffleTokenCompressor (3x3, 9x reduction)
                                   (384/16/3)^2 = 64 tokens per tile
                                                  |
                            merged with text via <image> placeholder replacement
                                                  |
                                        TinyDocDecoder (SmolLM2-135M, 30L)
                                                  |
                                   MultiTaskOutputHeads (JSON/KV/Table/OCR/QA)
```

**Total parameters**: ~290M (SigLIP-B/16 93M + compressor ~3M + SmolLM2-135M + output heads 59M)

---

## 3. Repository Map (Complete Status)

```
TinyDoc-VLM/
├── tinydoc_vlm/                  COMPLETE package
│   ├── __init__.py               Registers AutoConfig + AutoModel
│   ├── configuration.py          TinyDocVLMConfig
│   ├── vision_encoder.py         SigLIPVisionEncoder wrapper
│   ├── token_compressor.py       PixelShuffleTokenCompressor + custom RMSNorm
│   ├── decoder.py                TinyDocDecoder (SmolLM2 wrapper)
│   ├── attention.py              2D sinusoidal positional embeddings
│   ├── modeling.py               TinyDocVLMForConditionalGeneration (full VLM)
│   ├── image_processing.py       TinyDocImageProcessor (tiling, aspect-ratio pad)
│   ├── processing.py             TinyDocVLMProcessor (standalone, no ProcessorMixin)
│   ├── output_heads.py           MultiTaskOutputHeads (JSON/KV/Table/OCR/QA)
│   ├── data.py                   DocumentDataset + collate_fn
│   ├── losses.py                 CombinedLoss (stage-aware multi-task)
│   └── trainer.py                TinyDocVLMTrainer (3-stage, mixed precision)
│
├── data/
│   ├── datasets/                 Modular dataset loaders wrapper
│   │   ├── unified.py            Unified dataset loader implementation
│   │   ├── docvqa.py             DocVQA loader wrapper
│   │   ├── funsd.py              FUNSD loader wrapper
│   │   ├── cord.py               CORD loader wrapper
│   │   ├── sroie.py              SROIE loader wrapper
│   │   └── pubtabnet.py          PubTabNet loader wrapper
│   ├── tokenizer/
│   │   ├── special_tokens.py     30 doc-special tokens
│   │   └── extended_tokenizer/   Saved extended tokenizer
│   └── synthetic/
│       ├── templates/            10 HTML/Jinja2 templates
│       ├── pil_renderer.py       PIL-based renderer
│       ├── generator.py          Full pipeline: Faker -> render -> augment -> JSONL
│       └── output/
│           ├── manifest.jsonl    ~12MB manifest
│           └── images/           Generated document images
├── training/
│   ├── run.py                    CLI training launcher
│   ├── stage1_layout_pretrain.yaml
│   ├── stage2_doc_understanding.yaml
│   ├── stage3_instruction_tuning.yaml
│   └── tinydoc_colab_training.ipynb  Colab notebook (Drive auto-resume, T4 GPU)
├── evaluation/
│   ├── evaluate.py               ANLS, F1, DocVQA/FUNSD/CORD/OCRBench harness
│   └── download_benchmarks.py    Benchmark downloader
├── export/
│   ├── export_onnx.py            ONNX export with dynamic axes
│   └── export_gguf.py            GGUF export (llama.cpp compatible)
├── demo/
│   ├── app.py                    Gradio demo
│   ├── hf_space/                 HF Space deployment files
│   │   ├── Dockerfile            Python 3.11 Docker image
│   │   ├── app.py                Space entry point
│   │   ├── requirements.txt      Space dependencies
│   │   ├── README.md             Space metadata
│   │   └── tinydoc_vlm/          Model source code (local copy for Space)
│   └── examples/                 Pre-generated document images
├── sdk/                          COMPLETE package (published to PyPI)
│   ├── setup.py                  pip install setup script
│   ├── MANIFEST.in               Manifest for README inclusion
│   ├── README.md                 PyPI long description
│   └── tinydoc/
│       ├── __init__.py           Exposes TinyDocExtractor
│       ├── extractor.py          High-level extractor APIs (QA, Extract, Table)
│       └── models.py             Pydantic models for outputs
├── docs/
│   └── index.html                GitHub Pages website
├── tests/
│   ├── test_model.py             Model architecture/processor tests
│   ├── test_datasets.py          Dataset loaders tests
│   └── test_sdk.py               SDK extractor tests
├── .github/workflows/
│   ├── ci.yml                    GitHub Actions CI (pytest + ruff)
│   └── gh-pages.yml              GitHub Pages deploy
├── LICENSE                       Apache 2.0
├── README.md                     GitHub root README
└── walkthrough.md                This file
```

---

## 4. Deployment Status

| Service | URL | Status |
|---------|-----|--------|
| GitHub Repo | https://github.com/eulogik/TinyDoc-VLM | ✅ Active |
| HF Model Hub | https://huggingface.co/eulogik/TinyDoc-VLM-256M | ✅ Published |
| HF Space Demo | https://huggingface.co/spaces/eulogik/TinyDoc-VLM | ⚠️ Building |
| PyPI Package | https://pypi.org/project/tinydoc/ | ✅ v0.1.0 |
| GitHub Pages | https://eulogik.github.io/TinyDoc-VLM/ | ✅ Deployed |
| Twitter | https://twitter.com/eulogik | ✅ Listed |

---

## 5. Training (Completed on Colab)

3 stages completed successfully on Colab T4 GPU:
- **Stage 1**: Layout pretraining
- **Stage 2**: Document understanding  
- **Stage 3**: Instruction tuning
- **Weights**: 290M params, 1.1GB F32, pushed to `eulogik/TinyDoc-VLM-256M`

---

## 6. Test Status

```bash
PYTHONPATH=. ./venv/bin/pytest -v
```
All 13 unit tests passing:
```
tests/test_datasets.py::test_docvqa_dataset PASSED
tests/test_datasets.py::test_funsd_dataset PASSED
tests/test_datasets.py::test_cord_dataset PASSED
tests/test_datasets.py::test_sroie_dataset PASSED
tests/test_datasets.py::test_pubtabnet_dataset PASSED
tests/test_datasets.py::test_synthetic_doc_dataset PASSED
tests/test_model.py::test_config PASSED
tests/test_model.py::test_image_processor PASSED
tests/test_model.py::test_model_forward PASSED
tests/test_model.py::test_processor_integration PASSED
tests/test_sdk.py::test_sdk_extractor_initialisation PASSED
tests/test_sdk.py::test_sdk_extractor_methods PASSED
tests/test_sdk.py::test_html_table_to_markdown_converter PASSED
```

---

## 7. Key Contacts

- **Company**: eulogik (https://eulogik.com)
- **Twitter**: @eulogik
- **Email**: hello@eulogik.com
- **Author**: Sunday Shah

---

*Built by [eulogik](https://eulogik.com) — AI infrastructure for document intelligence.*
