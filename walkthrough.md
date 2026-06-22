# TinyDoc-VLM — Living Walkthrough & Handover Doc

> **Last updated**: 2026-06-22
> **Repo**: https://github.com/eulogik/TinyDoc-VLM  
> **All 13 unit tests: PASSING**

This is the single source of truth for any human or AI agent picking up TinyDoc-VLM. It is updated continuously as work progresses.

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

**Total parameters**: ~235M (SigLIP-B/16 93M + compressor ~3M + SmolLM2-135M)

---

## 3. Repository Map (Complete Status)

```
TinyDoc-VLM/
├── tinydoc_vlm/                  COMPLETE package
│   ├── __init__.py               Registers AutoConfig + AutoModel
│   ├── configuration.py          TinyDocVLMConfig (fixed AutoConfig.for_model bug)
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
│   │   ├── special_tokens.py     30 doc-special tokens added to SmolLM2 tokenizer
│   │   └── extended_tokenizer/   Saved extended tokenizer
│   └── synthetic/
│       ├── templates/            10 HTML/Jinja2 templates (invoice, receipt, form, etc.)
│       ├── pil_renderer.py       PIL-based renderer (no WeasyPrint needed)
│       ├── generator.py          Full pipeline: Faker -> render -> augment -> JSONL
│       └── output/
│           ├── manifest.jsonl    ~12MB manifest (thousands of samples)
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
│   ├── app.py                    Gradio demo (extract JSON / QA / table / OCR)
│   └── examples/                 Pre-generated invoice/receipt/table images
├── sdk/                          COMPLETE package
│   ├── setup.py                  pip install setup script
│   └── tinydoc/
│       ├── __init__.py           Exposes TinyDocExtractor
│       ├── extractor.py          High-level extractor APIs (QA, Extract, Table)
│       └── models.py             Pydantic models for outputs
├── tests/
│   ├── test_model.py             Model architecture/processor tests
│   ├── test_datasets.py          Dataset loaders tests
│   └── test_sdk.py               SDK extractor tests
└── .github/workflows/ci.yml      GitHub Actions CI (pytest + ruff)
```

---

## 4. Test Status

```bash
PYTHONPATH=. ./venv/bin/pytest -v
```
All 13 unit tests are passing successfully:
```
tests/test_datasets.py::test_docvqa_dataset PASSED                       [  7%]
tests/test_datasets.py::test_funsd_dataset PASSED                        [ 15%]
tests/test_datasets.py::test_cord_dataset PASSED                         [ 23%]
tests/test_datasets.py::test_sroie_dataset PASSED                        [ 30%]
tests/test_datasets.py::test_pubtabnet_dataset PASSED                    [ 38%]
tests/test_datasets.py::test_synthetic_doc_dataset PASSED                [ 46%]
tests/test_model.py::test_config PASSED                                  [ 53%]
tests/test_model.py::test_image_processor PASSED                         [ 61%]
tests/test_model.py::test_model_forward PASSED                           [ 69%]
tests/test_model.py::test_processor_integration PASSED                   [ 76%]
tests/test_sdk.py::test_sdk_extractor_initialisation PASSED              [ 84%]
tests/test_sdk.py::test_sdk_extractor_methods PASSED                     [ 92%]
tests/test_sdk.py::test_html_table_to_markdown_converter PASSED          [100%]

================== 13 passed, 2 warnings in 55.45s ===================
```

---

## 5. Bugs Fixed

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| `AutoConfig.for_model() multiple values for model_type` | model_type in both positional + kwargs | `.pop()` key from config copy before passing |
| `torch.nn has no attribute RMSNorm` | nn.RMSNorm added in PyTorch 2.4 | Custom RMSNorm(nn.Module) in token_compressor.py |
| `transformers has no attribute TinyDocImageProcessor` | ProcessorMixin does global name lookup | Rewrote TinyDocVLMProcessor as standalone class |

---

## 6. Synthetic Data

```bash
# Generate documents (PIL renderer, no WeasyPrint needed)
python data/synthetic/generator.py --num-docs 1000 --output-dir data/synthetic/output
```

Manifest (~12MB) already populated. Each JSONL entry:
```json
{"image_path": "synthetic/output/images/invoice_000001.png",
 "doc_type": "invoice",
 "metadata": {"vendor_name": "...", "total": "$1,234.56"},
 "text": "Extract document information: <image>",
 "qa_pairs": [{"question": "What is the total?", "answer": "$1,234.56"}]}
```

---

## 7. Training

```bash
# Stage 1: Layout pretraining
python training/run.py --config training/stage1_layout_pretrain.yaml
# Stage 2: Document understanding
python training/run.py --config training/stage2_doc_understanding.yaml
# Stage 3: Instruction tuning
python training/run.py --config training/stage3_instruction_tuning.yaml
```

**Colab**: `training/tinydoc_colab_training.ipynb` — T4 GPU, auto-saves to Drive.

---

## 8. Gap Analysis

### Done
- Full model architecture (all unit tests pass)
- Synthetic data pipeline (10 templates, Faker, PIL renderer, 12MB manifest)
- Training infrastructure (3-stage trainer, YAML configs, Colab notebook)
- Evaluation harness (ANLS, F1, DocVQA, OCRBench)
- Gradio demo app
- Export scripts (ONNX, GGUF)
- GitHub CI (pytest + ruff)
- **Real dataset loaders** (`data/datasets/` wrappers exposing DocVQA, FUNSD, CORD, SROIE, PubTabNet and Unified loaders)
- **Python SDK** (`sdk/tinydoc/extractor.py`) with high-level `extract`, `ask` (VQA), and `extract_table` APIs and dynamic PyTorch/ONNX Runtime auto-routing

### Remaining
1. **Actual training** — No trained checkpoint yet. Use Colab notebook or cloud GPU.
2. **More synthetic data** — Need 10M+ samples; currently 10,000 samples generated.
3. **HuggingFace Hub push** — After training, push to `eulogik/TinyDoc-VLM-256M`.

---

## 9. Git History

```
72e7200 Implement Python SDK and modular dataset loader files with comprehensive unit tests
60f59f7 Add unified dataset loaders for training
5e25970 Fix: Rewrite TinyDocVLMProcessor as standalone class (all 4 tests pass)
268d746 Add __init__.py to data/ and data/synthetic/
be0947d Fix CWD issues in Colab
8db8ea8 Final Colab notebook with Drive auto-resume
abaffc6 Add Colab training notebook
995cb51 Fix ruff lint errors
95b6a32 Add GitHub Actions CI
0987e85 Add demo, export, eval scripts
8a78fb2 Complete training pipeline, synthetic data engine, eval suite
6004faf Initial commit
```
