# TinyDoc-VLM — Living Walkthrough & Handover Doc

> **Last updated**: 2026-07-16
> **Repo**: https://github.com/eulogik/TinyDoc-VLM  
> **HF Model**: https://huggingface.co/eulogik/TinyDoc-VLM-256M  
> **HF Space**: https://huggingface.co/spaces/eulogik/TinyDoc-VLM  
> **PyPI**: https://pypi.org/project/tinydoc/  
> **Website**: https://eulogik.github.io/TinyDoc-VLM/  

---

## 1. Environment & Platform

| Setting | Value |
|---------|-------|
| OS | macOS (Apple Silicon, M4) |
| Python | 3.14 |
| PyTorch | 2.12.1 (MPS available) |
| Transformers | latest (4.45+ OK now; torch>=2.4) |
| Virtual env | `venv/` at repo root, or system Python 3.14 |

**Notes**: Training/inference run on the M4 (MPS) and on Colab T4 / GPU for heavier jobs. `nn.RMSNorm` works (PyTorch 2.4+). The old x86_64 / PyTorch 2.2.2 constraint no longer applies.

---

## 2. Architecture Overview

```
Input Image -> TinyDocImageProcessor -> [tiles: (B, N, 3, 768, 768)]
                                                   |
                                        SigLIPVisionEncoder (93M params)
                                        (runs with interpolate_pos_encoding=True)
                                                   |
                                    PixelShuffleTokenCompressor (3x3)
                                    (768/16/3)^2 = 256 visual tokens per tile
                                                   |
                             merged with text via <image> placeholder replacement
                             (+ learnable 2D visual_pos_embed)
                                                   |
                                         TinyDocDecoder (SmolLM2-135M, 30L)
                                                   |
                                    LM head -> prompt-routed output:
                                    "Extract all text:" | "Convert to markdown:"
                                    "Answer the question: ..." | JSON-field extraction
```

**Total parameters**: ~262M (SigLIP-B/16 93M + compressor ~3M + SmolLM2-135M).  
The earlier multi-task output heads (JSON/KV/Table/OCR/QA) were **removed** — they emitted to tiny fixed vocabularies (128–256 tokens) and could not generate real text, which is why the original model scored 0% on OCRBench. Output now flows through the decoder LM head, prompt-routed like GLM-OCR / DeepSeek-OCR-2 / Unlimited-OCR.

**Resolution**: default input is now **768×768** (was 384×384 → 64 tokens/tile). 768 gives 256 visual tokens/tile (4× more detail) at the same parameter count. `SigLIPVisionEncoder.resize_pos_embeddings()` interpolates a 384-pretrained encoder into the 768 grid when loading weights. Existing 384 checkpoints remain loadable (their `config.json` pins 384 → 64 tokens).

**Decoding**: `generate()` is overridden to inject `no_repeat_ngram_size=20, repetition_penalty=1.1` by default — the anti-repetition trick from Baidu Unlimited-OCR that prevents the looping collapse on long document output.

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
│   ├── modeling.py               TinyDocVLMForConditionalGeneration (full VLM, LM-head output)
│   ├── image_processing.py       TinyDocImageProcessor (tiling, aspect-ratio pad, 768)
│   ├── processing.py             TinyDocVLMProcessor (standalone, no ProcessorMixin)
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
| HF Model Hub | https://huggingface.co/eulogik/TinyDoc-VLM-256M | ⚠️ Legacy (base, 384, pre-fix) |
| HF LoRA Adapter | https://huggingface.co/eulogik/TinyDoc-VLM-LoRA | ⚠️ Legacy (on 256M base, pre-fix) |
| HF 768 model | *(planned)* `eulogik/TinyDoc-VLM-768` | 🚧 In progress (Colab retrain) |
| HF Space Demo | https://huggingface.co/spaces/eulogik/TinyDoc-VLM | ✅ Live (HTTP 200) |
| PyPI Package | https://pypi.org/project/tinydoc/ | ✅ v0.2.0 |
| GitHub Pages | https://eulogik.github.io/TinyDoc-VLM/ | ✅ Deployed |
| Awesome-list PRs | kba/awesome-ocr #154, awesome-open-source-ai #10, awesome-small-language-models #4 | 🟡 Open (awaiting maintainers) |

---

## 5. Training Status (honest)

- **Original 3-stage training** (Colab T4: layout pretrain → doc understanding → instruction tuning) produced the 290M base model, but it scored **0% on OCRBench** — the multi-task output heads could not emit real text.
- **LoRA fine-tune** (`training/fast_train.py`, `training/overnight_train.py`): 2.7M trainable params (0.93%), trained on 3K synthetic docs / 6,815 QA pairs. Loss 43.3 → 15.0 (best @ step 14K). Adapter at `eulogik/TinyDoc-VLM-LoRA`. Full eval on OCRBench still pending (generation too slow on M4 CPU/MPS — needs GPU/Colab).
- **Architecture fixes A/B/C** (2026-07-16): removed output heads, raised resolution 384→768, added ngram repetition penalty. These require a **full retrain** to take effect (existing 384 checkpoints stay at 384). See §8.

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

## 7. Recent Changes & Roadmap (2026-07-16)

### Done this session
- **A — Removed multi-task output heads** (`output_heads.py` deleted). Model now generates text/markdown/JSON through the decoder LM head, prompt-routed. Root cause of the 0% OCRBench score fixed at the architectural level.
- **B — Resolution 384→768** (free in params). `configuration.py` + `image_processing.py` default `image_size=768`; `vision_encoder.py` defaults `interpolate_pos_encoding=True` and gained `resize_pos_embeddings()`. 256 visual tokens/tile (was 64). Params ~262M (was ~290M).
- **C — Ngram repetition penalty**: `generate()` overridden to inject `no_repeat_ngram_size=20, repetition_penalty=1.1` by default (Unlimited-OCR trick).
- **D — Markdown-conversion training data**: `data/synthetic/markdown_dataset.py` generates prompt→target pairs (markdown / text / VQA / JSON) from synthetic docs with perfect ground truth; `data/real_benchmarks.py` converts OCRBench/FUNSD/CORD into training pairs; `data/build_training_dataset.py` merges them (target 50K+).
- **E — Full-model retrain infra**: `training/init_768_model.py` (builds 768 model, interpolates vision pos emb) + `training/full_train.py` (full fine-tune, no LoRA, at 768). Validated end-to-end on a 2507-pair pilot (loss ~10.8, decreasing). Full 50K/768 run is for Colab T4.
- Research write-up: `docs/model_improvements.md` (GLM-OCR, DeepSeek-OCR-2, Unlimited-OCR competitive analysis).

### Next (full 768 retrain — run on Colab T4)
> **M4 16GB constraint (empirical, 2026-07-16):** Full fine-tune does NOT run reliably on the M4. @768 it OOMs instantly / >7 min per step. @384 (bf16 + grad-checkpoint + `empty_cache` + seq 256) runs at ~0.2 steps/s but OOMs after ~100–110 steps due to MPS memory growth. Only **LoRA** (`training/fast_train.py`) is stable on this Mac (proven 17K steps). The 50K/768 full retrain must run on Colab T4.

**Preferred path — resumable Colab notebook** (`training/colab_full_retrain.ipynb`):
- Mounts Drive, clones `main` via zip, installs deps, downloads the 3 training benchmarks
  (OCRBench / FUNSD / CORD — DocVQA & SROIE are skipped; they aren't used for training),
  generates 50K synthetic markdown docs, initializes the 768 model, then full-trains 30K steps.
- Every stage is **guarded** (skips if its output exists), so *Runtime → Run all* resumes after a
  disconnect. Generation streams live progress; benchmark `load_dataset` has a 300s timeout + retries
  so it can't hang. Intermediate checkpoints are throttled (`--save-every 10000`) to avoid filling disk;
  only the final checkpoint is synced to Drive.
- Note: if the training *cell itself* is interrupted before `final` exists, re-running restarts that
  stage from scratch (no in-training resume yet). For a long T4 run, monitor or background it.

**Manual path (commands):**
1. Download training benchmarks: `python evaluation/download_benchmarks.py --data-dir evaluation/data --benchmarks ocrbench funsd cord` (~3–5 min; default now scopes to these three).
2. Generate the 50K+ set: `python data/build_training_dataset.py --num-docs 50000 --output-dir data/training --data-dir evaluation/data` → `data/training/manifest.jsonl`.
3. Init 768 model: `python training/init_768_model.py --out checkpoints/init_768`.
4. Retrain full model: `python training/full_train.py --model-id checkpoints/init_768 --manifest data/training/manifest.jsonl --steps 30000 --batch-size 4 --grad-accum 8 --device cuda --bf16 --grad-checkpoint`.
5. Eval on OmniDocBench V1.5, olmOCR-bench, OCRBench; push to HF (`eulogik/TinyDoc-VLM-768`) + PyPI.
6. (Optional) Dynamic multi-crop tiling, learned visual resampler, MTP loss, layout-stage wrapper.
7. Re-sync `demo/hf_space/tinydoc_vlm/` (still has old heads) and republish `eulogik/TinyDoc-VLM-768` on next Space deploy.

> **CI note (2026-07-16):** A/B/C raised the default resolution to 768, so `tests/test_model.py`
> asserts `image_size == 768` and `test_model_forward` uses 256 visual tokens/tile. All 13 tests pass.

---

## 8. Key Contacts

- **Company**: eulogik (https://eulogik.com)
- **Twitter**: @eulogik
- **Email**: hello@eulogik.com
- **Author**: Sunday Shah

---

*Built by [eulogik](https://eulogik.com) — AI infrastructure for document intelligence.*
