# TinyDoc-VLM Walkthrough

> **Living document** — last updated: 2026-06-18
> Maintain this file as the single source of truth for project status, architecture decisions, and the execution plan.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Architecture Analysis](#3-architecture-analysis)
4. [Current Code State (Gap Analysis)](#4-current-code-state-gap-analysis)
5. [Execution Plan](#5-execution-plan)
6. [Component Details](#6-component-details)
7. [How to Run](#7-how-to-run)
8. [Benchmark Tracking](#8-benchmark-tracking)
9. [Decision Log](#9-decision-log)

---

## 1. Project Overview

**Goal:** Build a 150M–300M parameter vision-language model specialized **exclusively** for document understanding (invoices, receipts, forms, tables, charts, ID cards). It must run on a Raspberry Pi (<1GB VRAM, >100 tok/s on CPU).

**Strategy:** SmolVLM2 proved small LMs benefit little from huge encoders. TinyDoc-VLM pushes this further with document-specific architecture optimizations — layout-aware attention, pixel-shuffle token compression, structured output heads (JSON, KV, Table).

**Target benchmarks:** DocVQA >85%, OCRBench >75%, FUNSD >95% F1, CORD >95% F1.

---

## 2. Repository Structure

```
TinyDoc-VLM/
├── tinydoc_vlm/                  # Core Python package
│   ├── __init__.py               # Package init + HF Auto registration
│   ├── configuration.py          # TinyDocVLMConfig (PretrainedConfig)
│   ├── vision_encoder.py         # SigLIPVisionEncoder wrapper
│   ├── token_compressor.py       # PixelShuffleTokenCompressor + RMSNorm
│   ├── decoder.py                # TinyDocDecoder (LlamaForCausalLM wrapper)
│   ├── modeling.py               # TinyDocVLMForConditionalGeneration (main model)
│   ├── attention.py              # 2D sinusoidal positional embeddings
│   ├── image_processing.py       # TinyDocImageProcessor (resize, tile)
│   └── processing.py             # TinyDocVLMProcessor (image + text pipeline)
├── data/
│   ├── synthetic/
│   │   └── templates/            # HTML/CSS templates for doc generation
│   │       ├── invoice.html
│   │       └── receipt.html
│   └── tokenizer/
│       ├── special_tokens.py     # Script to extend SmolLM2 tokenizer
│       └── extended_tokenizer/   # Saved extended tokenizer
│           ├── tokenizer_config.json
│           ├── chat_template.jinja
│           └── tokenizer.json
├── tests/
│   └── test_model.py             # Unit tests (config, processor, forward pass)
├── pyproject.toml                # Build config + dependencies
├── setup.py                      # Editable install
├── requirements.txt              # Pinned dependencies
├── .gitignore
├── README.md
├── TinyDoc-VLM-Project-Document.md  # Original project doc (vision/market)
└── walkthrough.md                # ← YOU ARE HERE
```

---

## 3. Architecture Analysis

### 3.1 Current Architecture (as implemented)

```
Input Image → TinyDocImageProcessor → SigLIPVisionEncoder → PixelShuffleTokenCompressor → TinyDocDecoder (Llama)
```

| Component | What's implemented | What's planned but missing |
|-----------|-------------------|---------------------------|
| **Vision Encoder** | Thin wrapper around HF `SiglipVisionModel`. Accepts multi-tile input (batch, tiles, C, H, W). Returns (batch, tiles, patches, dim). | Custom tiny ViT (50M params), region-aware attention, layout pretraining, 8x8 patch high-res mode |
| **Token Compressor** | `PixelShuffleTokenCompressor` - space-to-depth + 2-layer MLP + RMSNorm. Works on square grids. | Document-aware pooling (merge tokens from same text line/table cell), layout token injection |
| **Decoder** | Wrapper around HF `LlamaForCausalLM` (SmolLM2). Passes through all kwargs. | Custom 8-layer decoder (not full Llama), structured output heads (JSON, KV, Table, OCR, QA), RoPE with document-specific base |
| **Positional Encoding** | `get_2d_sincos_pos_embed` - standard sinusoidal. Learned `visual_pos_embed` param added to compressed tokens. | 2D relative position bias in attention, layout-preserving cross-attention, ROI masking |
| **Image Processing** | Resize + pad + tile (max 2x2 grid + thumbnail). Returns (tiles, 3, H, W). | High-res mode, aspect-ratio-aware tiling, augmentation pipeline |
| **Processor** | Tokenizes text, expands `<image>` placeholder to N image tokens, processes images. | Config-driven scale/patch size (currently hardcoded), batch padding |

### 3.2 Architecture Gaps (Planned vs Actual)

The project document describes a **custom architecture** (tiny ViT encoder, custom 8-layer decoder, structured output heads, document-specific attention). The **actual implementation** is a composition of existing HF models (SigLIP + LlamaForCausalLM) with a pixel-shuffle connector. This is fine as an **MVP/prototype** but the custom architecture must be built to hit the benchmark targets.

### 3.3 Data Flow (forward pass)

1. `TinyDocVLMProcessor.__call__` takes text + PIL images
2. `TinyDocImageProcessor.preprocess` → tensor (num_tiles, 3, 384, 384)
3. `TinyDocVLMProcessor._expand_image_tokens` → replaces `<image>` with N copies in text
4. Tokenizer → `input_ids`, `attention_mask`
5. `TinyDocVLMForConditionalGeneration.forward`:
   - `SigLIPVisionEncoder(pixel_values)` → (B, N, 576, 768) for 384px, patch16
   - `PixelShuffleTokenCompressor` → (B, N, 64, 576) with scale=3 (576/3=192, 192^2... wait, 384/16=24, 24/3=8, 8^2=64)
   - Add `visual_pos_embed` → (B, 1, 64, 576)
   - Flatten tiles → (B, N*64, 576)
   - Overwrite image tokens in `inputs_embeds`
   - Forward through Llama decoder

### 3.4 Parameter Counts

| Component | Config Default | Approx Params |
|-----------|---------------|---------------|
| Vision Encoder | SigLIP-B/16 (12 layers, 768d, 12 heads) | ~93M |
| Token Compressor | scale=3, 768→576→576 | ~1M |
| Decoder | SmolLM2-135M (30 layers, 576d, 9 heads, GQA=3) | ~135M |
| **Total** | | **~229M** |

---

## 4. Current Code State (Gap Analysis)

### 4.1 What works ✅
- Model forward pass (tested)
- Image processing with tiling
- Pixel shuffle compression
- Processor pipeline (text + image → model inputs)
- Tokenizer extension with document special tokens
- HF AutoConfig/AutoModel registration
- 2D sinusoidal positional embeddings
- **Structured output heads** (JSON, KV, Table, OCR, QA) — implemented in `output_heads.py`
- **Training pipeline** — `trainer.py` (FSDP-ready, mixed-precision, checkpointing, scheduler), `data.py` (DocumentDataset + collate), `losses.py` (CombinedLoss for multi-stage)
- **Synthetic data generator** — `data/synthetic/generator.py` with Faker content generation, HTML→PDF→Image rendering, augmentation, and QA pair generation
- **HTML templates** — invoice, receipt, form, table (4 of 10 planned)
- **Evaluation suite** — `evaluation/evaluate.py` with DocVQA (ANLS), OCRBench, FUNSD, CORD harnesses
- **Training configs** — Stage 1/2/3 YAML configs in `training/`
- **Training launcher** — `training/run.py` with YAML config loading and CLI overrides
- **CI/CD** — `.github/workflows/ci.yml` with pytest + ruff
- **All 4 tests passing** — config, image processor, model forward, processor integration

### 4.2 What's implemented but needs fixes 🟡
- **`processor.py` hardcoded values** — ✅ FIXED: now reads from config (scale, patch_size, image_size)
- **`modeling.py` `visual_pos_embed` shape** — per-tile broadcast works but thumbnail gets same grid positions (minor)
- **`image_processing.py`** - tiling capped at 2x2 grid; no variable aspect ratio support
- **`decoder.py`** - uses full LlamaForCausalLM (SmolLM2-135M is 30 layers), not the planned 8-layer custom decoder
- **`TinyDocImageProcessor`** — ✅ FIXED: now inherits from `ImageProcessingMixin` for HF compatibility

### 4.3 What's still missing 🔴
- **Layout-aware attention mechanisms** (region-of-interest, hierarchical, layout-preserving) — not implemented
- **Export/Optimization** (ONNX, GGUF, CoreML) — no scripts exist
- **SDK** (client library for production) — not started
- **Demo** (Gradio interface) — dependency listed but no code
- **Flash attention** support — listed in requirements but not wired in
- **Additional HTML templates** — still need id_card, chart, contract, letter, medical, handwritten, mixed
- **Actual training run** — needs compute (8× A100 or equivalent)
- **Benchmark data** — public datasets need to be downloaded and formatted
- **Custom 8-layer decoder** — deferred to v2 (using SmolLM2-135M for MVP)

### 4.4 Code Quality Notes

| Aspect | Assessment |
|--------|-----------|
| Type hints | Good - all functions have typed signatures |
| Docstrings | Good - clear and informative |
| Style | Clean, follows PEP 8, consistent naming |
| Tests | 4 tests covering config, processor, forward, integration — good coverage for prototype |
| Error handling | Minimal (e.g., `_expand_image_tokens` does a simple `replace` which could match unintended occurrences) |
| Logging | None (uses print in special_tokens.py) |
| Configuration | Good - uses PretrainedConfig properly |
| Dependency management | Good - pyproject.toml + requirements.txt |

---

## 5. Execution Plan

### Phase 0: Foundation Fixes ✅
- [x] Fix processor hardcoded values (read from config)
- [x] Fix visual_pos_embed to account for tile dimensions
- [x] Fix test_processor_integration (vocab mismatch between model and tokenizer)
- [x] Fix TinyDocImageProcessor base class (inherit ImageProcessingMixin)

### Phase 1: Training Pipeline ✅  
- [x] Build `tinydoc_vlm/trainer.py` — training loop with FSDP support, mixed precision, checkpointing
- [x] Build `tinydoc_vlm/data.py` — DocumentDataset + collate_fn for variable-size tiles/text
- [x] Build `tinydoc_vlm/losses.py` — CombinedLoss for multi-stage (stage 1/2/3)
- [x] Write configuration YAMLs for all 3 training stages
- [x] Build `training/run.py` — config-driven training launcher with CLI overrides

### Phase 2: Synthetic Data Engine ✅  
- [x] Build `data/synthetic/generator.py` — Faker content gen + HTML render + PIL fallback
- [x] Add PIL-based document renderer (`data/synthetic/pil_renderer.py`) with 9 document type renderers
- [x] HTML templates: invoice, receipt, form, table, id_card, chart, contract, letter, medical, mixed (10/10)
- [x] QA pair generation per document type
- [x] Manifest output in JSONL format
- [ ] **Still needed:** LLM integration for richer text generation
- [ ] **Still needed:** Augraphy/Albumentations pipeline for advanced augmentation

### Phase 3: Structured Output Heads ✅  
- [x] Implement `tinydoc_vlm/output_heads.py` — JSONHead, KVHead, TableHead, OCRHead, QAHead
- [x] MultiTaskOutputHeads container with task routing
- [x] KV head with confidence scoring
- [x] All 5 heads tested with forward pass
- [ ] **Still needed:** Integrate heads into `TinyDocVLMForConditionalGeneration.forward`
- [ ] **Still needed:** Constrained decoding for JSON output
- [ ] **Still needed:** Joint multi-task training in the trainer

### Phase 4: Evaluation Suite ✅  
- [x] Build `evaluation/evaluate.py` — unified harness with ANLS, exact_match, fuzzy_match
- [x] DocVQA benchmark (ANLS metric)
- [x] OCRBench, FUNSD, CORD benchmark stubs
- [ ] **Still needed:** Download and cache public benchmark datasets
- [ ] **Still needed:** Build TinyDoc-Bench (proprietary 10K doc benchmark)
- [ ] **Still needed:** Integrate evaluation into training loop

### Phase 5: Export & Optimization 🔴  
- [ ] ONNX export with dynamic axes
- [ ] INT8/INT4 quantization
- [ ] GGUF export for llama.cpp/ollama
- [ ] CoreML export for Apple Silicon
- [ ] Build performance benchmark suite

### Phase 6: SDK & Demo 🔴  
- [ ] Build `tinydoc_vlm/sdk/` — lightweight client
- [ ] Build Gradio demo (`demo/app.py`)
- [ ] Add OpenRouter-compatible API endpoint

### Phase 7: Launch & Community 🔴  
- [ ] HuggingFace release with model card
- [ ] OpenRouter integration
- [ ] Documentation site
- [ ] Benchmark blog post
- [ ] Community setup (Discord/GitHub Discussions)

---

## 6. Component Details

### 6.1 Configuration (`TinyDocVLMConfig`)

```python
# Default config creates:
# - SigLIP-B/16 vision encoder: 768d, 12 layers, 12 heads, patch16, 384px
# - SmolLM2-135M decoder: 576d, 30 layers, 9 heads, 3 KV heads, 8K context
# - Pixel shuffle scale: 3
#
# Total: ~229M params (93M vision + 1M connector + 135M decoder)
```

**Key fields:**
- `vision_config`: dict or PretrainedConfig for the vision encoder
- `decoder_config`: dict or PretrainedConfig for the language decoder
- `pixel_shuffle_scale`: int (default 3) — compression factor
- `image_size`: int (default 384) — input resolution
- `patch_size`: int (default 16) — ViT patch size

### 6.2 Pixel Shuffle Compression

Given patches arranged in a grid of `(G, G)` where `G = image_size / patch_size`:
1. Reshape to `(G//s, s, G//s, s, C)` where `s = pixel_shuffle_scale`
2. Permute to `(G//s, G//s, s, s, C)`
3. Reshape to `((G//s)^2, s*s*C)` — groups `s×s` patches into one
4. Project with 2-layer MLP: `s*s*C → decoder_dim → decoder_dim`
5. Apply RMSNorm

**Effect with defaults:** `384/16 = 24` → grid `24×24 = 576` patches. With `scale=3` → `(24/3)^2 = 8^2 = 64` compressed patches. Compression ratio = `576/64 = 9×`.

### 6.3 Tile Processing

For images larger than `image_size`, the processor:
1. Creates a thumbnail (1 tile at `image_size×image_size`)
2. Splits into at most a 2×2 grid of tiles
3. Returns `(1 + rows*cols, 3, H, W)`

**Example:** 800×600 image → thumbnail + 2×2 grid = 5 tiles.

### 6.4 Visual Token Injection

The model uses `image_token_id` (default 49153) as a placeholder in `input_ids`. During forward:
1. All image tokens are replaced with visual features via embeddings
2. `inputs_embeds[b, mask] = flat_visual_features[b, :num_places]`
3. The decoder receives `inputs_embeds` instead of `input_ids`

**Caveat:** If `num_places` < visual features, features are truncated. If `num_places` > visual features, leftover placeholders remain as random embeddings.

---

## 7. How to Run

### Installation
```bash
pip install -e .
pip install flash-attn --no-build-isolation  # optional GPU
```

### Test
```bash
pytest tests/ -v
```

### Quick Inference (conceptual — generation untested end-to-end)
```python
from PIL import Image
from tinydoc_vlm import TinyDocVLMConfig, TinyDocVLMForConditionalGeneration, TinyDocVLMProcessor

config = TinyDocVLMConfig()
model = TinyDocVLMForConditionalGeneration(config)
processor = TinyDocVLMProcessor()

img = Image.open("invoice.png")
inputs = processor(text="Extract details: <image>", images=img)
outputs = model.generate(**inputs, max_new_tokens=256)
print(processor.tokenizer.decode(outputs[0]))
```

---

## 8. Benchmark Tracking

| Benchmark | Current SOTA (tiny) | Our Target | Our Current | Best Checkpoint |
|-----------|---------------------|-----------|-------------|-----------------|
| DocVQA (ANLS) | 70.5% (SmolVLM2-500M) | >85% | — | — |
| OCRBench | 52.6% (SmolVLM2-256M) | >75% | — | — |
| FUNSD (F1) | 93.76% (LayoutLMv3) | >95% | — | — |
| CORD (F1) | 97.23% (LayoutLMv3) | >95% | — | — |
| SROIE (F1) | 96.5% (LayoutLMv3) | >95% | — | — |
| ChartQA | ~65% (SmolVLM2-500M) | >75% | — | — |
| Table Extract | ~70% (SmolVLM2-500M) | >85% | — | — |
| VRAM | 0.8GB (SmolVLM2-256M) | <1GB | ~1.5GB* | — |
| CPU Speed | ~50 tok/s | >100 tok/s | — | — |

*\*Estimated from parameter count*

---

## 9. Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-06-18 | Use HF SigLIP + LlamaForCausalLM for MVP | Faster iteration; validates data pipeline and training before building custom architecture |
| 2026-06-18 | Pixel shuffle scale=3 default | 9× compression (576→64 tokens) balances quality vs sequence length |
| 2026-06-18 | Tile processing (thumb + 2×2 grid) | Handles high-res documents without modifying encoder architecture |
| 2026-06-18 | Processor expands `<image>` to N tokens in text | Clean separation between text and vision; standard tokenizer handles it |
| 2026-06-18 | Skip custom decoder for now | SmolLM2-135M is proven; custom 8-layer decoder can come in v2 |
| 2026-06-18 | Create walkthrough.md as living doc | Single source of truth for humans and AI agents |
| 2026-06-18 | Build training pipeline (trainer/data/losses) | Critical path to actual training; enables iterative experimentation |
| 2026-06-18 | Build synthetic data generator with Faker | Quick start for data generation; can upgrade to LLM-based later |
| 2026-06-18 | Build structured output heads (5 heads) | Key architectural differentiator; enables multi-task training |
| 2026-06-18 | Fix TinyDocImageProcessor base class | Required for HF ProcessorMixin compatibility |
| 2026-06-18 | Defer custom 8-layer decoder to v2 | SmolLM2-135M proven; allows focusing on data+training+heads first |
| 2026-06-18 | Defer export/optimization to Phase 5 | Requires trained model first to validate correctness |
| 2026-06-19 | Build PIL-based document renderer as primary renderer | Playwright/WeasyPrint broken on this macOS; PIL is reliable, fast, and has zero deps |
| 2026-06-19 | Wire MultiTaskOutputHeads into model forward | Enables multi-task training with task-specific routing |
| 2026-06-19 | All 10 document type templates complete | invoice, receipt, form, table, id_card, chart, contract, letter, medical, mixed |
| 2026-06-19 | Synthetic generator fully working end-to-end | Generates images + manifest.jsonl with QA pairs |

---

## 10. What's Next (Immediate Next Steps)

### Priority 1: Generate synthetic data at scale ✅
```bash
# Running: 10K docs via PIL renderer (~9 min)
nohup python data/synthetic/generator.py --num-docs 10000 --output-dir data/synthetic/output --seed 42 --no-augment &
```

### Priority 2: Run Stage 1 pretraining
```bash
# After 10K docs generated:
python training/run.py --config training/stage1_layout_pretrain.yaml
```

### Priority 3: Build demo app ✅
```bash
# Gradio demo with JSON/KV/Table/OCR extraction
python demo/app.py --model-path checkpoints/best
```

### Priority 4: Download benchmark datasets
```bash
python evaluation/download_benchmarks.py --data-dir evaluation/data
```

### Priority 5: Export to ONNX/GGUF (after training)
```bash
python export/export_onnx.py --model-path checkpoints/best --output tinydoc-vlm.onnx
python export/export_gguf.py --model-path checkpoints/best --output tinydoc-vlm.gguf
```

### Priority 6: Push CI workflow (requires PAT with `workflow` scope)
The `.github/workflows/ci.yml` exists locally but the PAT token lacks `workflow` scope.
To enable CI, either:
- Regenerate the PAT with `workflow` scope, or
- Push the file manually via the GitHub web interface

---

## 11. Key Open Questions

1. **Custom vs pretrained:** Should we train a custom tiny ViT + tiny decoder from scratch, or fine-tune SmolVLM2 checkpoints? Decision: for v1, fine-tune SmolVLM2-256M with our document pipeline.
2. **Synthetic data quality:** At what point does synthetic data quality bottleneck performance? Need iterative human validation.
3. **Structured output heads:** Joint training vs post-hoc fine-tuning? Decision: start with post-hoc (fine-tune on structured data), move to joint for v2.
4. **Multilingual:** English-only for v1, multilingual adapters for v2.
5. **Edge deployment target:** Raspberry Pi 5 as primary target; iPhone 15 as stretch goal.
