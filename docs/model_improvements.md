# TinyDoc-VLM: Competitive Analysis & Improvement Plan

**Date:** 2026-07-16
**Goal:** Close the gap with SOTA document VLMs (GLM-OCR, DeepSeek-OCR-2, Unlimited-OCR) by a *good margin* while keeping the ≤256M / CPU-friendly advantage.

---

## 1. SOTA Landscape (what the leaders do)

### GLM-OCR — `zai-org/GLM-OCR` (1B, MIT)
- **Architecture:** GLM-V encoder–decoder. **CogViT** visual encoder (pretrained on large image–text data) + lightweight cross-modal connector with efficient token downsampling + **GLM-0.5B** decoder.
- **Training tricks:** **Multi-Token Prediction (MTP) loss** + stable full-task **RL**.
- **Pipeline:** two-stage — layout analysis with **PP-DocLayout-V3** then parallel recognition.
- **Output:** prompt-routed. "Text Recognition:" / "Formula Recognition:" / "Table Recognition:" for parsing; strict **JSON schema** prompt for information extraction.
- **Results:** **94.62 on OmniDocBench V1.5 (#1)**, 75.2 on olmOCR-bench. Throughput 1.86 PDF pages/s, 0.67 img/s.
- **Lesson for us:** single decoder that emits markdown/text/JSON via prompt routing; layout stage is optional tooling, not baked into the model.

### DeepSeek-OCR-2 — `deepseek-ai/DeepSeek-OCR-2` (3B, Apache-2.0)
- **Architecture:** "Visual Causal Flow" + **Contexts Optical Compression** (extreme learned visual-token compression).
- **Resolution:** **dynamic** — `(0-6)×768×768 + 1×1024×1024` → `(0-6)×144 + 256` visual tokens.
- **Output:** markdown conversion ("Convert the document to markdown.") and "Free OCR." Trained on massive synthetic corpus.
- **Results:** 76.3 olmOCR-bench, 41.2 ParseBench mean.
- **Lesson for us:** dynamic high resolution + markdown-as-universal-format is the dominant paradigm. Visual-token compression matters for long docs.

### Unlimited-OCR — `baidu/Unlimited-OCR` (builds on DeepSeek-OCR, MIT, 14.3k★)
- **"One-shot Long-horizon Parsing."**
- **Critical inference trick:** `no_repeat_ngram_size=35, ngram_window=128` → kills the repetition collapse on long document output.
- Multi-page / PDF support, vLLM + SGLang serving.
- **Lesson for us:** a tiny decoding-side change (ngram repetition penalty) is essential for readable long outputs. This is free and we don't have it.

---

## 2. TinyDoc-VLM Current State (from code)

| Component | Current | Issue |
|---|---|---|
| Vision encoder | SigLIP-B/16 @ **384×384** → 576 patches | Fixed low res; can't read small text. **#1 bottleneck.** |
| Compressor | PixelShuffle 3× → **64** visual tokens | Fine, but only 64 tokens for the whole page. |
| Decoder | SmolLM2-135M (576 hidden, 30 layers) | OK for size budget. |
| Output | **MultiTaskOutputHeads** (json/kv/table/ocr/qa) | **Fundamentally flawed** — OCRHead outputs to a 256-char vocab, KV/JSON/Table heads output 128–256 tokens. They cannot generate real text/markdown. This is the main reason for the 0% OCRBench. |
| Resolution | Single fixed 384 image, no tiling | No multi-crop. |
| Repetition control | None | Long outputs will loop (no ngram penalty). |
| Training data | 3K synthetic docs / 6,815 QA pairs (LoRA only) | Far too little; LoRA-only can't fix a broken head design. |

---

## 3. Prioritized Improvement Plan

### Tier 1 — Highest impact, lowest cost (do first)

**A. Drop the task heads; generate text/markdown/JSON from the decoder LM head.**
The heads are non-functional for real OCR (wrong vocab sizes). Replace with prompt-routed generation exactly like GLM-OCR / DeepSeek-OCR:
- `"Extract all text:"` → plain text
- `"Convert the document to markdown:"` → markdown
- `"Answer the question: {q}"` → short answer
- `"Extract fields as JSON: {schema}"` → JSON
Delete `output_heads.py`, route everything through `decoder.lm` head. Removes ~2M dead params, unifies capacity, matches SOTA.

**B. Raise resolution 384 → 768 (free in params).**
SigLIP-B/16 params don't change with input size — only patch count grows (576 → 2304), PixelShuffle 3× → **256 visual tokens** (4× more detail). Use `interpolate_pos_encoding=True`; optionally fine-tune the position embeddings. Vision encoder is still 93M. Slower on CPU (4× patches) but quality jumps massively. Keep 256M total budget.

**C. Add ngram repetition penalty at inference (Unlimited-OCR trick).**
In `generate()`: `no_repeat_ngram_size=20, repetition_penalty=1.1`. Zero training cost, fixes looping on long docs.

### Tier 2 — Retrain the full model (needed for real gains)

> **Status (2026-07-16): D, E implemented & validated end-to-end. A/B/C from Tier 1 are in.**
> - `data/synthetic/markdown_dataset.py` — generates markdown/text/VQA/JSON prompt→target pairs from synthetic docs (reuses `ContentGenerator` + PIL renderer). Verbose markdown transcription = perfect ground truth.
> - `data/real_benchmarks.py` — converts already-downloaded **OCRBench (1000 VQA), FUNSD (199 OCR), CORD (900 JSON/KIE)** into training pairs.
> - `data/build_training_dataset.py` — merges synthetic + real into `data/training/manifest.jsonl` (target 50K+).
> - `training/init_768_model.py` — builds a 768 model from the 384 base (copies decoder/compressor, interpolates vision pos emb 384→768).
> - `training/full_train.py` — full-model (no LoRA) training at 768 on the combined manifest.
> - Pilot: 300 synthetic docs → 1308 pairs + 1199 real = 2507 pairs; CPU full-train ran (loss ~10.8, dropped over steps). MPS full-fine-tune is too slow per step for M4 — the **real 50K/768 run should run on Colab T4** (see `training/full_train.py` usage).


**D. Markdown-conversion training data at scale.**
SOTA trains on millions of (rendered doc → markdown/HTML) pairs. Build a pipeline:
1. Author/collect documents as **markdown/HTML/LaTeX** (use real corpora: DocLayNet, PubLayNet, arXiv, Wikipedia, synthetic).
2. Render to image (our PIL renderer, or LaTeX/WeasyPrint → image).
3. Train pairs: `image + "Convert the document to markdown:" → markdown`.
Target **50K–200K** docs. Train **full model** (not LoRA-only) — the head removal makes this necessary.

**E. Mix in real benchmarks as training data.**
OCRBench (1000), FUNSD (199), CORD (900), SROIE (626), OmniDocBench samples, ICDAR. Convert each to prompt→answer pairs so the model sees eval-style tasks during training (standard practice).

**F. Train with a repetition/ordering-aware loss.**
Add unigram/char FocalLoss or a small n-gram regularization term during training to reduce repetition from the start (complements Tier-1C).

### Tier 3 — Advanced (optional, bigger effort)

**G. Dynamic multi-crop tiling (DeepSeek-OCR-2 style).**
Split a high-res scan into overlapping tiles, encode each, concatenate visual tokens with 2D position ids. Lets us handle A4 @ 300dpi without blowing up tokens. Needs tile + position handling in `vision_encoder.py` + `modeling.py`.

**H. Learned visual resampler (Q-Former / Perceiver).**
Replace fixed PixelShuffle with a small cross-attention resampler (e.g., 64 learned queries attending to patch features). Better compression than PixelShuffle for mixed layouts. ~3–5M params.

**I. Multi-Token Prediction (MTP) head (GLM-OCR insight).**
Auxiliary MTP loss on the decoder speeds convergence and improves accuracy. Medium complexity.

**J. Layout-stage tooling (optional, out-of-model).**
Wrap inference with a lightweight layout detector (e.g., a small YOLO or PP-DocLayout-V3) to crop regions before OCR — mirrors GLM-OCR's two-stage pipeline without bloating the 256M model.

---

## 4. Recommended Sequencing

1. **This week (no retrain):** A (remove heads) + C (ngram penalty). Re-evaluate the existing LoRA checkpoint via the LM head — may already recover a lot.
2. **Next (retrain):** B (768 res) + D/E (50K+ markdown data, full-model train). This is the "good margin" jump.
3. **Later:** G/H/I for ceiling gains; J as optional wrapper.

## 5. Why this wins
- **Resolution bump (B)** is the single biggest OCR lever and costs zero parameters.
- **Head removal (A)** fixes the actual cause of 0% and aligns us with every SOTA model.
- **Markdown training (D)** is what GLM-OCR/DeepSeek/Unlimited all converge on — one format, prompt-routed, generalizes across OCR/VQA/KIE/table.
- **ngram penalty (C)** is free and matches Unlimited-OCR's key long-output fix.

## 6. Benchmarks to track
- OmniDocBench V1.5 (SOTA uses this — GLM-OCR 94.62)
- olmOCR-bench (DeepSeek-OCR-2 76.3, GLM-OCR 75.2)
- OCRBench (we already have 1000 samples)
- FUNSD / CORD / SROIE (we have these)
- ParseBench (DeepSeek 41.2, GLM-OCR 29.6 mean — note: these are hard, long-form)
