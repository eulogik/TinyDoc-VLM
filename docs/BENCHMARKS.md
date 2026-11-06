# TinyDoc-VLM Benchmark Results

> **Rewritten 2026-09-24 — measured numbers only.**  
> Earlier versions of this file (and `docs/paper.md`, `docs/benchmark_results.json`,  
> the pitch/reddit/twitter/launch drafts) carried **unmeasured estimates**  
> (DocVQA ~55%, FUNSD ~70%, …) presented as results. They are withdrawn.  
> Every number below recomputes from a committed artifact.

---

## 1. Research checkpoint: TinyDoc-VLM-256M is retired from claims

Full OCRBench evaluation (all 1,000 samples, all 10 categories, greedy decoding):

| Metric | Score |
|---|---|
| OCRBench strict accuracy | **0.0%** (0/999 scored, 1 loader error) |
| Unique predictions | 962/999 → output is image-conditioned, but never correct |

Artifacts: `evaluation/phase0/results/ocrbench_256m_full.json` (+ `.jsonl` per-sample), pilot `ocrbench_256m_pilot.*`.
Gates: `node scripts/verify-ocrbench.mjs controls|vision|tokenid|pilot|full|recompute` — all pass.

**What was verified (not speculation):**

1. The vision tower is alive: encoder features have std≈1 and differ across images
   (cross-image cosine −0.57). The dead-tower defect affected the 768-run checkpoints, not this one.
2. The processor is aligned (`image_token_id=49152` resolved from the rebuilt processor; missing
   `<image>` tokens in the prompt were a harness bug, fixed — it does not explain the failures).
3. The model degenerates under **every** prompt, **including the exact training format**
   `"Extract document information: <image>"`: `no_repeat_ngram_size=6` or
   `repetition_penalty=1.2` turn output into fluent word-salad unrelated to the image;
   without them it loops (`"$ "$ "$"`, `22222`). The LoRA adapter is worse (literal degeneration).
4. The checkpoint still contains dropped `output_heads.*` weights (UNEXPECTED on load).

**Conclusion:** the training run itself failed (insufficient data / broken loss), not the harness.
The 256M weights remain on HuggingFace as a **research artifact**; they are not a product.

*Note (2026-11-06): `evaluation/evaluate.py` is deprecated. Its OCRBench/FUNSD/CORD
scorers now return `None` / `not_measured`, not zeros. The numbers in this file come
from the dedicated runners below, never from `evaluate.py`.*

---

## 2. Shipped product: grounded extraction pipeline (measured)

All rows: same 100 held-out SROIE receipt images, same scorer
(`evaluation/phase0/metrics.py`, field-level P/R/F1 over company/date/address/total).

> ⚠️ **The 100-doc SROIE set is NOT a clean held-out set** (`eval_leakage_audit.json`):
> 60% of its receipts also appear in the public SROIE *train* split (perceptual match across two
> resamplers; exact hashing missed the re-encoded duplicates). Absolute scores are optimistically biased for any
> model trained on public SROIE, possibly including the evaluated 3B. The **paired** pipeline-vs-baseline
> comparison is still apples-to-apples because every engine saw the identical documents.
> A verified-clean 100-doc set (`sroie_eval_clean.json`) is the bar for first-party models.

| System | Field F1 | Notes / artifact |
|---|---|---|
| **TinyDoc pipeline (`ollama:qwen2.5vl:3b`, free & local)** | **0.870** | `results/scores_ollama.json` — schema 1.000, address 0.72 |
| PP-OCR (RapidOCR ONNX) + heuristics | 0.376 | `results/scores_ppocr_heuristics.json` — the free-competitor baseline |
| SmolVLM2-500M local baseline | 0.330 | `results/scores_smolvlm2.json` |
| Tesseract + regex | 0.227 | `results/scores_ocr_regex.json` |

First-party edge track (verified-clean 100, `results/sroie_eval_clean.json`):

| System | Field F1 | Notes / artifact |
|---|---|---|
| **Hybrid v2 product path (`hybrid_2p2b`: MLX SmolVLM2-2.2B 4-bit + tesseract F3 address)** | **0.603 — SHIPPED v2.0 (2026-10-07)** | `results/scores_product_hybrid_clean.json` — schema 0.93, address 0.37 (beta, bar amended 0.50→0.35 with sign-off), address paired 31/7/62 (p=0.00012), per-doc F1 32/7/61, ~8s/doc; decision `results/v2_product_decision.json` |
| SmolVLM2-2.2B zero-shot (MLX 4-bit) | 0.536 | `results/scores_smolvlm2-2.2b_base_clean.json` — address 0.13 |
| 2.2B LoRA SFT (gold / distill / r32 / 768px) | 0.535–0.540 | five configs, paired ≤5/5/90 vs base — SFT moves nothing; address 0.15–0.18 |
| 3B re-measured on clean-100 (reference, not shippable first-party) | 0.928 → **0.9475** | `results/scores_pipeline_ollama_qwen2.5vl_3b_clean.json` had 5 HTTP-400 context rejections; corrected 8k-context rerun `results/scores_pipeline_ollama_qwen2.5vl_3b_clean_ctx8k.json` — address 0.86, schema 1.00, 0 errors |
| Qwen3-VL-2B-Instruct (fast tier, reference) | **0.9200** | `results/scores_pipeline_ollama_qwen3-vl_2b-instruct_clean_ctx8k.json` — address 0.84, schema 1.00, 0 errors, 2.0× faster than 3B |

Per-field hit rates (pipeline vs PP-OCR+heuristics): company 0.85 vs 0.28, date 0.94 vs 0.52,
address 0.72 vs 0.03, total 0.97 vs 0.60.

> **2026-11-06 correction (pilot artifacts):** the earlier **0.928** 3B row was depressed by
> five Ollama HTTP-400 context rejections on 2481×3508 scans (Ollama defaults to 4096 tokens).
> With `TINYDOC_OLLAMA_NUM_CTX=8192`, the same 3B engine scores **0.9475** (address 0.86, schema
> 1.00, 0 errors) and Qwen3-VL-2B-Instruct scores **0.9200** (address 0.84, 2.0× faster).
> The pilot artifacts are committed and verified by `recompute_scores.py`; analysis in
> `docs/v3_plan.md`.

Supporting product metrics (n=100):

| Metric | Score | Artifact |
|---|---|---|
| Schema validity | 1.000 | `results/scores_ollama.json` |
| Evidence coverage (RapidOCR, after A/B) | 0.798 | `results/evidence_ab.json` |
| Evidence quote contains gold value | 0.535 | `results/evidence_ab.json` (tesseract arm: 0.668 / 0.415 → RapidOCR adopted) |
| Confidence ECE, field, 3B, evidence off (eval path) | **0.545** | `results/confidence_probe_qwen2.5vl_3b_clean.json` — constant 0.4025 |
| Confidence ECE, field, 3B, evidence on (default) | **0.135** | `results/confidence_probe_qwen2.5vl_3b_clean_evidence.json` |
| Confidence ECE, field, 3B, isotonic OOF | **0.014** | `results/calibration_qwen2.5vl_3b_clean.json` — 5 doc-grouped folds |
| Confidence ECE, field, 2B, evidence off (eval path) | **0.518** | `results/confidence_probe_qwen3-vl_2b-instruct_clean.json` — constant 0.4025 |
| Confidence ECE, field, 2B, evidence on (default) | **0.115** | `results/confidence_probe_qwen3-vl_2b-instruct_clean_evidence.json` |
| Confidence ECE, field, 2B, isotonic OOF | **0.005** | `results/calibration_qwen3-vl_2b-instruct_clean.json` — 5 doc-grouped folds |
| VLM bbox hit@0.5 (Qwen3-VL-2B) | **0.20** | `results/grounding_probe_qwen3-vl_2b-instruct.json` — 100 synthetic boxes; gate is 0.70 |
| VLM bbox hit@0.5 (Qwen2.5-VL-3B) | **0.04** | `results/grounding_probe_qwen2.5vl_3b.json` — same set |
| FUNSD transfer probe (form→receipt keys) | 0.352 | `results/scores_funsd_ollama.json` — **nonstandard gold mapping** (date=answers[0]); a probe, not a leaderboard number |

The bake-off ran with evidence **off** for speed. That path produces a constant 0.4025
and is not the SDK default; its ECE (0.52–0.55) is a property of the eval harness, not
the shipped pipeline. On the evidence-on default, field ECE is 0.115–0.135 raw and
0.005–0.014 after document-grouped isotonic OOF. Document-level exact-4 confidence is
**not** calibrated (ECE 0.092–0.097; isotonic improves the 3B to 0.060 but makes the 2B
worse at 0.106). Use field-level confidence for review routing, never as a document
probability. All calibration numbers are out-of-fold on clean-100, not an external test.

VLM-predicted boxes are **not** shipped. On 100 boxes from 30 synthetic rendered documents
(seed `20261106`), hit@0.5 was **0.20** for Qwen3-VL-2B and **0.04** for Qwen2.5-VL-3B,
against a 0.70 product gate. Evidence remains OCR-span only. This is a synthetic probe,
not a public benchmark.

Recompute: `python3 evaluation/phase0/recompute_scores.py` (or open the JSONs — `macro_prf` over
the committed preds/scores rows).

Machine-readable provenance and SHA-256 hashes for every eval set:
`results/eval_provenance.json` (guarded by `tests/test_eval_provenance.py`).

---

## 3. Methodology

- **Scorer:** ANLS + field-level P/R/F1 + schema-valid rate; identical for every engine, no
  engine-specific tuning of gold.
- **Eval set:** 100 SROIE receipts from the public train pool (duplicate-filtered vs our
  holdout) — **not an external test**; provenance and hashes in `results/eval_provenance.json`.
- **Baselines:** PP-OCR+heuristics is the honest free local competitor (OCR + rule extraction,
  no VLM, no API, no GPU); its heuristics are standard (business-token company line, date regex,
  total-near-"total", street/postal-cue address) with no per-doc tuning and no gold leakage.
- **Engine latency (M4, Ollama Q4, warm, per doc, clean-100):** 3B ≈ 14.1 s; 2B ≈ 7.0 s.
  The v2 hybrid is ≈8 s/doc on the MLX engineering track. A 7B pilot was rejected (identical
  F1, 4.5× slower, unstable on the 16 GB host).

---

## 4. Reproducibility

```bash
# Recompute every F1 from committed artifacts (no model calls)
python3 evaluation/phase0/recompute_scores.py

# PP-OCR competitor baseline ( RapidOCR venv + 100-doc eval )
python3 evaluation/phase0/run_ppocr_baseline.py

# Full pipeline eval
python3 evaluation/phase0/run_pipeline_eval.py --limit 100

# OCRBench gates (1000-sample artifacts already committed)
node scripts/verify-ocrbench.mjs all
```

---

*Contact: eulogik · repo: github.com/eulogik/TinyDoc-VLM ·
decision record: `docs/pivot_plan.md`.*
