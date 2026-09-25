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

*Note: `evaluation/evaluate.py` OCRBench/FUNSD/CORD scorers still return hardcoded `0.0`
placeholders (marked in code) — the numbers in this file come from the dedicated runners below,
not from `evaluate.py`.*

---

## 2. Shipped product: grounded extraction pipeline (measured)

All rows: same 100 held-out SROIE receipt images, same scorer
(`evaluation/phase0/metrics.py`, field-level P/R/F1 over company/date/address/total).

| System | Field F1 | Notes / artifact |
|---|---|---|
| **TinyDoc pipeline (`ollama:qwen2.5vl:3b`, free & local)** | **0.870** | `results/scores_ollama.json` — schema 1.000, address 0.72 |
| PP-OCR (RapidOCR ONNX) + heuristics | 0.376 | `results/scores_ppocr_heuristics.json` — the free-competitor baseline |
| SmolVLM2-500M local baseline | 0.330 | `results/scores_smolvlm2.json` |
| Tesseract + regex | 0.227 | `results/scores_ocr_regex.json` |

Per-field hit rates (pipeline vs PP-OCR+heuristics): company 0.85 vs 0.28, date 0.94 vs 0.52,
address 0.72 vs 0.03, total 0.97 vs 0.60.

Supporting product metrics (n=100):

| Metric | Score | Artifact |
|---|---|---|
| Schema validity | 1.000 | `results/scores_ollama.json` |
| Evidence coverage (RapidOCR, after A/B) | 0.798 | `results/evidence_ab.json` |
| Evidence quote contains gold value | 0.535 | `results/evidence_ab.json` (tesseract arm: 0.668 / 0.415 → RapidOCR adopted) |
| FUNSD transfer probe (form→receipt keys) | 0.352 | `results/scores_funsd_ollama.json` — **nonstandard gold mapping** (date=answers[0]); a probe, not a leaderboard number |

Recompute: `python3 evaluation/phase0/recompute_scores.py` (or open the JSONs — `macro_prf` over
the committed preds/scores rows).

---

## 3. Methodology

- **Scorer:** ANLS + field-level P/R/F1 + schema-valid rate; identical for every engine, no
  engine-specific tuning of gold.
- **Held-out set:** 100 SROIE receipt images, images resolvable from stored `image_path`
  (untracked symlink to the dataset).
- **Baselines:** PP-OCR+heuristics is the honest free local competitor (OCR + rule extraction,
  no VLM, no API, no GPU); its heuristics are standard (business-token company line, date regex,
  total-near-"total", street/postal-cue address) with no per-doc tuning and no gold leakage.
- **Engine latency (MPS, per doc):** 3b ≈ 2.3 s; 7b ≈ 45 s (in progress, `results/`).

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
