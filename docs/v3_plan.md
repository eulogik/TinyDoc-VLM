# TinyDoc-VLM v3 — Deep Review and Way Ahead

**Date:** 2026-11-06
**Status:** Decision document — evidence-based review + recommended next version
**Audience:** Project owner (eulogik)
**Scope:** Whole repo: model track, eval track, SDK/product surface, public claims

---

## 0. One-page verdict

**The evaluation and product shell are real. The 256M and SmolVLM2-2.2B model tracks are not.**

Three measured facts decide the next version:

1. **The from-scratch tiny model is dead.** TinyDoc-VLM-256M scored **0.0% OCRBench** (0/999 scored, 962/999 unique outputs — image-conditioned but never correct). The 768 retrain had a **dead vision tower**. External evidence agrees: SmolDocling-256M scores **0.493 overall edit distance (EN)** on OmniDocBench — not a usable end-to-end reader.
2. **The SmolVLM2-2.2B SFT track is also dead.** Base **0.536** clean-100 field F1; every language-LoRA config tested on the clean-100 lands **0.535–0.540** (gold, 768px, r32, teacher distillation); 22 val-40 configs looked like **0.552–0.621** but did not transfer; vision-LoRA at 5e-5 **0.0395** (template collapse); vision-LoRA at 1e-6 **0.4277** (below base). Zero real gain across the track.
3. **A free 2B/3B already clears the product bar.** On the clean-100, **Qwen2.5-VL-3B scores 0.9475** field F1 (address 0.86) and **Qwen3-VL-2B-Instruct scores 0.9200** (address 0.84) at **half the latency**; the shipped first-party hybrid scores **0.603** (address 0.37). The first-party gap is **perception**, not task understanding.

**Therefore the next version should not be another 256M/2.2B fine-tune.** It should be:

> **v3.0 — a perception-strong 2B–3B VLM (Qwen3-VL-2B / Qwen2.5-VL-3B class) + a grounded extraction trust layer, measured on a multi-vertical, provenance-audited benchmark.**
>
> **v3.1 — a TinyDoc-2B grounded LoRA only if v3.0 leaves a measured residual.**
>
> **v4 — a 0.9B “tiny” parser/extractor only after the trust layer and data engine are proven.**

The product moat is not the base model. It is **field-level accuracy + bbox/quote evidence + calibrated confidence/abstention + vertical schemas + honest eval**. The repo already has most of that machinery; it has not been wired to a model that can read, and neither confidence calibration nor VLM boxes ship yet (measured below).

---

## 1. Evidence ledger

### 1.1 Model track — what was measured

| Track | Best measured result | Artifact | Verdict |
|---|---|---|---|
| 256M from scratch (384) | **OCRBench 0.0%** (0/999; 1 loader error) | `evaluation/phase0/results/ocrbench_256m_full.json` | Dead; retired from claims |
| 256M 768 retrain | Vision tower dead at init (constant features) | `docs/BENCHMARKS.md`, `docs/pivot_plan.md` | Abandoned |
| SmolVLM2-500M zero-shot | 0.219 clean-100 | `scores_smolvlm500m_zeroshot.json` | Floor |
| SmolVLM2-500M LoRA | 0.229 clean-100 | `scores_smolvlm500m_lora_clean.json` | +0.01, noise; not viable |
| SmolVLM2-2.2B base | **0.5357** clean-100; address 0.13; ANLS 0.7565 | `scores_smolvlm2-2.2b_base_clean.json` | Weak perception |
| 2.2B language LoRA (4 clean-100 configs: gold / 768px / r32 / distill) | **0.5352–0.5395** clean-100 | `scores_smolvlm2-2.2b_lora_clean*.json` | **No gain vs base** |
| 2.2B val-40 language LoRA (22 configs: s2e/f/g/h/i) | **0.5517–0.6207** val-40 | `scores_smolvlm2-2.2b_lora_s2*.json` | Val looked better; did not transfer to clean-100 |
| 2.2B teacher distillation | 0.5395 clean-100; address 0.15 | `scores_smolvlm2-2.2b_lora_clean_distill.json` | No gain |
| 2.2B vision LoRA, 5e-5 | **0.0395** (40-doc val); fluent template placeholders | `scores_smolvlm2-2.2b_kv_kreal100val.json` | Catastrophic collapse |
| 2.2B vision LoRA, 1e-6 | **0.4277** (40-doc val); flat 50/75/100 | `scores_smolvlm2-2.2b_kv_kll*.json` | Below base |
| Hybrid v2 (2.2B + Tesseract F3 address) | **0.6028** clean-100; address 0.37; schema 0.93 | `scores_product_hybrid_clean.json` | Shipped v2; not competitive |
| **Qwen2.5-VL-3B (free, local, ctx 8k)** | **0.9475** clean-100; address 0.86; ANLS 0.9715; schema 1.00; 14.2 s | `scores_pipeline_ollama_qwen2.5vl_3b_clean_ctx8k.json` | **Best measured system** |
| **Qwen3-VL-2B-Instruct (free, local, ctx 8k)** | **0.9200** clean-100; address 0.84; ANLS 0.9340; schema 1.00; 7.0 s | `scores_pipeline_ollama_qwen3-vl_2b-instruct_clean_ctx8k.json` | Fast tier; 2× faster, 2.75 pts behind (95% CI −5.25…−0.50) |
| Qwen2.5-VL-3B, original 100-doc set | 0.870; address 0.72 | `scores_ollama.json` | Optimistically biased (60% leak) |
| PP-OCR + heuristics | 0.3763 | `scores_ppocr_heuristics.json` | Free competitor floor |
| Tesseract + regex | 0.2270 | `scores_ocr_regex.json` | Floor |
| FUNSD transfer probe | 0.3523 (n=50) | `scores_funsd_ollama.json` | Nonstandard gold mapping; probe only |

**Interpretation:** the 2.2B base is not “almost there.” Its address hit rate is **0.13**, and single-character errors dominate. No amount of task-format SFT fixes that, because the failure is reading, not formatting.

### 1.2 The error analysis says “perception,” not “reasoning”

On the 100 clean-100 base predictions (`preds_smolvlm2-2.2b_base_clean.jsonl`):

| Field | Hit rate | Mean ANLS | Mean CER | Empty | Median length (pred / gold) |
|---|---:|---:|---:|---:|---:|
| company | 0.490 | 0.811 | 0.170 | 0 | 24 / 26 |
| date | 0.770 | 0.829 | 0.147 | 0 | 10 / 10 |
| address | **0.130** | 0.599 | **0.320** | 0 | 56 / 64 |
| total | 0.750 | 0.787 | 0.180 | 1 | 5 / 5 |

Address failure taxonomy (87 misses): **35 partial**, **24 wrong/hallucinated**, **22 mostly-right-format**, **6 prefix**. Many are one-glyph errors:

- `8110` vs `81100`, `45600` vs `45000`, `47400` vs `4700`
- `JALAN DEKO` vs `JALAN DEDAP 13`
- `BANGUNAN` vs `BANGUN`, `PERINDUSTRIAN` vs `PRINCIPAL`
- `NO 29` vs `NO 290`

These are exactly the errors a **stronger visual encoder + higher effective resolution** fixes, and a language-only LoRA does not. Qwen2.5-VL-3B gets address **0.86** on the same documents because its native dynamic-resolution vision path reads the glyphs.

### 1.3 The eval is narrower and more contaminated than the docs imply

This is the part that needs a truth reset before any more model claims.

1. **The 0.870 headline is on a 60%-leaked set.** `eval_leakage_audit.json`: 61/100 eval receipts also appear in the public SROIE **train** split (perceptual corr ≥ 0.90; exact hashing misses re-encoded copies). The README does disclose this, but the website headline does not.
2. **The “clean-100” is not an external test set.** `sroie_eval_clean.json` has `source: sroie_public_train_clean` — it is a 100-doc split carved from the **same public SROIE train pool** after removing duplicates of the original 100 (`build_stage1_manifest.py`). It is clean **relative to our holdout**, not relative to Qwen’s pretraining. Any model trained on public SROIE — including possibly Qwen2.5-VL-3B — may have seen it.
3. **The 40-doc val is too small to select on, and on the 2.2B track it was easier than the test.** Base: val 0.600 vs test 0.536 (−6.4 pts). The 22 val-40 SFT configs reached 0.552–0.621, but every language-LoRA config tested on the clean-100 landed 0.535–0.540; the val-selected “best” did not survive the test. On the Qwen bases, val-40 and clean-100 agree closely (both 0.9250 val; 0.9200/0.9475 test), so val is a sanity check, not a substitute for the 100-doc test. With n=40, a 0.60 field hit rate has a 95% CI of roughly ±15 points; n=40 cannot resolve 1–3 point differences.
4. **The earlier 3B clean-100 run had 5 errors** (`clean_0017`–`clean_0021`). They were **Ollama HTTP 400 context-window rejections**, not timeouts: 2481×3508 scans need ~4.2k visual tokens and Ollama defaults to 4096. Fixing the harness (`TINYDOC_OLLAMA_NUM_CTX`, default 8192) changed the 3B reference from **0.9282 → 0.9475** and the 2B from **0.9061 → 0.9200**; both reruns have zero errors and schema 1.00. **Re-run any older Ollama number before quoting it** — the default context was silently truncating large scans.
5. **No real multi-vertical test.** CORD images are empty on disk; FUNSD is a nonstandard 4-key remap; the invoice benchmark is synthetic and page-1-only; there is no real-supplier invoice set, no full PDF/multi-page test, no abstention test, no bbox mAP. “Superb usable” cannot be claimed from one receipt layout family.

### 1.4 The public product surface is stale

The repo says v2 is “shipped,” but the public surface still ships v1:

| Surface | What it actually serves | Evidence |
|---|---|---|
| **PyPI `tinydoc`** | **0.1.2 (2026-06-24)** — old `TinyDocExtractor` / 256M, no `ReceiptPipeline`, no schemas, no evidence | `pypi.org/pypi/tinydoc/json`; repo `sdk/setup.py` is 0.1.3, unpublished |
| **HF Space** | Runs `eulogik/TinyDoc-VLM-256M` (retired, 0.0% OCRBench) | `demo/hf_space/app.py`; live Space raw `app.py` |
| **README hero** | “256M … Runs on CPU · <1GB VRAM … OCR, VQA, Form Extraction, Table Parsing” | `README.md` lines 1–35; the working engine is 2.2B/3B |
| **Website** | “Field F1 0.870 on held-out SROIE” with no leakage caveat; “quality tiers up to 7B” (7B was rejected: crashed the 16GB host) | `docs/index.html` lines 86–95, 161–174 |
| **Model card** | Honest about retirement, but points users to `pip install tinydoc` for the working product — which is not the working product | HF model card raw README |
| **`evaluation/evaluate.py`** | Still returns hardcoded `0.0` placeholders for OCRBench/FUNSD/CORD | `docs/BENCHMARKS.md` note |

**This is the biggest usability failure in the project.** A superb model behind a broken `pip install` is not usable. Fix this before training anything.

### 1.5 What is genuinely good (keep and build on it)

- **Leakage audit methodology** (`build_stage1_manifest.py`): perceptual dedupe across two resamplers, union-find on duplicate components, explicit integrity checks. This is better than most published doc-AI repos.
- **Claim audit** (`recompute_scores.py`): 9 required pairs + 4 optional bake-off pairs recompute; exits 0. Extend it, don’t replace it.
- **Eval harness** (`evaluation/phase0/`): field P/R/F1, ANLS, schema-valid, exact-4/4, per-field hit, paired decisions. Reusable.
- **Product layer**: schema validation, `sanitize_fields`, `_extract_json` salvage, evidence spans, overlay, CLI, confidence. Real code, 123 tests passing, CI-scope ruff clean.
- **Docs honesty**: `BENCHMARKS.md`, model card, and pivot plan explicitly retire false numbers. Rare and valuable.

**The project’s problem is not lack of rigor. It is that rigor has been applied to a base model that cannot read, while the public surface still advertises the dead model.**

### 1.6 Claim-audit findings to fix

These are small but matter for “everything evidence based”:

1. **Paired hybrid counts (fixed 2026-11-06).** Shipped path `preds_product_hybrid_clean.jsonl`: address **31/7/62** (p=0.000116); per-doc F1 **32/7/61**. The harness path `preds_hybrid-2.2b-tess_clean.jsonl` was **31/8/61** (p=0.00029). `v2_product_decision.json` now records the basis for both; README/BENCHMARKS quote the shipped path. `recompute_scores.py` should verify these counts.
2. **“Verified-clean 100” provenance (fixed 2026-11-06).** `results/eval_provenance.json` records source, leakage, counts, and SHA-256 for every set; `tests/test_eval_provenance.py` guards it. Docs call the set “SROIE public-train-derived clean-100,” not an external test.
3. **The 0.870 headline should be retired from the website** or carry the 60% caveat inline. The honest headline is **0.9475** (Qwen2.5-VL-3B) or **0.9200** (Qwen3-VL-2B) on clean-100, or **0.603** (first-party hybrid, clean-100).
4. **`AGENTS.md` says 69 tests; actual 82.** Update.
5. **`docs/index.html` says 7B quality tier**; engine ladder decision says 3B stays, 7B crashes the 16GB host. Update.
6. **PyPI 0.1.2 vs repo 0.2.0.** The 0.2.0 wheel/sdist build and install cleanly; upload is pending a PyPI token. Until then, the README points at the source install.

### 1.7 Confidence: field-level calibration is viable, document-level is not (measured 2026-11-06)

The clean-100 bake-off ran with evidence **off** for speed. That path produces a constant
score (0.4025), so its calibration is terrible and it is **not** the SDK default
(`ReceiptPipeline.attach_evidence = True`). Two measurements now separate the paths.

**Evidence off (fast eval path, not shipped default):**

| Engine | Level | Accuracy | Mean confidence | ECE (10-bin) | Brier | AURC | Top-50% precision |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-VL-3B | field | 0.9475 | 0.4025 | **0.545** | 0.347 | 0.053 | 0.925 |
| Qwen3-VL-2B | field | 0.9200 | 0.4025 | **0.518** | 0.341 | 0.082 | 0.880 |
| Qwen2.5-VL-3B | doc exact-4 | 0.830 | 0.4025 | **0.427** | 0.324 | 0.162 | 0.780 |
| Qwen3-VL-2B | doc exact-4 | 0.770 | 0.4025 | **0.368** | 0.312 | 0.241 | 0.660 |

**Evidence on (shipped default):** replaying only the OCR/evidence stage on the saved
predictions (RapidOCR, no VLM calls; ~0.9 s/doc) restores per-field confidence.

| Engine | Level | Accuracy | Mean confidence | ECE (10-bin) | Brier | AURC | Top-50% precision |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-VL-3B | field, raw | 0.9475 | 0.8237 | **0.135** | 0.077 | 0.019 | 0.990 |
| Qwen2.5-VL-3B | field, isotonic OOF | 0.9475 | 0.9496 | **0.014** | 0.049 | 0.023 | 0.965 |
| Qwen3-VL-2B | field, raw | 0.9200 | 0.8215 | **0.115** | 0.094 | 0.038 | 0.980 |
| Qwen3-VL-2B | field, isotonic OOF | 0.9200 | 0.9249 | **0.005** | 0.072 | 0.047 | 0.965 |
| Qwen2.5-VL-3B | doc exact-4, raw | 0.830 | 0.7921 | **0.097** | 0.152 | 0.150 | 0.820 |
| Qwen2.5-VL-3B | doc exact-4, isotonic OOF | 0.830 | 0.8540 | **0.060** | 0.150 | 0.157 | 0.820 |
| Qwen3-VL-2B | doc exact-4, raw | 0.770 | 0.7893 | **0.092** | 0.185 | 0.229 | 0.760 |
| Qwen3-VL-2B | doc exact-4, isotonic OOF | 0.770 | 0.7558 | **0.106** | 0.194 | 0.254 | 0.740 |

Isotonic is fit with **5 folds grouped by document** (`run_confidence_calibration.py`),
so fields from the same document never cross the split. Field-level ECE drops from
0.12–0.14 to 0.005–0.014 and Brier from 0.08–0.09 to 0.05–0.07. Document-level
calibration does **not** reliably improve: the 3B gains (0.097 → 0.060), but the 2B gets
worse (0.092 → 0.106), and AURC rises for both at field level because isotonic creates
ties. The honest product statement is:

- **Field-level confidence is usable after calibration** for review routing.
- **Document-level confidence is not a probability.** Do not rank documents by it yet.
- These are out-of-fold estimates on clean-100 (public-train-derived, not external).
  Fit the production map on a dedicated calibration split and re-measure before shipping.

Artifacts: `results/confidence_probe_*_clean_evidence.json`,
`results/calibration_*_clean.json`, and the evidence-on predictions
`results/preds_pipeline_*_clean_ctx8k_evidence.jsonl`.

### 1.8 Grounding probe: VLM boxes do not clear the gate (measured 2026-11-06)

100 synthetic boxes from 30 rendered docs, identical prompt, `<box>` parsing,
coordinates normalized to [0,1000]:

| Engine | Parse rate | Mean IoU | Hit@0.5 | p50 latency |
|---|---:|---:|---:|---:|
| Qwen2.5-VL-3B | 1.00 | **0.041** | **0.04** | 655 ms |
| Qwen3-VL-2B-Instruct | 1.00 | **0.308** | **0.20** | 394 ms |

2B by field: names **0.667** (n=12), dates **0.368** (n=19), IDs **0.333** (n=12),
amounts **0.000** (n=26), line items **0.000** (n=23). The 3B was worse on 92/100
rows and often boxed the field *label* instead of the value; that may be
prompt/format-sensitive, but no prompt variant has cleared the gate.

**Decision:** v3.0 ships **OCR-span evidence only**. Do not claim VLM-predicted
boxes or bbox grounding. The product gate is mAP@0.5 ≥0.70; measured hit@0.5 is
0.04–0.20. Revisit only with a native-grounding prompt and a larger box set.

Artifacts: `results/grounding_probe_qwen2.5vl_3b.json`,
`results/grounding_probe_qwen3-vl_2b-instruct.json`.

### 1.9 Invoice transfer probe: field accuracy clears the bar; a third of documents still need review (measured 2026-11-06, re-normalized 2026-11-07)

Set: **public Invoice Extraction Benchmark v1** (CC BY 4.0,
`drew-ipp/invoice-extraction-benchmark`), 181 synthetic invoices generated by
InvoiceParser Pro from seed 42, scored with the benchmark's own scorer. This is
**synthetic, vendor-made, page-1-only, and not independent**; it is not a
real-supplier invoice measurement. The prompt was fixed before the run and not
tuned on the answers. The first pass scored valid currency symbols (`N$`, `R`,
`€`) as non-ISO strings; the corrected normalizer maps symbols to ISO codes and
the raw model outputs are unchanged. The superseded pass is archived as
`*_v1_currency_symbols.*`.

| Metric | Result |
|---|---:|
| Key header field accuracy | **91.9%** (1448/1576; 95% CI 89.8–93.9) — clears the 0.80 field bar |
| Documents with every key header field right | **65.7%** (119/181; 95% CI 58.6–72.4) |
| Line-item F1, mean | 0.929 (179 docs; 158 with all lines right) |
| p50 / mean latency | 26.5 s / 28.7 s per doc |
| Inconsistent invoices flagged | 0 of 5 |
| Multi-invoice files flagged | 0 of 2 |
| Consistent documents falsely flagged | 2 of 172 |
| References extractable by strict string match | 1 of 25 docs (4%) |

By category (key-header accuracy / fully correct):

| Category | n | Key-header | Fully correct |
|---|---:|---:|---:|
| scan | 15 | 100.0% | 100.0% |
| credit | 10 | 97.5% | 80.0% |
| clean | 31 | 95.8% | 74.2% |
| handwritten | 8 | 95.7% | 75.0% |
| photo | 45 | 95.3% | 86.7% |
| tax | 14 | 91.1% | 50.0% |
| currency | 19 | 89.3% | 52.6% |
| freight | 16 | 88.8% | 43.8% |
| hard | 13 | 87.0% | 30.8% |
| multipage | 10 | 61.1% | 0.0% |

Per field: document_type 98.9%, due_date 96.4%, invoice_date 95.0%,
invoice_number 93.3%, currency 95.0%, vendor_name 90.1%, total 87.7%, tax
87.2%, subtotal 84.4%.

**Decision:**

- The 0.80 **field-accuracy** gate is met on this synthetic page-1 probe. That
  proves prompt/schema portability, not real-invoice accuracy. For context, the
  benchmark vendor reports 96.1% fully correct for its own production system;
  our 65.7% is still a document-level gap even granting vendor advantage.
- **Do not claim straight-through invoice processing.** A third of documents
  still have at least one key-header miss, and flagging is unusable (0/7 recall,
  2 false alarms). Invoices need human review or a consistency checker before
  they can be a default vertical.
- **Do not claim multipage.** Page 1 only explains the 61.1% / 0% result; a
  full-page renderer is required before that number means anything.
- Latency is 26.5 s p50 on Ollama Q4, far above the 4 s/page target. This is a
  batch tier until MLX/quantization or a narrower prompt lands.
- The residual is now specific: subtotal/tax/total exactness (84–88%), document
  consistency/flags, references, dual-currency subtotals, and multipage.
  Invoice number/date/currency are 93–95%, so a v3.1 LoRA should target the
  arithmetic/consistency residual, not re-teach layout.

Residual taxonomy from the 3B run (field misses, not documents):

| Failure mode | Misses | Fix |
|---|---:|---|
| Multipage totals on page 1 | 30 (10 subtotal + 10 tax + 10 total) | Render all pages; sum or read the final page |
| Tax-inclusive layouts (gross used as subtotal) | 6 subtotal + 3 total | Prompt for net vs gross explicitly; check `subtotal + tax = total` |
| Dual-currency invoices (wrong column) | 4 subtotal + 3 currency | Prompt for the settlement currency and the matching column |
| Shipping/discount changes the tax base | 5 tax | Ask for adjustments separately; do not infer tax from merchandise |
| Multi-invoice files (one invoice's tax) | 4 tax | Detect multiple invoice numbers; per-invoice schema |
| Freight totals | 4 total + 3 subtotal | Same arithmetic check; freight rows mix charges |
| Photo blur/rotation digit errors | 3 subtotal + 3 tax + 2 total | Higher-resolution tiles or OCR-span cross-check |
| Two-column EU decimal shift | 1 subtotal + 1 total | Locale-aware decimal parsing in post-processing |

A naive `sum(lines) == subtotal` checker fails badly on this corpus (47 false
flags on 172 consistent documents) because the benchmark's `subtotal` excludes
shipping/discount adjustments. A product consistency checker needs explicit
adjustment lines before it can be trusted.

Artifacts: `evaluation/invoice/results/summary_invoice_qwen2.5vl_3b.json`,
`scores_invoice_qwen2.5vl_3b.json`, `preds_invoice_qwen2.5vl_3b.json`,
`rows_invoice_qwen2.5vl_3b.jsonl`; superseded pass under
`*_v1_currency_symbols.*`.

---

## 2. Root causes — why the current track failed

Ranked by evidence strength:

1. **Base perception is the bottleneck.** SmolVLM2-2.2B’s address hit rate is 0.13; Qwen2.5-VL-3B’s is 0.86 on the same docs. That 0.73 gap is not a task-format gap. The base model cannot resolve the glyphs at the effective resolution/training it has.
2. **419–46k task-format pairs cannot teach vision.** The vision-LoRA runs prove it: high LR destroys the tower (0.0395), low LR moves nothing (0.4277). Language LoRA changes style, not reading. The 2.2B model already knows the JSON format; it does not know the characters.
3. **NLL SFT on a narrow schema memorizes format.** Loss → 0.06, eval entropy 0.065 vs base 0.920 (grounding calibration), and test F1 flat. This is the documented “SFT memorizes” failure mode, not a data-volume problem alone.
4. **The eval loop selected noise.** 40-doc val is systematically easier and too small; the “best val” SFT configs did not transfer to the 100-doc test. The test set was spent once (good), but the val decisions were not trustworthy.
5. **No grounding or abstention supervision.** The product claims evidence/confidence, but the model has no bbox loss, no “absent field” training, and confidence is a heuristic. Field-level isotonic calibration is measured (§1.7) but not wired into the SDK; document-level confidence is not calibrated.
6. **The public surface outran the model.** PyPI/Space/README sold the 256M checkpoint long after it was retired. Even a good v3 model would inherit this credibility debt unless it is fixed first.

---

## 3. What “superb usable” must mean for v3

A v3 release is only “done” when all of these are true on a frozen, provenance-audited benchmark:

### Product gates

| Gate | Target | Why |
|---|---|---|
| Field F1, receipts (clean-100) | **≥ 0.92** | Match/beat the free 3B (0.9475); 2B is at 0.9200 |
| Address hit rate | **≥ 0.80** | The current first-party failure mode |
| Field accuracy, invoices (synthetic benchmark) | **≥ 0.80** | Proves it is not a one-layout receipt prompt — **synthetic probe 91.9% key-header accuracy; real invoices unmeasured; fully correct only 65.7%, flags 0/7** |
| Field F1, forms (FUNSD official or private) | **≥ 0.80** | Second layout family |
| Schema-valid rate | **≥ 0.98** | Product must not emit broken JSON |
| Evidence coverage (quote + bbox) | **≥ 0.90** of non-empty fields | Trust layer |
| bbox mAP@IoU 0.5 (where GT boxes exist) | **≥ 0.70** | Real grounding, not OCR-span only — **not met: 0.04–0.20 hit@0.5; OCR spans ship** |
| Calibration (field ECE) | **≤ 0.05** | Confidence must mean something — **met out-of-fold: 0.005–0.014**; production map still needs a dedicated calibration split. Doc-level ECE is 0.09–0.10 and is **not** a probability |
| Top-band precision at 50% coverage | **≥ 0.98** | HITL routing story |
| Latency, M4 4-bit | **≤ 4 s/page** | Usable local |
| RAM, M4 4-bit | **≤ 4 GB** | 16 GB host with headroom |
| Install | `pip install tinydoc` → cited JSON in <5 min | The actual product |

### Model gates (only if v3.1 is trained)

| Gate | Target |
|---|---|
| v3.1 vs v3.0 on receipts | **+2 F1** (paired bootstrap 95% CI excludes 0) |
| v3.1 vs v3.0 on invoices | **+5 F1** |
| Address | **≥ 0.85** |
| bbox mAP@0.5 | **≥ 0.70** |
| OCRBench | **no regression > 2 pts** vs base |
| DocVQA (held-out subset) | **no regression > 2 pts** |
| General VQA probe | **no regression > 2 pts** |
| Latency | **≤ 1.5×** v3.0 |

If v3.1 fails, **do not ship it**; ship v3.0 and publish the negative result. That is the precedent the repo already set with the 256M and 2.2B tracks.

---

## 4. The v3 ladder

### v3.0 — “TinyDoc Engine 3.0” (2–3 weeks, no training)

**Definition:** a perception-strong 2B–3B VLM as the default engine, wrapped in the existing grounded extraction layer:

- **Quality default: Qwen2.5-VL-3B** — measured **0.9475 F1 / 0.86 address / 1.00 schema / 14.2 s** on clean-100 with the fixed 8k context. Apache 2.0, already local.
- **Fast tier: Qwen3-VL-2B-Instruct** — measured **0.9200 F1 / 0.84 address / 1.00 schema / 7.0 s**; 1.9 GB Q4; native dynamic resolution, 2D grounding, 262k context. Use the **`-instruct`** tag: the plain `qwen3-vl:2b` tag is a thinking variant that spent all 512 tokens reasoning and returned empty content under the current token cap.
- **Fallback if both miss a new vertical: Qwen3-VL-4B** (vendor-reported DocVQA 95.3) — check peak memory on the 16 GB host before enabling it.
- Runtime: Ollama Q4 for the pilot; MLX 4-bit for the Mac product; GGUF/llama.cpp for CPU later.
- Layer: schema-driven prompts (receipt/invoice/form), JSON Schema validation, quote + bbox evidence, confidence/abstention (field-level ECE 0.115–0.135 raw, 0.005–0.014 isotonic OOF; doc-level not calibrated). PDF/multi-page is **not shipped**: the invoice probe is page-1 only.
- **No model training.** The base is the model; the product is the trust layer.

**Why this is the right first move:** the fair bake-off already clears the receipt gate with zero errors. We do not need to train to get a usable model; we need to package the right one, add the second vertical, and measure it honestly. The 2B is the better edge bet and the v3.1 training target; the 3B is the measured quality ceiling today. The invoice probe now shows the residual is **document consistency, flags, amount exactness, and multipage**, not basic layout transfer.

### v3.1 — “TinyDoc-2B” grounded LoRA (4–6 weeks, conditional)

**Trigger (now measured):** v3.0 passes the receipt gate but leaves a document-level residual that prompting/OCR fusion has not closed:

| Residual | Measured |
|---|---|
| Invoice key-header accuracy | 91.9% (gate met) |
| Invoice fully correct | **65.7%** — a third need review |
| Invoice flags | **0/7** inconsistent/multi-invoice; 2/172 false alarms |
| Invoice multipage | 61.1% key-header, **0% fully correct** (page 1 only) |
| Invoice amount exactness | currency 83.8%, subtotal 84.4%, tax 87.2%, total 87.7% |
| VLM boxes | hit@0.5 0.04–0.20 vs 0.70 gate |

So the v3.1 target is **document consistency + amount exactness + boxes**, not re-teaching receipt layout. Do not train on the benchmark answers; build a separate 10k–20k grounded set and a 200-doc multi-vertical val. If only synthetic invoices are available (CORD images absent, DocILE gated), run a **small** synthetic-only adapter as a controlled experiment and kill it if it does not transfer to a held-out real/synthetic mix.

**Definition:** Qwen3-VL-2B + LoRA on ~10k–20k grounded extraction examples, released as an adapter. This is the “our model” artifact: same base, but specialized to output schema’d fields with boxes and abstention.

**Why 10k–20k, not 100k:** the base already reads; the SFT is format/grounding/domain adaptation, not perception learning. DocVAL’s validated-trace result (25% of 76k → 86.2 vs full → 91.4) shows most of the gain is available from a few tens of thousands of high-quality examples. The 2.2B runs failed because the base was weak, not because 46k was too few.

### v4 — “TinyDoc-0.9B” (quarter, research bet)

Two candidate routes:

1. **Parser + extractor:** PaddleOCR-VL-1.5/1.6 (0.9B, Apache 2.0, OmniDocBench v1.5 **94.5%**, text edit 0.035) parses page → structured text + boxes; a 135M–0.5B text model maps to schema fields. Total ~1.5 GB; CPU-friendly. This is the strongest “tiny” route because the parser is already SOTA at reading.
2. **Distill the v3 teacher:** generate validated traces from Qwen3-VL-2B/4B or a 32B teacher, distill into a 0.9B student. Higher risk; only after v3.1 has a data engine.

**Kill criterion:** if the 0.9B route cannot reach **0.85 receipts / 0.75 invoices** at ≤1.5 GB int4, kill it and keep the 2B tier. Do not spend a quarter on a 256M from-scratch revival.

---

## 5. Phase plan

### Phase 0 — Truth reset and public surface (days 1–3, no GPU)

**Goal:** make the repo’s public claims match its artifacts, and make `pip install tinydoc` serve the working pipeline.

| # | Task | Deliverable / gate |
|---|---|---|
| 0.1 | Freeze v2; mark `SmolVLM2-2.2B` SFT and 256M tracks closed in README/docs | Status table says “closed — evidence” |
| 0.2 | Publish **tinydoc 0.2.0** with `ReceiptPipeline`, schemas, evidence, CLI, optional `[vlm]` extra | `pip install tinydoc` + `tinydoc extract` works from a clean venv |
| 0.3 | Point HF Space at the working engine **or** replace it with a cached/static demo; never serve the retired 256M | Space no longer runs `TinyDoc-VLM-256M` |
| 0.4 | Rewrite README hero: “local grounded extraction,” engine = Qwen3-VL-2B/Qwen2.5-VL-3B, retired 256M in a research section | No “256M / <1GB” hero |
| 0.5 | Add the 60% leakage caveat inline on the website; lead with **0.9475** (Qwen2.5-VL-3B) / **0.9200** (Qwen3-VL-2B) and 0.603 (first-party) on clean-100; label clean-100 provenance | One number, one artifact, one caveat |
| 0.6 | Recompute paired hybrid counts from shipped preds; update decision JSON/README/BENCHMARKS; extend `recompute_scores.py` to verify paired counts and p-values | `recompute_scores.py` exits 0 and checks 31/7/62 + p |
| 0.7 | Deprecate `evaluation/evaluate.py` placeholders; add a banner and point to the real runners | No hardcoded 0.0 in a live path |
| 0.8 | Update `AGENTS.md` test count (82), add a release checklist | CI + release docs current |

**Exit:** a clean venv can `pip install tinydoc`, run `tinydoc extract` on an image, and get cited JSON; the public site no longer claims 256M or an uncaveated 0.870.

**Status (2026-11-06):** 0.1 and 0.3–0.8 done; 0.2 built and clean-venv tested, **PyPI upload pending a token**; 0.3 static card deployed and Space is RUNNING.

### Phase 1 — Base bake-off (completed 2026-11-06)

**Goal:** pick the smallest base that clears the product gates.

**Measured result** (same `EXTRACT_PROMPT`, same clean-100, Ollama Q4, `num_ctx=8192`, temperature 0):

| Engine | Clean-100 F1 | Address | ANLS | Schema | Errors | Median latency |
|---|---:|---:|---:|---:|---:|---:|
| **Qwen2.5-VL-3B** | **0.9475** | **0.86** | **0.9715** | 1.00 | 0 | 14.1 s |
| **Qwen3-VL-2B-Instruct** | **0.9200** | 0.84 | 0.9340 | 1.00 | 0 | **7.0 s** |
| Hybrid v2 (first-party, shipped) | 0.603 | 0.37 | — | 0.93 | — | ~8 s |

- Paired doc-level F1: 2B wins 4, loses 13, ties 83; two-sided sign test **p = 0.049**; bootstrap F1 difference **−0.0275 (95% CI −0.0525…−0.0050)**. The 3B is better by a small but statistically detectable margin.
- Address difference is **not** significant: 0.84 vs 0.86 (95% CI −0.09…+0.05).
- Val-40 cross-check: both score **0.9250**; the clean-100 ordering is not a val/test artifact.
- The 2B is **2.0× faster** and **40% smaller** (1.9 GB vs 3.2 GB Q4), making it the right fast tier and the v3.1 training target.

**Decision:** ship **both tiers** in v3.0 — 3B as the quality default, 2B as the fast/edge tier. Neither needs training to clear the receipt gate. The 2B does not beat the 3B, so a v3.1 LoRA on the 2B is justified only if it closes the 2.75-point gap or unlocks boxes/abstention on a new vertical.

**Measured follow-ups (2026-11-06):**

- **Grounding fails the gate:** hit@0.5 is **0.20** for Qwen3-VL-2B and **0.04** for Qwen2.5-VL-3B, versus the 0.70 product gate. v3.0 ships **OCR-span evidence**, not VLM boxes. See §1.8.
- **Confidence is field-level usable, not document-level calibrated:** evidence-off eval runs are constant 0.4025 (field ECE 0.518–0.545). The evidence-on default is field ECE **0.115–0.135**, and doc-grouped isotonic OOF reaches **0.005–0.014**; document-level ECE stays **0.092–0.097** and isotonic does not reliably improve the 2B. See §1.7.
- **Invoice probe:** public synthetic benchmark (181 docs, page 1) measures **91.9% key-header field accuracy / 65.7% fully correct**; flagging is 0/7 and multipage is 0% fully correct. Real supplier invoices remain unmeasured. See §1.9.
- **Multi-vertical:** FUNSD images/labels are present (199 docs; 50 scored in the probe); CORD images are absent and DocILE is gated, so real invoices/forms beyond FUNSD are still unmeasured. Do not claim production invoice performance.

Reproduce the grounding probe:

```bash
python data/synthetic/gen_grounding_sample.py --num-docs 30 \
    --output-dir /Volumes/KIOXIA\ 1TB/tinydoc/grounding_probe_20261106
python evaluation/phase0/run_grounding_probe.py \
    --manifest /Volumes/KIOXIA\ 1TB/tinydoc/grounding_probe_20261106/manifest.jsonl \
    --model qwen3-vl:2b-instruct --limit 100 \
    --out evaluation/phase0/results/grounding_probe_qwen3-vl_2b-instruct.json
python evaluation/phase0/run_grounding_probe.py \
    --manifest /Volumes/KIOXIA\ 1TB/tinydoc/grounding_probe_20261106/manifest.jsonl \
    --model qwen2.5vl:3b --limit 100 \
    --out evaluation/phase0/results/grounding_probe_qwen2.5vl_3b.json
```

**Multi-vertical status:** FUNSD is a 50-doc nonstandard transfer probe; the public synthetic invoice benchmark measures 91.9% key-header accuracy / 65.7% fully correct (page 1, flags 0/7). CORD images are absent and DocILE is gated, so **real** invoices/forms remain unmeasured. Do not claim production invoice performance.

### Phase 2 — v3.0 product hardening (weeks 1–3, no training)

**Goal:** make the chosen base genuinely usable.

| # | Task | Details |
|---|---|---|
| 2.1 | Schema-driven prompts | Generate the prompt from JSON Schema; receipt/invoice/form packs; `additionalProperties: false`; typed money/date; absent-field behavior |
| 2.2 | PDF/multi-page | `pdf2image`/pypdfium; page-level extraction; document assembly; page citations |
| 2.3 | Evidence + boxes | OCR-span evidence ships (measured 0.80 coverage); VLM bbox only if a future probe clears the 0.70 gate. Normalize to page coordinates; overlay. |
| 2.4 | Calibrated confidence | Evidence-on field confidence is measured (ECE 0.12–0.14; isotonic OOF 0.005–0.014). Wire a field-level map fit on a dedicated split; doc-level confidence stays heuristic. Then ECE/AURC report + `needs_review` bands. |
| 2.5 | HITL routing | Per-field confidence bands; `needs_review` flag; top-band precision/coverage curve |
| 2.6 | Engine router | `auto` picks the quality default (Qwen2.5-VL-3B) or fast tier (Qwen3-VL-2B) from config, falls back to OCR floor with a loud reason; no silent degradation |
| 2.7 | Batch CLI | Folder recursion, JSONL, `--schema`, `--overlay`, `--fail-fast`, `--limit`; progress + errors |
| 2.8 | Demo | Local Gradio on the working engine; HF Space either GPU-backed or cached/static |
| 2.9 | Release | 0.2.0 to PyPI; model/pipeline card; artifact links; repro command |

**Exit:** the product gates in §3 pass on clean-100 + one new vertical; an install from PyPI works offline on the M4; the HF Space no longer shows the dead model.

### Phase 3 — v3.1 grounded LoRA (weeks 3–6, conditional on Phase 1/2 residual)

**Data (10k–20k):**

| Source | Rows | Labels | Purpose |
|---|---:|---|---|
| Synthetic renderer (existing) | 5k–10k | field + bbox, perfect | Format, boxes, layout families, hard negatives |
| SROIE clean train (420) + CORD (800) + FUNSD (149) | 1.4k | field; boxes for FUNSD/CORD | Real receipts/forms |
| Teacher-labeled hard docs (photos/skew/low light) | 3k–5k | field + bbox, validated | Robustness; use Qwen3-VL-32B/4B or 2B + validator; keep only cross-checked rows |
| Abstention/negatives | 1k–2k | empty/`null` fields | Stop hallucinating absent fields |
| General mix (DocVQA/ChartQA/text) | 1k–2k | QA | Protect cognition (Bunny/Qwen practice) |

**Quality filter (DocVAL recipe):** reject rows where (a) the field is not present in the OCR text and not visually verifiable, (b) two independent prompts/models disagree, (c) the target violates type rules, (d) the image is unreadable/blurred. Measure the keep rate; report it.

**Training:**

- Base: Qwen3-VL-2B-Instruct.
- Freeze vision tower; LoRA r=16, α=32, dropout 0.05 on LLM q/k/v/o/gate/up/down; optionally train the projector.
- 1 epoch, lr 1e-4 cosine, warmup 3%, batch 1–2 × grad-accum 8, max seq 2048–3072, bf16 on A100 / fp16 on T4, grad checkpoint.
- Assistant-only loss (verify with a debug batch).
- Native dynamic resolution; cap longest edge at 1280 (receipts) / 1600 (invoices). **Do not silently train at 384px** — the MLX/Unsloth path has a known 384px default; verify the processor’s pixel budget.
- Checkpoint selection on a **200-doc multi-vertical val**, not the 40-doc SROIE val. Never touch the test until the final adapter.

**Compute:**

- 10k–20k rows, 1 epoch, 2B LoRA: **~6–12 h on 1×A100/L40S** (est. $30–80) or **2–4 Kaggle T4 sessions** (free, slower; ~5 s/sample at batch 8).
- Prefer one paid burst over another multi-week Kaggle grind; the repo has already paid that tax twice.

**Gates:** §3 model gates. If fail, publish negative and ship v3.0.

### Phase 4 — v4 tiny (optional, next quarter)

- Request **DocILE** access (6,680 real business documents, mostly invoices; 55 field types; boxes) — the right benchmark for “grounded extraction,” not just receipts.
- Evaluate **PaddleOCR-VL-1.5/1.6** as the parse tier; if its text+boxes + a small extractor beats the 2B on invoices at ≤1.5 GB, ship `TinyDoc-Parse-0.9B`.
- Only then consider distillation. Kill at the stated gate.

---

## 6. Evaluation protocol (the scoreboard)

### 6.1 Datasets and provenance

| Set | Size | Source | Use | Leakage status |
|---|---:|---|---|---|
| SROIE clean-100 | 100 | public SROIE train, filtered vs our holdout | Receipt regression | **Not external**; may be in Qwen pretraining |
| SROIE official test | 347 | SROIE test images (labels via `urbikn/sroie-datasetv2` if verified) | External receipt | Check labels/licensing; likely public |
| CORD test | 100 | `naver-clova-ix/cord-v2` | Receipt multi-field | Public test; may be in pretraining |
| Invoice benchmark v1 | 181 | `drew-ipp/invoice-extraction-benchmark` (synthetic, seed 42) | Invoice transfer probe, page 1 | Public, synthetic, vendor-made; not real invoices |
| FUNSD test | 50 | `nielsr/funsd` | Forms | Public test; may be in pretraining |
| DocILE test | 6.7k (gated) | Rossum DocILE | Business documents (mostly invoices), 55 field types, boxes | Best external invoice-style benchmark; request access |
| Private wild set | 150 | user/consented docs | True product truth | **Never train**; hash; store outside repo |
| Synthetic hard set | 200 | existing renderer + augmentations | Grounding/robustness | Guaranteed clean |

**Rule:** every reported number names its set and provenance. “Clean-100” is renamed to “SROIE-train-derived holdout-100” in docs, or the JSON gets a `provenance` field.

### 6.2 Metrics

- **Field exact match / P/R/F1** (primary), per field and macro.
- **ANLS** (DocVQA-style), **CER** for transcription.
- **Schema-valid rate**, **exact-4/4**, **abstention rate**.
- **Evidence coverage** (non-empty field with quote), **gold-in-quote**, **bbox mAP@IoU 0.5**.
- **Confidence:** ECE, Brier, AURC, risk-coverage curve, top-band precision at 50%/70% coverage.
- **Latency:** warm/cold, p50/p95, M4 4-bit and CPU; RAM peak.
- **Paired statistics:** bootstrap over documents (10k resamples), 95% CI; sign test for paired wins. Report n and errors.

### 6.3 Artifacts and gates

- All predictions/scores under `evaluation/v3/results/`; `recompute_scores.py` recomputes every published number and paired count.
- `scripts/verify-product.mjs` gains `v3-claims`, `v3-install`, `v3-eval` gates.
- CI runs the recompute + unit tests; no number outside the artifact set.

---

## 7. What to stop doing

| Stop | Why (evidence) |
|---|---|
| 256M from-scratch VLM | 0.0% OCRBench; dead 768 tower; SmolDocling-256M 0.493 OmniDocBench |
| SmolVLM2-2.2B SFT / vision LoRA | every clean-100 language-LoRA config 0.535–0.540; 22 val-40 configs did not transfer; vision LoRA 0.0395 / 0.4277 |
| 40-doc val loops for model selection | val 0.600 vs test 0.536; n=40 CI ±15 pts; selected noise |
| Treating train loss / eval accuracy as success | loss 0.06, entropy 0.065, test flat — memorization |
| Shipping a model before the package/Space serves it | PyPI 0.1.2 + Space 256M while repo claims v2 |
| Unmeasured marketing numbers | repo policy already says this; enforce it on the website too |
| 7B on the 16GB M4 | crashed the host; 3B stays |
| Adding a third LoRA variant before the base bake-off | the base is the bottleneck, not the LoRA config |

---

## 8. Risks and kill criteria

| Risk | Likelihood | Mitigation / kill |
|---|---|---|
| Qwen3-VL-2B underperforms 3B on SROIE | **Confirmed** | Measured 0.9200 vs 0.9475 (paired 95% CI −5.25…−0.50); ship the 3B as quality default and the 2B as the 2× faster tier. Re-test on invoices/forms before choosing a single default. |
| Public SROIE/CORD numbers are contaminated for Qwen | High | Private 150-doc wild set + synthetic hard set; label public numbers as “public-split” |
| DocILE access delayed | Medium | Start with CORD + FUNSD + synthetic invoices; request DocILE in parallel |
| v3.1 LoRA overfits / collapses | Medium | Freeze vision, low LR, 10k–20k quality-filtered, 200-doc multi-vertical val, early stop on F1; kill if gates fail |
| Paid GPU cost overruns | Low | Cap at 1–2 days; 10k rows; checkpoint/resume; Kaggle fallback |
| HF Space can’t run 2B on free CPU | High | Use a GPU Space, a cached/static demo, or a “run locally” page; never serve the retired 256M |
| Another “story > product” cycle | High | Phase 0 ships the package/Space before any training; public claims gated by recompute |

**Kill criteria for the whole v3.1 track:** if after 10k–20k grounded examples the adapter does not beat v3.0 by +2 F1 on receipts or +5 on invoices with a paired CI excluding 0, stop training and ship v3.0. The evidence says the base is the lever; if a strong base + good data does not move, the problem is the data/eval, not the model size.

---

## 9. Immediate next actions (in order)

1. **Bake-off complete (2026-11-06).** 3B 0.9475 / 2B 0.9200 on clean-100, both zero errors. Commit the six new preds/scores artifacts and extend `recompute_scores.py` to audit them before quoting externally.
2. **Fix the public surface in parallel** (no GPU): publish `tinydoc 0.2.0`, replace the Space, rewrite the README hero, add the leakage/provenance caveats, recompute paired counts.
3. **Build the 200-doc multi-vertical val + private 150-doc test** (CORD/FUNSD + synthetic invoices + wild docs). Freeze them; hash them; never train.
4. **Run the grounding/confidence probes** on the 3B and 2B. If boxes work, v3.0 ships with real bbox evidence; if not, keep OCR-span evidence honest.
5. **Only then** decide v3.1: if the residual is invoice/forms/grounding — not receipts — train the 10k–20k LoRA on one paid GPU burst. If the residual is small, skip training and ship v3.0.

**The one-sentence version:** *Stop trying to make a weak 2.2B read; put a perception-strong 2B/3B behind the trust layer you already built, prove it on a clean multi-vertical benchmark, and train only the residual that the proof reveals.*

---

## 10. Sources

### Internal artifacts

- `evaluation/phase0/results/ocrbench_256m_full.json` — 256M OCRBench 0.0% (n=1000)
- `evaluation/phase0/results/scores_smolvlm2-2.2b_base_clean.json` — 0.5357, address 0.13
- `evaluation/phase0/results/scores_smolvlm2-2.2b_lora_clean*.json` — 0.5352–0.5395
- `evaluation/phase0/results/scores_smolvlm2-2.2b_kv_kreal100val.json` — 0.0395
- `evaluation/phase0/results/scores_smolvlm2-2.2b_kv_kll100val.json` — 0.4277
- `evaluation/phase0/results/scores_product_hybrid_clean.json` — 0.6028, address 0.37
- `evaluation/phase0/results/scores_pipeline_ollama_qwen2.5vl_3b_clean.json` — 0.9282 (superseded: 5 context-400 errors)
- `evaluation/phase0/results/scores_pipeline_ollama_qwen2.5vl_3b_clean_ctx8k.json` — 0.9475, address 0.86, 14.2 s (bake-off control)
- `evaluation/phase0/results/scores_pipeline_ollama_qwen3-vl_2b-instruct_clean_ctx8k.json` — 0.9200, address 0.84, 7.0 s (bake-off candidate)
- `evaluation/phase0/results/confidence_probe_qwen2.5vl_3b_clean.json` — evidence-off ECE 0.545, uncalibrated heuristic
- `evaluation/phase0/results/confidence_probe_qwen3-vl_2b-instruct_clean.json` — evidence-off ECE 0.518, uncalibrated heuristic
- `evaluation/phase0/results/confidence_probe_qwen2.5vl_3b_clean_evidence.json` — evidence-on ECE 0.135 (field), 0.097 (doc)
- `evaluation/phase0/results/confidence_probe_qwen3-vl_2b-instruct_clean_evidence.json` — evidence-on ECE 0.115 (field), 0.092 (doc)
- `evaluation/phase0/results/calibration_qwen2.5vl_3b_clean.json` — isotonic OOF ECE 0.014 (field), 0.060 (doc)
- `evaluation/phase0/results/calibration_qwen3-vl_2b-instruct_clean.json` — isotonic OOF ECE 0.005 (field), 0.106 (doc; not improved)
- `evaluation/phase0/results/scores_pipeline_ollama_qwen2.5vl_3b_val_ctx8k.json` — 0.9250 (val-40)
- `evaluation/invoice/results/summary_invoice_qwen2.5vl_3b.json` — invoice probe: 90.6% key-header, 55.8% fully correct, p50 26.5 s
- `evaluation/invoice/results/scores_invoice_qwen2.5vl_3b.json` — scorer output, 181 docs
- `evaluation/invoice/results/preds_invoice_qwen2.5vl_3b.json` — predictions in benchmark format
- `evaluation/invoice/results/rows_invoice_qwen2.5vl_3b.jsonl` — raw outputs, errors, latency
- `evaluation/phase0/results/scores_pipeline_ollama_qwen3-vl_2b-instruct_val_ctx8k.json` — 0.9250 (val-40)
- `evaluation/phase0/results/scores_ollama.json` — 0.870 (60% leaked set)
- `evaluation/phase0/results/eval_leakage_audit.json` — 61/100 overlap
- `evaluation/phase0/results/stage1_manifest_summary.json` — clean-100 provenance, 420 train/40 val/100 eval
- `evaluation/phase0/results/v2_product_decision.json` — 0.6028; paired counts (inconsistent; fix)
- `README.md`, `docs/BENCHMARKS.md`, `docs/pivot_plan.md`, `docs/index.html`
- `sdk/setup.py` (0.1.3), PyPI `tinydoc` 0.1.2
- `demo/hf_space/app.py` (retired 256M)

### External (primary / vendor / leaderboard — label when citing)

- Qwen3-VL-2B-Instruct model card (Apache 2.0, 262k context): https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct
- Qwen3-VL technical report (DocVQA 93.3 for 2B; vendor-reported): https://arxiv.org/pdf/2511.21631v2
- Qwen2.5-VL-3B model card (DocVQA 93.9; vendor-reported): https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
- PaddleOCR-VL-1.5 (0.9B, 94.5 OmniDocBench v1.5): https://arxiv.org/html/2601.21957v2
- PaddleOCR-VL model card: https://huggingface.co/PaddlePaddle/PaddleOCR-VL
- dots.ocr (1.7B; OmniDocBench 87.5 EN; olmOCR 79.1): https://huggingface.co/dots-studio/dots.ocr · https://arxiv.org/abs/2512.02498
- GLM-OCR (1B; OmniDocBench 94.62): GLM-OCR technical report (cited in `docs/model_improvements.md`)
- SmolDocling-256M (OmniDocBench 0.493 EN overall edit): dots.ocr comparison table
- DocILE benchmark (6,680 real business documents, mostly invoices; 55 field types; boxes): https://docile.rossum.ai/ · https://arxiv.org/abs/2302.05658
- ExtractConf (multi-signal confidence; DocILE 0.928 AUC, 70% risk reduction): https://arxiv.org/html/2606.24420
- Ollama qwen3-vl:2b (1.9 GB Q4): https://ollama.com/library/qwen3-vl:2b
- MLX-VLM Qwen3-VL support: https://github.com/Blaizzy/mlx-vlm/pull/528

---

*If only one sentence survives review: **the base model is the bottleneck; swap it, prove it on a clean multi-vertical benchmark, ship the trust layer, and train only the residual.***
