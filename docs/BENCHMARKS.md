# TinyDoc-VLM Benchmark Results

> **Rewritten 2026-09-24 — measured numbers only.**  
> Earlier versions of this file (and `docs/paper.md`, `docs/benchmark_results.json`,  
> the pitch/reddit/twitter/launch drafts) carried **unmeasured estimates**  
> (DocVQA ~55%, FUNSD ~70%, …) presented as results. They are withdrawn.  
> Every number below recomputes from a committed artifact. The measured DocVQA
> reading probe (0.829 single-reference ANLS) is a separate, caveated result — it
> is not the withdrawn estimate and is not a leaderboard number.

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

## Invoice transfer probe (public synthetic benchmark)

**Set:** [Invoice Extraction Benchmark v1](https://github.com/DrewKraken/invoice-extraction-benchmark)
(CC BY 4.0), 181 synthetic invoices with answer keys, scored by the benchmark's own
scorer. **Synthetic, vendor-made, page 1 only, not independent, not real supplier
invoices.** The prompt was fixed before the run and not tuned on the answers.

| Metric | Result |
|---|---:|
| Key header field accuracy | **91.9%** (1448/1576; 95% CI 89.8–93.9) |
| Documents with every key header field right | **65.7%** (119/181; 95% CI 58.6–72.4) |
| Line-item F1, mean | 0.929 (179 docs) |
| p50 / mean latency | 26.5 s / 28.7 s |
| Inconsistent invoices flagged | 0 of 5 |
| Multi-invoice files flagged | 0 of 2 |
| Consistent documents falsely flagged | 2 of 172 |

Paired tier comparison on the same 181 documents (B − A, 95% CI):

| Metric | Qwen2.5-VL-3B | Qwen3-VL-2B-Instruct | Difference |
|---|---:|---:|---:|
| Key header accuracy | 91.9% | **93.5%** | +1.6 pts (+0.4 to +2.9) |
| Documents fully correct | 65.7% | **71.8%** | +6.1 pts (0.0 to +11.6) |
| Line-item F1, mean | 0.929 | 0.904 | −0.025 |
| p50 latency | 26.5 s | **16.1 s** | −10.4 s |

The 2B is at least as good on invoices and 40% faster; the point estimates
favor it but the per-document sign test is not significant (p=0.092), so read
this as “no evidence that 3B is better on invoices”. It does **not** transfer to
receipts: the 3B still leads clean-100 (0.9475 vs 0.9200).

| Category | n | Key-header (3B) | Fully correct (3B) | Fully correct (2B) |
|---|---:|---:|---:|
| scan | 15 | 100.0% | 100.0% | 100.0% |
| credit | 10 | 97.5% | 80.0% | 100.0% |
| clean | 31 | 95.8% | 74.2% | 87.1% |
| handwritten | 8 | 95.7% | 75.0% | 37.5% |
| photo | 45 | 95.3% | 86.7% | 84.4% |
| tax | 14 | 91.1% | 50.0% | 57.1% |
| currency | 19 | 89.3% | 52.6% | 68.4% |
| freight | 16 | 88.8% | 43.8% | 56.2% |
| hard | 13 | 87.0% | 30.8% | 53.8% |
| multipage | 10 | 61.1% | 0.0% | 0.0% |

Per field: document_type 98.9%, due_date 96.4%, invoice_date 95.0%,
invoice_number 93.3%, currency 95.0%, vendor_name 90.1%, total 87.7%, tax 87.2%,
subtotal 84.4%. References (strict string match) are 1/25 and are not a headline
claim.

**Read this as:** the prompt/schema transfers to invoices at the field level
(above the 0.80 bar), but **not** as straight-through processing: 28–34% of
documents still have at least one key-field miss, flagging is unusable (0/7),
and the multipage row is **page-1 only**. Currency symbols were normalized to ISO
after the first pass; the superseded run is archived as
`*_v1_currency_symbols.*`. Do not cite this as real-invoice accuracy. The
separate all-pages probe below replaces the page-1 0% for the same 10 documents.

### Multipage invoices — page 1 vs all pages (2B, n=10)

| Metric | Page 1 | All pages | Delta |
|---|---:|---:|---:|
| Key-header accuracy | 63.3% (57/90) | **71.1%** (64/90) | +7.8 pts |
| Documents fully correct | 0/10 | **4/10** | +4 docs |
| Mean line-item F1 | 0.418 | 0.322 | −0.096 |
| p50 latency per document | 22.4 s | 34.7 s | +12.3 s |
| p50 latency per page | — | 8.9 s | — |

Each document's pages were sent together in one request (2–8 images; 46 pages
total) at 1400 px; the 8-page retry used a 24,576-token context and an
8,192-token output budget. The paired sign tests are
**not significant**: key header p=0.289 (6 improved, 2 regressed, 2 unchanged),
fully correct p=0.125 (4 gained, 0 lost). Treat this as a 10-document
descriptive probe, not proof that multipage is solved. Among the **8 documents
that produced valid JSON**, key-header is 62/72 (86.1%) and 4/8 are fully
correct; the two 8-page failures are counted in the 71.1% / 4/10 headline.

**The failures travel with the number:**

- **Two 8-page documents failed.** Against 117 and 110 expected line items, the
  model emitted **213 and 212** descriptions, fabricating repeated “(batch N)”
  line items until it hit the output limit; neither produced valid JSON. This is
  runaway generation, not a token-budget artifact — a larger budget made it
  worse.
- **Line items are not fixed.** Mean F1 falls from 0.418 to 0.322, no document
  has every line item right, and the 8-page documents fail outright. On some
  invoices the model reads **unit price instead of line amount**; on others it
  misses later pages’ items.
- **Totals still fail in specific ways.** One 3-page invoice read 70,390.50 as
  390.50 (digit truncation); 4-page and 6-page invoices used a page subtotal
  instead of the document total.
- **Resolution is a confound.** Page-1 renders at 1800 px, multipage at 1400 px;
  small table text is harder to read at 1400 px, so line-item F1 is not directly
  comparable across the two columns.

**Read this as:** reading every page fixes document-wide key headers on 4 of 10
synthetic invoices and moves fully-correct from 0/10 to 4/10, but it does **not**
make multipage invoices straight-through: two 8-page files fail, line items are
unreliable, and 6/10 still miss at least one key header field. The 0.3.0 SDK
preview exposes the page-wise path, but it is a probe-quality feature. Artifacts:
`evaluation/invoice/results/`.

### Page-wise multipage + reconciliation (2B, n=10, 46 pages)

The all-pages request's two 8-page failures are an output-control problem, so the
next probe reads **one page per request** and merges in code: first-page header,
concatenated line items, final-page totals, and narrow internal-consistency
repairs. It then adds deterministic reconciliation:

1. **Last-page currency.** The final page prints the totals and normally the
   currency; a printed `N$`/`R`/`$`/`€`/`£` beats a model-guessed code.
2. **Conservative arithmetic.** A printed total and a printed tax are
   authoritative. A small subtotal residual is resolved toward `total − tax`, a
   truncated tax is derived from `total − subtotal`, and a total is derived only
   when it is missing or structurally impossible (less than 95% of a printed
   subtotal). Inconsistencies are flagged; they are not silently rewritten.
   This policy was tightened after the real Portuguese probe below, where the
   old unconditional `total = subtotal + tax` rewrite double-counted VAT on 50
   tax-inclusive receipts. The 86/90 non-OCR row in the table was measured with
   the 0.3.0 merge; the OCR 90/90 row is unchanged.
3. **OCR printed totals.** `totals_ocr.py` reads the final page with tesseract
   and accepts a subtotal/tax/total block only when all three labels are found,
   `subtotal + tax == total` within 0.05, and the implied tax rate is 0–35%.
   A consistent block replaces the model's three amounts; a printed symbol/code
   replaces the currency. Line items, dates, and text are never changed.

No step reads gold answers. The ladder:

| Method | Key-header | Fully correct | Line F1 | p50/doc |
|---|---:|---:|---:|---:|
| All pages in one request (VLM) | 71.1% (64/90) | 4/10 | 0.322 | 34.7 s |
| One page per request (VLM) | 93.3% (84/90) | 6/10 | 0.848 | 70.7 s |
| + merge reconciliation | 95.6% (86/90) | 8/10 | 0.848 | 70.7 s |
| + OCR printed totals | **100% (90/90)** | **10/10** | 0.848 | 70.7 s |

All 46 pages were sent one at a time at 1400 px with an 8,192-token context and a
2,048-token output budget; all parsed and stopped cleanly. The two 8-page files
have **117/117** and **110/110** line items. OCR plus rebuild took **2.1 s for
all 10 documents** (~0.2 s/doc), so latency remains dominated by VLM inference.
Paired changes: page-wise VLM → OCR key header improved 4 / regressed 0
(p=0.125), fully correct gained 4 / lost 0 (p=0.125); all-pages → OCR key
improved 6 / regressed 0 (p=0.03125), fully correct gained 6 / lost 0
(p=0.03125). n=10, synthetic, clean PDFs.

**The residual travels with the number:**

- **10/10 is nine key-header fields, not line items.** Line-item F1 is 0.848,
  only **1/10** documents has every line right, and 8/10 trigger a
  line-sum-vs-subtotal review flag.
- **The 10/10 is a pipeline result, not a pure VLM score.** The VLM alone is
  **6/10** and 93.3% key-header. OCR reads the three printed monetary totals;
  it uses no gold but it is an independent reader of the same source.
- **The exact 95% CI for 10/10 fully correct is 69.2–100%** (n=10,
  Clopper-Pearson). This is a 10-document probe, not a production rate.
- **The 181-document page-1 probe was not re-run.** Its 91.9%/93.5% numbers are
  VLM-only; reconciliation was measured only on the 10 multipage documents.
- **The OCR parser is not held out.** It was developed against the same 10
  final pages it is reported on, so the 10/10 is not an independent test of the
  OCR layer. A held-out multipage set is required before a production claim.
- **Three documents have disclosed corrections.** `multipage-2p-za-1`
  (subtotal/tax/total from OCR), `multipage-3p-na-3` (subtotal residual plus
  currency from OCR), `multipage-6p-za-8` (truncated tax).
- **Prompt v2 failed.** An explicit currency-symbol prompt with a
  `currency_symbol` key scored **81/90 and 5/10** raw versus **84/90 and 6/10**
  for v1, and one final-page total regressed to 390.50. Deterministic
  reconciliation fixed the result. The canonical run keeps v1; v2 is archived
  as a failed variant.
- **It is ~2× slower.** 70.7 s/doc p50 and 16.1 s/page; chunking is the next
  latency experiment.

**Read this as:** output control and field conventions, not page coverage, were
the main multipage bottlenecks. Page-wise reading fixes the runaway, nearly
triples line-item F1, and deterministic reconciliation closes the monetary
fields on this probe. It is still not straight-through: line items, latency, and
real-invoice transfer remain open. The 0.3.0 SDK preview exposes the page-wise
path; that is not a production multipage claim. Artifacts:
`evaluation/invoice/results/summary_invoice_qwen3-vl_2b_mp_pagewise_ocr.json`,
`compare_invoice_pagewise_vs_ocr_2b.json`,
`compare_invoice_mp_allpages_vs_ocr_2b.json`,
`audit_invoice_qwen3-vl_2b_mp_pagewise_ocr.json`.

### Degraded multipage probe (2B, same 10 documents, 46 pages)

The clean multipage probe uses PDF-rendered text. Real uploads are photos, scans,
and re-compressed images, so the same 46 pages were degraded deterministically
(seed 42): downscale to 1100 px then upscale to 1400, rotate up to 1.2°, Gaussian
blur, brightness/contrast jitter, Gaussian sensor noise, and JPEG quality 58. The
page-wise prompt, merge, and OCR total reconciliation were unchanged.

| Metric | Clean (10 docs) | Degraded (8 complete docs) |
|---|---:|---:|
| Documents completed | 10/10 | **8/10** |
| Key-header accuracy | 100% (90/90) | **98.6% (71/72)** |
| Fully correct | 10/10 | **7/8** (exact 95% CI 47.3–99.7%) |
| Mean line-item F1 | 0.848 | **0.795** |
| All line items right | 1/10 | **0/8** |
| OCR consistent blocks | 10/10 | **10/10** |
| p50 latency per document | 70.7 s | **101.1 s** |

**Two documents failed on a page; the audit explicitly allows those two failures and lists them:**

- `multipage-3p-gb-4` page 1: the model emitted malformed JSON (an arithmetic
  expression in the total field). A targeted retry reproduced the failure.
- `multipage-6p-za-8` page 5: Ollama returned HTTP 500 under memory pressure. A
  targeted retry hit another HTTP 500 on page 1.

The scorer therefore sees 8 documents, not 10. This is a robustness finding, not
a hidden failure: degraded pages can break both the VLM output and the local
server. OCR totals were consistent on **all 10 degraded final pages**, which is a
held-out test of the OCR layer on these documents; it corrected three amounts or
currency values. Paired against the clean run on the same 8 documents:
key-header 72/72 → 71/72, fully correct 8/8 → 7/8, line F1 0.8525 → 0.795.

**Read this as:** the printed-total OCR reader generalizes to mild photo/scan
degradation on this set, but the VLM page path is not robust to it. Line items
remain weak. The degradation is mild and deterministic, n=10, synthetic, and not
a production multipage claim. Artifacts:
`evaluation/invoice/results/summary_invoice_qwen3-vl_2b_mp_pagewise_degraded_ocr.json`,
`compare_invoice_clean_vs_degraded_2b.json`,
`audit_invoice_qwen3-vl_2b_mp_pagewise_degraded_ocr.json`,
`degrade_mp_recipe_42.json`.

### Real Portuguese receipts and invoices (five fields, n=100, development set)

A separate probe adds **real documents**, not synthetic invoices: 100 phone
photos of Portuguese receipts and simplified invoices (80 from a private
company, 20 personal), CC BY 4.0, with human transcriptions of vendor, invoice
number, date, currency, tax, and total. It is a **five-field transfer probe**,
not the nine-field invoice benchmark and not a B2B supplier-invoice set.

Setup: `qwen3-vl:2b-instruct`, canonical page prompt v1, long side 1400 px,
8192 context, 2048 output cap, one page per document. One of 100 documents hit
the output cap with runaway line items (`done_reason: length`) and produced no
JSON; it is counted wrong. p50 latency 8.7 s/doc.

| Path (same model outputs) | Field accuracy | Fully correct | All six (n=80) |
|---|---:|---:|---:|
| 0.3.0 shipped merge | 374/577 = **64.8%** | 3/100 = **3%** | 2/80 = **2.5%** |
| Preserve-printed-total merge (unreleased) | 428/577 = **74.2%** | 18/100 = **18%** | 13/80 = **16.3%** |

Exact 95% intervals for the fixed merge: field accuracy 70.4–77.7%; fully
correct 11.0–26.9%. The field interval is optimistic because fields cluster
within documents.

| Field | Shipped merge | Fixed merge | 95% CI (fixed) |
|---|---:|---:|---:|
| Currency | 96/100 (96.0%) | 96/100 (96.0%) | 90.1–98.9% |
| Invoice date | 93/97 (95.9%) | 93/97 (95.9%) | 89.8–98.9% |
| Invoice number | 44/98 (44.9%) | 44/98 (44.9%) | 34.8–55.3% |
| Tax | 67/87 (77.0%) | 68/87 (78.2%) | 68.0–86.3% |
| Total | 19/100 (19.0%) | 72/100 (72.0%) | 62.1–80.5% |
| Vendor | 55/95 (57.9%) | 55/95 (57.9%) | 47.3–68.0% |

**What broke.** The 0.3.0 merge rewrote a printed total as `subtotal + tax`. On
tax-inclusive receipts that double-counted VAT and turned 50 correct printed
totals into wrong ones. Preserving the printed total and tax recovers 53
documents with no new model call. The fixed merge also leaves 84/100 documents
with at least one arithmetic review flag rather than silently repairing them.

**Residuals after the field-level reading (fixed merge).** Total 28/100 wrong:
five dot-decimal misreads (`8.88` → `8880`), nine model-computed totals that add
tax to a tax-inclusive amount, three missing, and the rest line-amount/digit
errors. Invoice number 54/98 wrong: 17 one-or-two-digit errors, 11 empty, 26
wrong-field/other. Vendor 40/95 wrong: 12 brand-vs-legal-entity cases and 27
wrong-seller cases. Currency is 96/100 (two `BRL`, one `E`, one empty); date is
93/97.

**Caveats that travel with every number.** The 100 documents are a development
set; the merge fix was designed after seeing their failures, so 74.2% is a
**post-hoc** result. A disjoint seed-43 holdout is frozen but **not run**. The
0.3.0 wheel on PyPI does not contain the fix. No line-item, due-date, flag
quality, confidence-calibration, or real-B2B-supplier claim is supported by this
probe. Labels are human transcriptions and contain noise; implausible tax values
were excluded, not corrected. Full method and artifacts:
[`evaluation/invoice/PT_REAL_RESULTS.md`](../evaluation/invoice/PT_REAL_RESULTS.md).

### DocVQA reading probe (2B, 200 questions, single reference)

**Set:** 200 public DocVQA images from the lmms-lab mirror, one question per
image (seed 42), images capped at 768 px. Each question has **one** target
answer; official DocVQA scoring takes the best ANLS over up to 10 human answers.
This is a reading probe, not field extraction, and the set may be in pretraining.

| Metric | Qwen3-VL-2B-Instruct |
|---|---:|
| ANLS (threshold 0.5, single reference) | **0.829** |
| Normalized exact match | **0.765** |
| Errors | 0 / 200 |
| p50 / mean latency | 3.36 s / 3.43 s |

The model reads most document text and numbers: high-confidence examples include
table values, names, and dates; misses are wrong-row selection (e.g. net block
vs a number) rather than empty output. **Do not cite 0.829 as a DocVQA
leaderboard score.** The single-reference metric, 768 px mirror, one-question
sampling, and public/pretraining exposure all break comparability. The number is
useful only as an internal reading signal for this engine and this mirror.

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
- **Output budget:** invoice page-1 runs used `num_predict=768`, the all-pages
  multipage run uses `5120` (and an 8-page retry at `8192`), the page-wise run
  uses `2048` per page, and the DocVQA probe uses `128`. The invoice runner records
  `prompt_eval_count`, `eval_count`, and `done_reason`; a `length` stop is a failed
  generation, not a result. Multipage documents carry 98–117 line items, so the
  page-1 budget truncates them.
- **DocVQA:** 200 public images, one question per image (seed 42), **one reference
  answer**. Single-reference ANLS is not comparable to DocVQA leaderboard numbers.

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
