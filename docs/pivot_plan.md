# TinyDoc-VLM Pivot Plan

**Date:** 2026-09-22
**Status:** Decision document — deep audit + verified research + recommended path
**Audience:** Project owner (eulogik)

---

## 0. One-line verdict

**Do not continue the current SmolVLM2-2.2B QLoRA run as-is, and do not ship another "fine-tuned checkpoint" as the product.** The evidence says that path lands on a model that loses to free off-the-shelf weights, with metrics that currently indicate memorization, not grounding. The winning move is to **keep the assets (data pipeline, eval gates, Kaggle infra, SDK/demo shell), reframe the product around grounded local extraction with field-level eval, and restart model work on a stronger base with quality-filtered + localization-aware supervision.**

---

## 1. Deep audit — what we actually did

### 1.1 Timeline of substance (from git + artifacts)

| Phase | What happened | Outcome |
|-------|---------------|---------|
| **v0 vision** | 256M from-scratch VLM (SigLIP-B + PixelShuffle + SmolLM2-135M), 10K synthetic docs, 3-stage curriculum, SDK + PyPI + HF + launch assets | **Failed.** OCRBench measured **0.0%**. Prompt-format mismatch + `image_token_id` bug + data ~1000× below plan. Documented in `docs/BENCHMARKS.md`, `docs/retraining_plan.md`. |
| **v0.1 "768 retrain"** | Full retrain at 768×768, synthetic markdown + OCRBench/FUNSD/CORD, overnight/M4/Kaggle runs | **Failed.** Vision tower **dead at init** — outputs constant features for any input (cosine 1.000). Decoder memorized text templates. Root cause established before pivot. |
| **Pivot (f522c08)** | Abandon TinyDoc-768; fine-tune **SmolVLM2-2.2B-Instruct** on **46,171 real pairs** (docvqa/sroie/docmatix/ocrbench/funsd), no synthetic templates | Sensible response to a dead tower — but changed base model without changing **product thesis** or **eval methodology**. |
| **Kaggle grind v1→v31** | ~30 notebook revisions: num2words, PEFT double-wrap, `loss_type=nll`, dtype mismatches, remove BitsAndBytes → fp16, force `CUDA_VISIBLE_DEVICES=0` (DataParallel+PEFT = StopIteration), eval batch=1 OOM fix, torchao≥0.16, local `local_files_only`, hub resume, non-fatal pushes | Infra now **works**: ~71–72 s/step, loss 18.5→0.06, eval acc ~98.7% @ step 100, reached **step 493/3000** before 12 h `CellTimeoutError`. |
| **Now (v31 pending)** | EVAL_EVERY 250, hub-download resume, final push non-fatal | Blocked on **stale Kaggle HF_TOKEN** (401). Hub repo `eulogik/SmolVLM2-TinyDoc-real` has **zero checkpoints**. |

### 1.2 Assets that are real and reusable

- **Data:** 46,171 real image–QA pairs, tarball `eulogik/TinyDoc-VLM-real-data`, rebuild/validate/push scripts under `data/`.
- **Synthetic engine:** `data/synthetic/` (renderer, generator, grounding pairs) — capable of layout-templated docs with perfect GT.
- **Eval gates:** `training/grounding_gate.py` (image-dependence, entropy, QA acc) + `training/eval_instruct.py` + benchmark data under `evaluation/` (symlinked to backup disk).
- **Training infra:** `training/smolvlm2_qlora.py` + `training/kaggle/kaggle_smolvlm2.ipynb` — proven on T4, resume + hub push logic landed.
- **Product shell:** `sdk/tinydoc/`, `demo/`, `export/` (ONNX/GGUF), README/launch assets — **shipped narrative has outrun shipped weights**.

### 1.3 Honest problems in our own artifacts

1. **`docs/paper.md` contains non-measured numbers.** Table 4 (DocVQA 65.3, OCRBench 60.8, FUNSD 85.2, …) is **not** what `docs/BENCHMARKS.md` measured (OCRBench **0.0%**). Do not publish or pitch these figures.
2. **README / project document** still describe the 256M product as the headline; both published HF models are labeled legacy; 768 retrain never shipped. The public story and the repo state diverge.
3. **Grounding-gate thresholds are self-defined.** `PASS_ENT ≥ 1.0 nat` and `dep ≥ 0.70` are **not** calibrated to any baseline model’s entropy on this tokenizer; literature uses related but different constructs (e.g., VES = entropy(image) − entropy(no-image), CVPR 2026 VES-RFT). Treat as provisional until a **base-model baseline** is recorded.
4. **Observed eval entropy ≈ 0.065 vs our own gate ≥ 1.0.** With loss → 0.06 and token acc → 98.9%, this matches the documented **SFT-memorizes** pattern (Chu et al., *SFT Memorizes, RL Generalizes*): aggressive SFT on a narrow mixture collapses diversity and can degrade OOD/vision behavior. **We are likely re-creating the TinyDoc-768 failure mode with a healthier encoder.**
5. **Training recipe diverges from every official recipe we found** (see §2.3): vision tower **not frozen**, lr **2e-4** (HF SmolVLM2 notebooks use 1e-4 LoRA / 2e-5 full, **1 epoch**), multi-epoch 3000-step aggressive schedule, no general-data mixture, no assistant-only loss audit beyond TRL defaults.
6. **"QLoRA" is a misnomer in the current script** — fp16 base, no NF4. Fine for T4, but don't cite QLoRA papers for this run.

### 1.4 Compute / ops reality (verified in-session)

- Kaggle free tier: **30 h/week**, **hard 12 h per-kernel timeout**.
- T4 = Turing (sm_75): **fp16 only**, no bf16; ~14.56 GiB budget; batch 2 maxes it with grad-ckpt.
- Steady **~71 s/step** → **~500 steps / 12 h session**; 3000 steps ≈ **3 quota weeks** with checkpoint persistence.
- Model cached as Kaggle dataset → `/kaggle/input/smolvlm2-2-2b-instruct` (avoids 4.4 GB download).
- Local mirror: `/Volumes/KIOXIA 1TB/smolvlm2-2-2b-instruct/`.

---

## 2. Verified external research (no guesses)

All figures below were retrieved from primary sources (papers, model cards, vendor pricing pages) during this review. Items we could not verify are marked.

### 2.1 Where SmolVLM2-2.2B actually sits

| Benchmark | SmolVLM2-2.2B (official card) | Source |
|-----------|-------------------------------|--------|
| DocVQA (val) | **79.98** | [HF model card](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct) |
| OCRBench | **72.9** | same |
| TextVQA | 73.21 | same |
| ChartQA | 68.84 | same |

**Comparators you can download today, zero training:**

| Model | DocVQA | OCRBench | Notes | Source |
|-------|--------|----------|-------|--------|
| **PP-DocBee-2B** (Qwen2-VL-2B FT) | **90.6** (test) | **82.8** (83.5 w/ OCR post) | ViT frozen, LLM updated, ~5M samples | [arXiv:2503.04065](https://arxiv.org/html/2503.04065v2) |
| **Qwen3-VL-4B** | **94.9** (cited) | 881/1000 | general small | cited in [arXiv:2603.13398](https://arxiv.org/html/2603.13398v1); OCRBench via llm-stats |
| **DocVAL Gemma3-4B** (student) | **88.7 ANLS**, 69.1 mAP | — | validated spatial-CoT distillation | [arXiv:2511.22521](https://arxiv.org/abs/2511.22521) |
| **DocVAL Gemma3-12B** | **91.4 ANLS**, **82.4 mAP** | — | same | same |
| Human (DocVQA) | **94.36 acc / 0.981 ANLS** | — | — | [arXiv:2007.00398](https://arxiv.org/pdf/2007.00398v3) |

**Implication:** even a **successful** +6–10 ANLS fine-tune of SmolVLM2 lands ~86–90 DocVQA and likely **still trails untouched PP-DocBee-2B (90.6)**. That is the bar a checkpoint must clear to be worth downloading.

### 2.2 What actually moves the needle on documents (2024–2026)

| Method | Verified result | Takeaway for us |
|--------|-----------------|-----------------|
| **Doc-CoB** (chain-of-boxes, arXiv:2505.18603) | InternVL2-8B: DeepForm 36→80 F1, VRDU 58→94; DocVQA +0.9; **beats GPT-4o on 7/7** with 8B | **Grounding paradigm** (coarse→fine boxes) dominates IE tasks; ablation: gains are largely from the **two-stage inference paradigm**, not data alone |
| **DocVAL** (validated CoT, arXiv:2511.22521) | 95K **validated** traces (from 102K raw); no-val = 88.1/63.7 vs full = **91.4/82.4**; 25% data → 86.2 | **Quality filter ≫ volume.** Our unfiltered 46k is the weak link. Teacher CoT + validator is the recipe. |
| **ARIAL** (agentic, arXiv:2511.18192) | 88.7 ANLS / 50.1 mAP (Gemma3-27B + OCR + retrieval + localize) | Agentic OCR→RAG→GenQA→bbox pipeline works but needs **27B** — not a T4 story |
| **PP-DocBee** (arXiv:2503.04065) | Freeze ViT, update LLM only; dynamic ratio; resize 512→768 | **Freeze vision** is the standard recipe we violated |
| **olmOCR-2** (arXiv:2510.19817) | 7B + SFT 270k pages + **RLVR binary unit tests** → olmOCR-Bench 82.4 | Verifiable rewards, not just NLL — for parsing |
| **VARCO-VISION-2.0** (arXiv:2509.10105) | 4-stage + **DPO**, grounding tokens, 1.7B on-device variant | Preference stage + grounding tokens for hallucination control |
| **DocOwl 1.5** | Two-stage structure + **multi-grained text localization**; "correlating visual texts with positions helps documents" | Localization must be **supervised**, not hoped for |
| **EXSTRUCTINY** (EACL 2026) | Open VLMs: **max bbox IoU ≈ 0.14** | Text-correct ≠ spatially grounded. **2.2B + text-only QA will not learn boxes.** |

### 2.3 Official / recommended fine-tune recipes (what we should have matched)

| Source | Recipe |
|--------|--------|
| [HF SmolVLM2 blog](https://huggingface.co/blog/smolvlm2) | Full FT for 500M; **"try QLoRA" for 2.2B** |
| [HF SmolVLM2 video FT notebook](https://github.com/huggingface/smollm/blob/main/vision/finetuning/SmolVLM2_Video_FT.ipynb) | LoRA r=8, α=8, all linear proj; **freeze `vision_model` on full FT**; **1 epoch**, lr **1e-4**, bs 2 |
| [smol-course unit 3](https://huggingface.co/learn/smol-course/en/unit3/4) | ChartQA on 2.2B: 1 epoch, lr 1e-4, bs 4 × accum 4 |
| [Qwen finetune repo](https://github.com/QwenLM/Qwen2.5-VL/blob/main/qwen-vl-finetune/README.md) | **`tune_mm_vision=False` by default**; separate LRs (vision 1e-6, projector 1e-5, LLM 2e-7–1e-6) |
| Bunny (arXiv:2402.11530) | Keep **high-quality pure-text data** in the mix to protect cognition; LoRA mitigates forgetting |
| *SFT Memorizes, RL Generalizes* | Aggressive SFT ↓ OOD/vision metrics; with vision frozen use **very low LR (1e-6–1e-7)** |

**Our run vs recipes:** vision unfrozen (or at least not explicitly frozen in `smolvlm2_qlora.py` — LoRA targets LM modules only, but base vision grads are not frozen either — actually PEFT freezes base params by default; **LoRA adapters are on LM only**, vision is frozen via PEFT default — **correction: PEFT freezes non-target modules**. Still: lr 2e-4, no general mixture, 3000 steps on 46k narrow data, entropy collapse.) → align schedule + data mixture + early stop on **grounding gates**, not on loss.

### 2.4 Grounding / hallucination losses we could actually use

- **VES** (image vs no-image entropy gap) — diagnostic; **VES-RFT** uses it as an RL reward (CVPR 2026). Our gate is a crude cousin of this; calibrate against **base model** before trusting it.
- **V-DPO / CLIP-DPO / OPA-DPO** — 4.8k on-policy pairs can suffice for gains (OPA-DPO); good fit once we have a candidate adapter + failure cases.
- **Counter-evidence:** EMNLP 2024 found referring-expression + grounded-captioning objectives had **little effect** on open-generation object hallucination under sound protocol — don't expect bbox SFT alone to fix free-form hallucination.
- **Label masking:** assistant-only loss (mask system/user/image tokens) is standard in TRL VLM SFT and Roboflow Qwen extraction notebooks — **verify our `dataset_text_field="messages"` path actually masks user turns** (TRL chat template path does; confirm in a debug batch).

### 2.5 Market / business (verified pricing + product patterns)

**Cloud unit economics (primary sources, fetched 2026-09-22):**

| Vendor | Product | Price |
|--------|---------|-------|
| Google Document AI | Custom extractor / Form Parser | **$30 / 1K** ($20 over 1M) |
| Google | Layout Parser | $10 / 1K |
| Google | Invoice/Expense parser | $0.10 / count (≤10 pages) |
| Google | Enterprise OCR | $1.50 / 1K (free first 1K) |
| AWS Textract | Detect text | $0.0015 → $0.0006 / page |
| AWS Textract | **Forms** | $0.05 → $0.04 / page |
| AWS Textract | **Tables + Forms + Queries** | $0.070 → $0.055 / page |
| AWS Textract | Expense | $0.01 → $0.008 / page |
| Azure Document Intelligence | Free tier | 500 pages/mo; **"custom generative extraction" is a distinct SKU** (prices JS-rendered, not scraped) |

**What buyers pay for** (synthesis from vendor SKUs + OSS adoption patterns): extraction over plain OCR (**5–45×** unit price), schema'd fields, page/region evidence, human review loops, on-prem/privacy.

**What goes viral in doc-AI OSS** (pattern, not star counts — star data not re-verified this session): **usable pipelines and parsers** (olmOCR, MinerU, Marker, Docling, Surya), **not** another 2B adapter card. Differentiation = output format (markdown/JSON), layout fidelity, speed, local-first, and a demo you can paste a PDF into.

**Local-first:** MLX / Ollama / llama.cpp are the distribution rails for Mac+edge; a 2–4B VLM with quantization is a credible local engine; our original <1 GB / Raspberry Pi thesis is **still valid as a product constraint**, but the engine should start from a base that already scores 90+ DocVQA, not 80.

### 2.6 What we could NOT verify (do not build the plan on these)

- Exact Qwen3-VL-2B / Qwen2.5-VL-3B DocVQA test scores (tables rendered as images in HTML).
- Star counts / HN threads for olmOCR, MinerU, Marker, Docling launches (rate-limited this session).
- Azure Document Intelligence dollar prices (JS-rendered).
- Any paper that FTs **SmolVLM2-2.2B specifically** on ~46k doc QA pairs with reported DocVQA/OCRBench — **no direct precedent exists**; we are extrapolating from DocVAL/PP-DocBee/Doc-CoB.
- Unsloth does **not** list SmolVLM2 in its official model catalog.

---

## 3. Decision

### 3.1 Is the current path the best path?

**No.** Reasons, ranked by severity:

1. **Absolute ceiling is below free alternatives.** Realistic best case for this run ≈ 86–90 DocVQA ANLS (bounded by DocVAL's 19K→86.2 and 2B plateaus). **PP-DocBee-2B ships at 90.6 with zero training.** A checkpoint that loses to a free download has no download motive, no star motive, no pricing motive.
2. **Current signals say memorization, not grounding** (entropy 0.065, acc 98.9%, loss 0.06) — the same class of failure that killed TinyDoc-768, with a better encoder. We have not run `grounding_gate.py` on a completed adapter; our own gate would likely **fail**.
3. **No localization in the supervision.** Business willingness-to-pay is grounded extraction (Textract Forms $0.05/page vs OCR $0.0015). Text-only QA pairs **cannot** teach boxes (EXSTRUCTINY: IoU ≈ 0.14 ceiling for open VLMs without explicit supervision).
4. **No field-level eval, no schema, no held-out layout split.** Loss/ANLS on a random 500-pair holdout from the same mixture is not a product metric and will not survive contact with a real invoice folder (layout leakage across docvqa/sroie/funsd/docmatix is high).
5. **Ops tax is brutal and known:** 12 h wall, 30 h/week, stale token already burned weeks of attempts, hub empty. Continuing multi-week SFT on T4 for a sub-90 result is a **negative expected value** vs reusing PP-DocBee/Qwen + building eval + product.

### 3.2 Should we "innovate something marginally better / viral / business-maker"?

**Yes — but the innovation is not a cleverer LoRA config.** The innovation we can own, given our actual assets:

> **TinyDoc = local-first, grounded document extraction SDK:**
> best small open VLM(s) as engines + **layout-aware eval (field F1 + ANLS + mAP@IoU)** + **JSON schema + confidence + page-region citations** + vertical templates (receipts/invoices first) + runs on a Mac without a cloud bill.

That is **marginally better** where it matters (trust, local, priced against $30/1K custom extractors), **wantable** (one `pip install`, paste a PDF, get cited JSON), and **viral-capable** (side-by-side local vs Textract/Gemini, star-shaped demo, not a model card with loss curves).

### 3.3 What we explicitly stop

| Stop | Why |
|------|-----|
| Grinding v32+ of the same 3000-step SmolVLM2 run without gate results | Expected value negative; entropy signal already bad |
| Publishing `docs/paper.md` Table 4 / estimated benchmarks | Not measured; credibility risk |
| Treating loss / token-acc as success criteria | Memorization metrics |
| Building marketing (HN/Reddit/pitch) before a working extract demo | Story–product gap is the project's recurring failure |
| Training boxes into a model with **no bbox labels** in the 46k manifest | Impossible task; need new supervision format |

### 3.4 What we keep and double down on

- 46k real pairs as **domain adaptation + regression corpus** (not as the only training signal).
- Synthetic generators → **layout-family templates with perfect field + bbox GT**.
- Grounding-gate **idea** → upgrade to calibrated metrics (baseline first).
- Kaggle T4 pipeline → for **short** SFT/LoRA runs only (≤1 session) once recipe is fixed.
- SDK/demo/export shells → wire them to a **real engine** immediately.
- eulogik / Apache 2.0 / edge (<1 GB) positioning — still differentiated vs SaaS.

---

## 4. Recommended path (solid, evidence-linked)

Three tracks run in parallel. **Track A is the product spine. Track B is the model work. Track C is cleanup/credibility.** No track depends on 3 weeks of T4 SFT.

### Track A — Product: "Grounded local extraction" (week 1–2)

**Goal:** a demo and SDK that beat "chat with a PDF" on **trust metrics**, using **existing** open weights.

| # | Deliverable | Spec | Evidence anchor |
|---|-------------|------|-----------------|
| A1 | **Engine router** | `tinydoc.extract(image, schema=...)` → engine ∈ {`pp-docbee` (HF), `qwen2.5-vl`/`qwen3-vl` via MLX or Ollama, `smolvlm2-ft` (optional adapter), `cloud-textract` (optional)} | PP-DocBee 90.6 DocVQA; local rails: MLX/Ollama |
| A2 | **Schema layer** | JSON Schema per doc type (invoice, receipt, form); validate + retry-once on parse fail; typed fields (money, date, GSTIN/VAT patterns) | Azure sells "custom **generative** extraction" as its own SKU — schema is the product |
| A3 | **Grounding layer** | Every field carries `evidence: {page, bbox?, quote}` — bbox from engine if available (Qwen `data-bbox` style) **or** quote-match back to OCR/layout text (DocLayNet/paddle/pdfplumber level is OK for v1) | Doc-CoB/DocVAL: grounding is the differentiator; ARIAL mAP 50.1 |
| A4 | **Confidence + HITL flag** | per-field score (model logprob if available, else agreement across 2 prompts/engines); flag < threshold for review | Hyperscience/Textract Queries pattern; vendor pricing proves willingness to pay for Forms vs raw OCR |
| A5 | **Eval harness (build before more training)** | On held-out **layout families**: **field-level P/R/F1**, **ANLS**, **mAP@IoU** (if boxes), **schema-valid rate**, **grounding coverage %**. Baseline = PP-DocBee-2B + Qwen + base SmolVLM2. | KIEval (ICDAR 2025), ANLS (DocVQA), mAP@IoU (ARIAL/DLaVA) |
| A6 | **Demo** | HF Space + local CLI: drop folder → cited JSON + red-box overlay (even if boxes come from layout engine, not VLM) | What OSS users paste into Show HN |

**Acceptance gate for Track A:** on **≥100 real held-out docs of one vertical** (recommend **receipts/invoices** — we already have SROIE/CORD-shaped data), beat **OCR+regex baseline** on field F1 and ship schema-valid cited JSON from a laptop with no API key. If we cannot beat regex, **stop model training** — the product is eval + pipeline.

### Track B — Model: only if Track A shows a gap (week 2–4)

Run this **only** when A5 shows PP-DocBee/Qwen failing **our** vertical or latency/size budget.

**B1 — Decide base by evidence, not habit:**

| Constraint | Candidate | Why (verified) |
|------------|-----------|----------------|
| Best zero-shot small doc model | **PP-DocBee-2B** / Qwen2-VL-2B family | 90.6 DocVQA, 82.8 OCRBench |
| Strongest small general + bbox-native | **Qwen2.5-VL-3B** (or 7B if VRAM) | official bbox/HTML positioning |
| Tiny / edge story | keep **SmolVLM2-2.2B** only if size budget demands it | 79.98 base; ceiling ~86–90 after good FT |
| DocVAL-style distillation student | **Gemma3-4B** | 78.6→88.7 ANLS, 69.1 mAP with validated CoT |

**B2 — Fix the recipe to match literature (applies to whatever base):**

- Freeze vision encoder (PP-DocBee, Qwen official, HF notebook).
- LoRA: r=16, α=32, dropout 0.05, LM q/k/v/o + MLP (current) **or** r=8 α=8 all-linear (HF notebook) — pick one, don't invent a third.
- lr **1e-4** (LoRA) or **2e-5** (full), **1 epoch**, warmup ~50–90, early stop on **Track A metrics**, not loss.
- **Quality-filter the 46k** (drop mismatched Q/A, near-dupes, >1500-char table dumps already partly done; add image-QA consistency checks). DocVAL: unfiltered 102K → 88.1/63.7 vs validated 76K → 91.4/82.4.
- Mix **≥10–20% general/text data** (Bunny / Qwen practice) to protect cognition.
- Assistant-only loss: confirm masking on a debug batch.
- **Add localization targets only where GT exists** (FUNSD boxes, DocLayNet-style synthetic GT from our renderer, DocVQA page boxes if available). Do not emit fake boxes.
- Optional second stage: **tiny DPO/VES-style preference pass** on failure cases from A5 (VARCO stage-4 pattern; OPA-DPO: ~5k pairs can suffice).

**B3 — Compute policy:**

- **≤1 Kaggle session (~500 steps)** for any LoRA run on T4; if more, fix hub token **first** (`training/.hf_token` = `hf_TcqG...`; Kaggle Secret still stale).
- Prefer **1 epoch × filtered subset** over 3000-step grind.
- Paid burst only if B1 selects a 7B+ (A100/L4 spot ≈ hours, not weeks).

**B4 — Hard gates before calling any adapter "done":**

1. `grounding_gate.py` **after calibrating thresholds against base model** (record base dep/entropy/acc first; keep gates relative: e.g., adapter entropy ≥ 0.5 × base, dep ≥ base − 0.1).
2. Track A field-F1 **≥ PP-DocBee baseline on our vertical** (otherwise discard adapter).
3. No regression >2 points vs base on a **general** probe set (e.g., 200 held-out VQA/text pairs) — Bunny/forgetting guard.
4. ANLS on DocVQA **val subset we did not train on** (leakage audit: we trained on docvqa — need **split hygiene** or use a different public set for reporting).

**B5 — Explicit non-goal:** beating Qwen3-VL-4B (94.9) on DocVQA. We do not need leaderboard vanity; we need **vertical field-F1 + grounding + local**.

### Track C — Credibility cleanup (parallel, days)

| # | Action |
|---|--------|
| C1 | Mark `docs/paper.md` results as **DRAFT / NOT MEASURED** or replace with real numbers only |
| C2 | Rewrite README status table: published models = legacy; 768 = abandoned (dead vision tower); SmolVLM2 experiment = in progress / superseded by this plan |
| C3 | Fix Kaggle HF_TOKEN secret → `hf_TcqG...` **only if** we run Track B on Kaggle; otherwise archive kernel |
| C4 | Publish **negative results** honestly (dead tower, entropy collapse) — HN/LocalLLaMA reward this more than fake 65.3 DocVQA |
| C5 | One-page architecture: engines / eval / schema / grounding — replaces 619-line v1 project doc as source of truth |

---

## 5. Phased plan (concrete)

### Phase 0 — Freeze & baseline (days 1–2) — **DONE 2026-09-22**

- [x] Stop launching SmolVLM2 kernels until B-gates defined.
- [x] Run **eval harness skeleton (A5)** against: base SmolVLM2-2.2B, PP-DocBee-2B, one Qwen VL via Ollama/MLX — on **100 docs, one vertical**, field F1 + ANLS + schema-valid rate.
- [x] Record grounding-gate numbers for **base** (calibration).
- [x] Inventory 46k manifest: sources, target types, how many are pure VQA vs extraction vs table dumps; measure layout leakage risk (same image / near-dup across sources). → `evaluation/phase0/results/manifest_inventory.json` (265,802 rows; 46,186 real / 22,707 unique images).
- [x] C1–C2 doc truth pass.

**Exit:** a table of engine × metric on our vertical. This table decides Track B.

#### Phase 0 measured results (SROIE, n=100, fields company/date/address/total)

Harness: `evaluation/phase0/` · table: `evaluation/phase0/results/baseline_table.md` · holdout: `results/sroie_holdout.txt` (excluded from any future training).

| Engine | Field F1 | ANLS | Schema-valid | Exact-4/4 | Avg latency | Notes |
|--------|----------|------|--------------|-----------|-------------|-------|
| **ollama:qwen2.5vl:3b** | **0.870*** | **0.955*** | **1.000*** | 0.610* | 2.3–8.8 s* | free, local; unconstrained JSON + salvage; prompt-v1 (no `format=json`) |
| smolvlm2_base (2.2B) | 0.330 | 0.544 | 0.960 | 0.010 | 9.13 s | prompt-forced 4-key JSON |
| ocr_regex (floor) | 0.227 | 0.303 | 0.600 | 0.000 | 1.62 s | weak Tesseract heuristics |
| ppdocbee | — | — | — | — | — | 2B = Paddle `pdparams` only (not transformers-loadable); PPDocBee2-3B (Qwen2.5-VL safetensors, 7.5G) **download failed** (CAS Client format error after partial pull; only 4.4M config files remain at `/Volumes/KIOXIA 1TB/models/PPDocBee2-3B/`). Not needed for Phase 1 — free ollama engine already passes Track A gate. |

Per-field (pipeline n=100, current metrics): company 0.85 · date 0.94 · address **0.72** · total 0.97. **Metric fix:** company/address punctuation-insensitive (`normalize_text`). **Prompt-v1:** char-by-character address + lookalike alphabet + `num_predict=512` (synced `engines.py` / `pipeline.py`). History: format=json strict 0.840 → unconstrained strict 0.835 → metric re-score old prompt 0.875 → prompt-v1 re-run **0.870** (address 0.70→0.72, company −0.03 noise). Address misses **28** (true OCR/content errors).

**Grounding calibration vs base SmolVLM2** (`results/grounding_calib_smolvlm2.json`, 12 synthetic pages):

| Signal | Base value | Old absolute gate | Flag |
|--------|------------|-------------------|------|
| image-dependence | **1.00** | ≥0.70 | PASS |
| mean first-token entropy | **0.920 nat** | ≥1.0 | **FAIL against provisional gate** |
| QA acc (loose) | **0.348** (n=23) | ≥0.15 | PASS |

Implication: the provisional `PASS_ENT ≥ 1.0` gate is **above base** — any adapter must be judged **relative to base** (e.g. entropy ≥ 0.5 × 0.92, dep ≥ base − 0.1, QA ≥ max(base, 0.15)). Absolute entropy gate is invalidated.

**Phase 0 → Track B decision (evidence):**

1. Track A acceptance (beat OCR+regex on field F1, schema-valid, no API key): **PASS** with free `ollama:qwen2.5vl:3b` (0.870 ≫ 0.227, schema 1.000).
2. Track B trigger = free engines **fail** our vertical or size/latency budget. Qwen-3B does **not** fail accuracy on this vertical; residual gap is **address F1 0.72** (28 true content/OCR misses after metric+prompt-v1) and warm/cold **2–9 s/img** latency.
3. SmolVLM2-base (0.330) is far below the free engine — closing a **+0.54 F1** gap to justify a SmolVLM2 adapter is a high bar vs routing to Qwen/PP-DocBee2.
4. **Default: skip Kaggle grinding / Track B training for Phase 1.** Revisit only if (a) layout-family holdout shows Qwen failing — **measured 2026-09-22: free engine holds on unseen-layout templates (F1 0.955 n=11 current metrics / 0.932 strict, `evaluation/phase0/results/scores_ollama_unseen.json`)**, or (b) address/latency cannot be closed with schema+postprocess+engine router, or (c) a smaller/faster base is selected by evidence (B1).

\* Authoritative engine n=100 = **F1 0.870 · ANLS 0.955 · schema 1.000** after prompt-v1 re-run (2026-09-22). History: format=json strict 0.840 → unconstrained strict 0.835 → metric re-score old prompt 0.875 → prompt-v1 **0.870**. See `evaluation/phase0/results/scores_ollama.json` + `baseline_table.md`. Pipeline e2e **0.870**. Per-field pipeline: company 0.85 · date 0.94 · address 0.72 · total 0.97.

### Phase 1 — Product spine (days 3–10)

- [x] **A1 engine router**: `sdk/tinydoc/pipeline.py` — `build_engine(auto|ollama|ocr_regex|smolvlm2)`; `OllamaEngine` (no `format=json` — constrained decoding truncates address + drops `total`), `OcrRegexEngine` (Tesseract+regex floor), `SmolVLM2Engine` (KIOXIA path); **`RoutedEngine`** — auto = ollama if ping OK, OCR floor fills **empty** required fields only when primary fails `schema_is_valid` (never overwrites non-empty primary values). Name: `ollama:…+fb:ocr_regex`.
- [x] **A2 schema layer**: `schema_is_valid` via `jsonschema` against packaged `tinydoc/schemas/sroie_receipt.schema.json` (falls back to `evaluation/phase0/schemas/`); retry-once on missing total; `sanitize_fields` collapses degenerate VLM loops + enforces maxLength.
- [x] **A3 grounding layer**: `sdk/tinydoc/evidence.py` — Tesseract word boxes → `Evidence(kind="ocr_span", quote, bbox, page, score)`; `locate_field_evidence` + `attach_evidence`.
- [x] **A4 confidence + HITL flag**: `_field_confidence` per-field (evidence match + schema validity + value heuristics); `DocumentResult.confidence` aggregate; `evidence_coverage`.
- [x] **CLI**: `python -m tinydoc.cli extract <path> --engine ... --out results.jsonl --limit N --no-evidence`.
- [x] **Smoke verified**: n=3 CLI + n=10 pipeline eval — **schema 1.000, field F1 0.95, ANLS 0.964** (post-format=json fix). Clean prior: n=10 was F1 0.937 / schema 0.9 before fix.
- [x] **Full n=100 ReceiptPipeline e2e** (2026-09-22, prompt-v1): `evaluation/phase0/run_pipeline_eval.py --engine ollama` → **F1 0.870 · ANLS 0.952 · schema 1.000 (pipeline=metrics, 0 mismatch) · exact 0.620 · avg 7531 ms** (evidence off). Per-field: company 0.85 · date 0.94 · address 0.72 · total 0.97. Address misses **28** → `results/pipeline_address_failures.json`. Engine-only same prompt: **0.870**. Artifacts: `scores_pipeline_ollama_qwen2.5vl_3b.json`, `preds_pipeline_ollama_qwen2.5vl_3b.jsonl`. **Not** merged into Phase-0 `baseline_table.md`.
- [x] **Address metric fix** (2026-09-22): `metrics.normalize_text` — company/address alnum-token match (consistent with `evidence.py`). History: strict 0.835 → metric re-score old prompt **0.875** → prompt-v1 re-run **0.870**. 0 loose hits (gold token coverage <0.6).
- [x] **Prompt-v1** (2026-09-22): address char-by-character + lookalike alphabet (O/0, I/1, G/6, B/8, S/5, Z/2) + postcode digits exact; `num_predict=512`; synced in `engines.EXTRACT_PROMPT` and `pipeline.OllamaEngine`. n=100 + unseen re-run. Address hits **0.70→0.72** (misses 29→28); company 0.88→0.85 (entity OCR noise). Unseen pipeline address **11/11**.
- [x] **Evidence path n=100** (`--with-evidence`, 2026-09-22): fields unchanged (**F1 0.870 · schema 1.000 · exact 0.620**); mean confidence **0.403→0.694**; mean evidence coverage **0.668** (84/100 docs have ≥1 span; 16/100 cov=0 = honest OCR-miss signal); latency 2313→2823 ms warm. Artifacts: `scores_pipeline_ollama_qwen2.5vl_3b_evidence.json` (+ preds). `run_pipeline_eval.py` now suffixes `*_evidence` so it never clobbers the evidence-off baseline.
- [x] **A6**: demo overlay — `sdk/tinydoc/overlay.py` (`draw_overlay`: evidence bboxes + field chips); CLI `--overlay DIR` verified (PNG written). HF Space still pending.
- [x] **Layout-family split** + **cleaned holdout** (2026-09-22):
  - `evaluation/phase0/clean_holdout.py` → `data/training/manifest_train_clean.jsonl` (265,598 rows; **0** holdout path hits; dropped 204 contaminated SROIE rows). Report: `results/holdout_clean_report.json`.
  - `evaluation/phase0/layout_families.py` → dHash-8 single-link clusters on the 100-image holdout; ~18% of families held out as **unseen-layout** (`results/sroie_holdout_unseen.txt`, `layout_families.json`).
  - Unseen-layout scores: `results/scores_ollama_unseen.json` + `scores_ocr_regex_unseen.json` (free engine does **not** fail on held-out templates).
  - Future adapter training must use `manifest_train_clean.jsonl` (not raw `manifest.jsonl`).
  - `run_baseline.py` `merge_table()` **excludes** `scores_*_unseen.json` so the main n=100 table stays clean.
- [x] **pip install packaging** (`sdk/setup.py` 0.1.3): core deps pydantic/pillow/jsonschema; extras `ocr`/`smolvlm2`/`all`; `tinydoc` console script; packaged schema at `tinydoc/schemas/sroie_receipt.schema.json`. Verified: `pip install ./sdk` → schema in dist + `tinydoc extract --help`; editable reinstall for dev.
- [x] SDK matches `sdk/tinydoc` signatures already in repo. — *legacy `TinyDocExtractor` remains lazy-import for 256M path; pipeline surface is the product API.*
- [x] **Address postprocess attempts (2026-09-23, negative):** multi-PSM OCR fusion / OCR-hinted re-extract / crop re-extract / 4-way majority vote fixed **0–2 of 28** misses (oracle-with-gold 27/28 is invalid). Address stays **0.72**; ship-bar 0.80 **not met** — do not claim otherwise. Full table: `evaluation/phase0/README.md`.
- [x] **Second vertical FUNSD n=50** (2026-09-23): field F1 **0.352**, schema **0.880**, company/header 0.50 · answers 0.40. Receipt-shaped key mapping over form NER tags — different task than SROIE; no FT. Artifacts: `results/scores_funsd_ollama.json`, `preds_funsd_ollama.jsonl`, runner `run_funsd_eval.py`. CORD images still 0 on disk (HF download in progress / previously missing).
- [x] **Local Gradio demo** (`demo/app.py` rewrite): `ReceiptPipeline("auto")` + evidence + overlay + field table; smoke `HTTP 200` on `:7861`. Launch: `PYTHONPATH=sdk python demo/app.py --share --port 7860`.
- [x] **SDK unit tests** `tests/test_pipeline.py` — **30 passed** (schema, sanitize/collapse, `_extract_json` salvage, RoutedEngine fill-only-on-schema-fail, confidence, ReceiptPipeline.extract with stub engine, folder limit).

**Exit:** `pip install` → cited JSON on a folder of receipts, laptop-only. This is the **launchable artifact**.

**Phase 1 engineering notes (2026-09-22):**
- Do **not** set `format: "json"` in Ollama chat for this model — constrained decoding truncates long address fields and often returns empty `total`. Unconstrained + `_extract_json` salvage works reliably.
- `_extract_json` salvages truncated JSON (closes unclosed strings/braces, drops trailing incomplete fields).
- `collapse_repetition` cuts degenerate n-gram loops from VLM address hallucination (e.g. `"TAMPO 81200, PERAK," × 50` → single copy).
- pydantic required in venv (`sdk/tinydoc/models.py`); venv at `/tmp/tinydoc_phase0/venv`.
- Symlink caveat: `evaluation`, `data/datasets`, `data/training` are symlinks into `/Volumes/KIOXIA 1TB/TinyDoc-VLM-Backup/` — never `resolve()` for repo-root detection; use `Path(__file__).absolute()`.

### Phase 2 — Close the measured gap (days 10–25) — conditional

- [ ] If Phase 0/1 show gap: B1 model choice → B2 recipe → short Kaggle/paid run → B4 gates → merge into A1 as `engine="tinydoc-ft"`.
- [x] If no gap: **skip training entirely**; invest in schemas, doc types, and grounding quality. — **Decision 2026-09-22/23:** overall field F1 0.870 + unseen 0.955 + FUNSD second vertical measured at 0.352 (documented residual). Address-only 0.72 does **not** by itself trigger FT (postprocess tried and failed; revisit only with a stronger base per B1, not more SmolVLM2 QLoRA).

**Exit:** adapter passes B4 **or** explicit decision that base engines suffice. — **Decision: base engines suffice for Phase 1 ship bar** (address 0.80 sub-target open; other three ship items done).

### Phase 3 — Launch (days 25–40)

- [ ] Honest benchmark blog: methodology + numbers from A5 (including failures).
- [ ] Demo video: local extraction with evidence boxes vs API cost calculator (using **verified** Textract/Doc AI prices in §2.5).
- [ ] Show HN / r/LocalLLaMA / HF Space — **lead with product**, not with loss curves.
- [x] Local Gradio demo with `share=True` — `demo/app.py` (ReceiptPipeline; HF Space still separate/pending).
- [ ] Optional: distill/publish adapter only if it wins B4.

### Phase 4 — Business wedge (post-launch)

- [ ] Vertical packs (invoice, receipt, form) as schema + few-shot templates — the thing custom extractors charge **$30/1K** for.
- [ ] Self-host SKU story: "your Mac/your VPC, $0/page after hardware" vs cloud unit prices.
- [ ] Only then: OpenRouter listing, paid API, or enterprise HITL console.

---

## 6. Options considered and rejected

| Option | Why rejected (evidence) |
|--------|-------------------------|
| **Just finish 3000-step SmolVLM2 QLoRA and ship adapter** | Ceiling ~86–90 < free PP-DocBee 90.6; entropy collapse; no boxes; no field eval; multi-week T4 tax |
| **Rebuild 256M from scratch again** | Two failed training eras; no evidence 256M can hit >85 DocVQA (project doc target never approached); opportunity cost |
| **Agentic 27B pipeline (ARIAL-style) as the core** | Real SOTA (88.7/50.1) but wrong size/cost story for eulogik edge brand; can bolt on later as optional engine |
| **Pure eval/dataset paper (no product)** | Lower viral/business ceiling than a local cited-JSON tool; still do publish benchmarks as marketing |
| **Ignore grounding, optimize ANLS only** | ANLS is crowded (90+ commodity); ground+schema+local is where pricing and stars live |

---

## 7. Risk register

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| PP-DocBee/Qwen weak on **our** vertical | Medium | Phase 0 measures this first; short B2 run is the fallback |
| Layout leakage makes our eval lie | High if unaddressed | **Layout-family holdout** mandatory before any "we beat X" claim |
| Kaggle quota/token burns another week | High historically | Token fix **before** any run; cap runs at 1 session; consider $5–15 paid burst |
| Schema retry loops mask bad extraction | Medium | Report schema-valid **and** field-F1 separately; never ship schema-valid alone |
| Box claims without GT | Medium | v1 evidence = **text span + layout engine bbox**, clearly labeled; VLM boxes only with supervised GT |
| Docs/paper.md leaks into a pitch | Medium | C1 immediately |
| Scope creep to multi-vertical | Medium | Ship **one** vertical end-to-end first (Phase 1 exit) |

---

## 8. Success criteria (what "done" means)

**Product (primary):**
1. Local, no-API-key extraction on a held-out layout split of **one vertical** with **field F1 ≥ OCR+regex baseline** and **≥ PP-DocBee field F1** (or documented delta).
2. Every field has evidence (quote ± bbox); schema-valid rate reported honestly.
3. Demo installable in <5 minutes; works offline on Mac.

**Model (optional, only if run):**
4. Passes calibrated grounding gates + no general-probe regression + wins on our vertical F1.

**Credibility:**
5. Zero unpublished/misleading benchmark numbers in README/paper.
6. Negative results (768 tower, memorization) written up — trust asset.

**Business:**
7. One pricing page or calculator using **verified** AWS/Google rates vs local cost.
8. Clear wedge: schema packs / self-host / HITL — not "yet another VLM."

---

## 9. Immediate next actions (when execution resumes)

1. **Phase 0 day 1:** stand up eval harness on 100 real docs (receipts or invoices) → baseline table for SmolVLM2-base, PP-DocBee-2B, one local Qwen.
2. **Calibrate** `grounding_gate.py` against base model outputs; save `grounding_gate.base.json`.
3. **Do not** start Kaggle v32 until (1)+(2) exist; if/when needed, rotate Kaggle secret to `hf_TcqG...` first.
4. **C1/C2** honesty pass on `docs/paper.md` + README.
5. Pick **one vertical** and write its JSON Schema (A2) — this locks the product surface.

---

## 10. Source index (primary)

- SmolVLM2-2.2B card: https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct  
- HF SmolVLM2 FT blog/notebook: https://huggingface.co/blog/smolvlm2 · https://github.com/huggingface/smollm/blob/main/vision/finetuning/SmolVLM2_Video_FT.ipynb  
- smol-course unit 3: https://huggingface.co/learn/smol-course/en/unit3/4  
- PP-DocBee: https://arxiv.org/html/2503.04065v2  
- DocVAL: https://arxiv.org/abs/2511.22521  
- Doc-CoB: https://arxiv.org/abs/2505.18603  
- ARIAL: https://arxiv.org/html/2511.18192v1  
- olmOCR-2: https://arxiv.org/abs/2510.19817  
- VARCO-VISION-2.0: https://arxiv.org/abs/2509.10105  
- DocOwl 1.5: https://aclanthology.org/2024.findings-emnlp.175/  
- Qwen VL finetune: https://github.com/QwenLM/Qwen2.5-VL/blob/main/qwen-vl-finetune/README.md  
- Bunny: https://arxiv.org/pdf/2402.11530  
- SFT Memorizes, RL Generalizes: https://tianzhechu.com/SFTvsRL/assets/sftvsrl_paper.pdf  
- EXSTRUCTINY (bbox IoU): https://aclanthology.org/2026.eacl-long.265.pdf  
- DocVQA: https://arxiv.org/pdf/2007.00398v3  
- Google Doc AI pricing: https://cloud.google.com/document-ai/pricing  
- AWS Textract pricing: https://aws.amazon.com/textract/pricing/  
- Azure Doc Intelligence pricing: https://azure.microsoft.com/en-us/pricing/details/ai-document-intelligence/  
- KIEval: https://arxiv.org/html/2503.05488  
- ANLS\*: https://arxiv.org/pdf/2402.03848  
- Unsloth model catalog: https://unsloth.ai/docs/get-started/unsloth-model-catalog.md  

**Internal:** `README.md`, `TinyDoc-VLM-Project-Document.md`, `docs/BENCHMARKS.md`, `docs/retraining_plan.md`, `docs/paper.md`, `training/smolvlm2_qlora.py`, `training/grounding_gate.py`, git history through `26e812d`.

---

*If only one sentence survives review: **stop optimizing a checkpoint that cannot beat a free download; ship grounded local extraction with honest field-level numbers, and train only where those numbers show a gap.***
