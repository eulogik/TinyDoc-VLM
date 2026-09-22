# Phase 0 — SROIE field extraction baselines (measured)

**Status:** complete (2026-09-22) · n=100 full runs for 3 engines · grounding base-calibrated · **layout-family split + cleaned holdout landed**  
**Vertical:** SROIE receipts → `{company, date, address, total}`  
**Holdout:** `results/sroie_holdout.txt` — **exclude from any future training**; use `data/training/manifest_train_clean.jsonl` for any adapter training.

## Measured results (`results/baseline_table.md`)

| Engine | Field F1 | ANLS | Schema-valid | Exact-4/4 | Avg latency |
|--------|----------|------|--------------|-----------|-------------|
| **ollama:qwen2.5vl:3b** | **0.870** | **0.955** | **1.000** | 0.610 | 2573 ms* |
| smolvlm2_base | 0.330 | 0.544 | 0.960 | 0.010 | 9133 ms |
| ocr_regex (floor) | 0.227 | 0.303 | 0.600 | 0.000 | 1623 ms |

\* Warm-server local latency; earlier cold runs measured ~8–9 s/img. F1/schema stable across re-runs.

**Metric note (2026-09-22):** company/address use punctuation-insensitive token match (`normalize_text` in `metrics.py`, same as `evidence.py`). Strict normalize under-counted content-equal addresses. History: format=json strict 0.840 → unconstrained strict 0.835 → metric re-score (old prompt) **0.875** → prompt-v1 re-run (current) **0.870**. Prompt-v1 trades −0.005 overall for address **0.70→0.72** (character-by-character rules); company 0.88→0.85 (noise-level OCR entity confusions).

**Prompt (current):** shared `EXTRACT_PROMPT` in `engines.py` + `pipeline.py` — address char-by-char, lookalike alphabet, `num_predict=512`, no `format=json`. Re-run n=100 (both paths) 2026-09-22.

Per-field (pipeline n=100): company 0.85 · date 0.94 · **address 0.72** · total 0.97.

**Grounding base (SmolVLM2):** dep=1.00, mean_entropy=**0.920**, qa_acc=0.348 → provisional `PASS_ENT≥1.0` is **above base**; use relative gates (`results/grounding_calib_smolvlm2.json`).

**Exit decision:** free Qwen beats OCR floor and schema=1.000 → **skip Track B training** for Phase 1 (see `docs/pivot_plan.md` Phase 0). PP-DocBee-2B is Paddle-only; PPDocBee2-3B download **failed** (CAS error, only 4.4M config at `/Volumes/KIOXIA 1TB/models/PPDocBee2-3B/` — not needed for Phase 1).

**Ollama note:** Phase-0 baseline was originally measured with Ollama `format: "json"` (truncates addresses / drops `total`). Now unconstrained + salvage + prompt-v1. Current authoritative engine n=100: **F1 0.870 · ANLS 0.955 · schema 1.000** (`results/scores_ollama.json` + `baseline_table.md`).

**Contamination:** all 972 SROIE images appeared in `data/training/manifest.jsonl`. **Cleaned 2026-09-22:** `python evaluation/phase0/clean_holdout.py` → `data/training/manifest_train_clean.jsonl` (265,598 rows, **0** holdout hits; report `results/holdout_clean_report.json`). Free-engine numbers were never affected (models not trained on our data); **future adapter training must use the cleaned manifest.**

**Storage:** large files on `/Volumes/KIOXIA 1TB` only (local disk was 81% full). Checkpoints symlinks under `checkpoints_local/` on KIOXIA.

## What this measures

| Metric | Definition |
|--------|------------|
| **Field P/R/F1** | Micro over (example × field); company/address = punctuation-insensitive alnum tokens; type-aware match (money, date) after normalize |
| **ANLS** | DocVQA-style normalized Levenshtein, threshold 0.5, mean over 4 fields |
| **Schema-valid** | `jsonschema` against `schemas/sroie_receipt.schema.json` |
| **Exact-4/4** | All four fields match |

## Run

```bash
# venv: /tmp/tinydoc_phase0/venv  (transformers, torchvision, num2words, jsonschema, pillow)
export PATH="/opt/homebrew/bin:$PATH"   # tesseract for ocr_regex
python evaluation/phase0/run_baseline.py --engine ocr_regex
python evaluation/phase0/run_baseline.py --engine smolvlm2
python evaluation/phase0/run_baseline.py --engine ollama          # qwen2.5vl:3b
python evaluation/phase0/calibrate_grounding.py --engine smolvlm2
python evaluation/phase0/run_baseline.py --engine ocr_regex --table-only  # rebuild table
```

Scores land in `results/scores_*.json`; merged table `results/baseline_table.md`.

## Decision rule (from pivot_plan.md)

- If a **free engine** already wins on field F1 + schema + local latency → **skip Track B training**.
- If gap → short quality-filtered FT on a stronger base (Track B), gated on these metrics **not** on train loss.
- Grounding gates: use **base-relative** rules from `results/grounding_calib_*.json`, not absolute entropy ≥ 1.0 alone.

## Layout leakage note

Original split was **image-level**, not layout-family. **Layout-family split landed 2026-09-22:**

```bash
python evaluation/phase0/clean_holdout.py     # exclude holdout from training manifest
python evaluation/phase0/layout_families.py   # dHash-8 → families; ~18% held out as unseen-layout
python evaluation/phase0/run_baseline.py --engine ollama \
  --eval-path evaluation/phase0/results/sroie_eval_unseen.json --out-suffix _unseen
python evaluation/phase0/rescore_preds.py     # re-score existing preds after metrics changes
```

Unseen-layout scores live in `results/scores_*_unseen.json` (n=11 held-out template families): **ollama F1 0.955 · schema 1.000** · pipeline e2e **0.955** (address hits 11/11) · ocr_regex F1 0.132. Full-holdout vs unseen comparison is in `results/layout_families.json` + those score files. Inventory: `results/manifest_inventory.json` (265,802 rows; 46,186 real over 22,707 unique images).

## Phase 1 product status (2026-09-22)

Pipeline lives at `sdk/tinydoc/` (`ReceiptPipeline`, `build_engine`, evidence, confidence, overlay, CLI).

### Full n=100 ReceiptPipeline eval (authoritative e2e)

```bash
python evaluation/phase0/run_pipeline_eval.py --engine ollama
# → results/scores_pipeline_ollama_qwen2.5vl_3b.json
# → results/preds_pipeline_ollama_qwen2.5vl_3b.jsonl
# → results/pipeline_address_failures.json
```

| Metric | Value |
|--------|-------|
| Field F1 | **0.870** |
| ANLS | 0.952 |
| Schema-valid (pipeline + metrics) | **1.000** (0 mismatches) |
| Exact-4/4 | 0.620 |
| Avg latency | 7531 ms (evidence off; includes sanitize + jsonschema) |
| Mean confidence (no evidence) | 0.403 (coverage 0 by design) |
| Per-field hit | company 0.85 · date 0.94 · **address 0.72** · total 0.97 |
| Address misses | **28** / 100 → `results/pipeline_address_failures.json` |

Prompt-v1 re-run (char-by-char address rules + `num_predict=512`), same metrics as Phase-0. Engine-only same run: **0.870**. Prior metric-fix re-score of old prompt: 0.873 (pipeline) / 0.875 (engine) — address was 0.71. Pipeline scores are **not** merged into `baseline_table.md`.

Engine-only smoke n=3 + n=5/n=10: schema 1.000. Unseen-layout n=11 pipeline: **F1 0.955 · address 11/11 · schema 1.000**.

### Evidence path (n=100, `--with-evidence`)

```bash
python evaluation/phase0/run_pipeline_eval.py --engine ollama --with-evidence
# → results/scores_pipeline_ollama_qwen2.5vl_3b_evidence.json  (never clobbers evidence-off)
# → results/preds_pipeline_ollama_qwen2.5vl_3b_evidence.jsonl
```

| Metric | Evidence off | Evidence on |
|--------|--------------|-------------|
| Field F1 / ANLS / schema | 0.870 / 0.952 / 1.000 | **0.870 / 0.952 / 1.000** (same fields) |
| Exact-4/4 | 0.620 | 0.620 |
| Mean confidence | 0.403 | **0.694** |
| Mean evidence coverage | 0.0 (by design) | **0.668** (84/100 docs ≥1 field; 67/100 cov≥0.75; 16/100 cov=0) |
| Avg latency (warm) | 2313 ms | 2823 ms (+Tesseract) |
| Address misses | 28 | 28 |

Evidence does **not** change extracted fields (ocr_span attach only). Coverage 0 means Tesseract found no matching span — honest HITL signal, not a field error. Script writes `*_evidence` suffix so evidence-on never overwrites the evidence-off baseline.

Install + run:

```bash
export PATH="/opt/homebrew/bin:$PATH"
pip install ./sdk          # or: pip install -e ./sdk
tinydoc extract <img_or_dir> --engine ollama --out results.jsonl --overlay overlays/
# dev without install:
PYTHONPATH=sdk /tmp/tinydoc_phase0/venv/bin/python -m tinydoc.cli extract <img> --engine ollama
```

Do **not** set `format: "json"` in Ollama chat — constrained decoding truncates address + drops total. Use unconstrained + `_extract_json` salvage. Evidence boxes are OCR-span (Tesseract), labeled `ocr_span` — not VLM boxes.
