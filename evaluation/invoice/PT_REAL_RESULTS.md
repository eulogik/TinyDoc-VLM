# Real Portuguese probe — first results (2026-11-12)

**Status:** measured once on the frozen seed-42 development probe. The
100-document holdout is prepared but **has not been run**. This is a five-field
real-document transfer probe, not the nine-field synthetic invoice benchmark and
not an external leaderboard.

## Setup

| | |
|---|---|
| Documents | 100 real phone photos (80 company, 20 personal), seed 42 |
| Source | [PT_REAL.md](PT_REAL.md), CC BY 4.0, Zenodo 6371710 + 7213544 |
| Model | `qwen3-vl:2b-instruct` via Ollama |
| Prompt | canonical page prompt v1, page 1 of 1 |
| Input | long side 1400 px (the product's page cap), JPEG q92 |
| Context / output cap | 8192 / 2048 tokens |
| Scored fields | vendor, invoice number, date, currency, tax, total |
| Scored judgments | 577 (some documents lack a gold field) |
| Failures | 1 document hit the 2048-token cap with runaway line items (`done_reason: length`, 45 s) and produced no JSON; it is counted wrong |
| Latency | p50 8.7 s, p90 11.0 s per document |

The images are resized before inference with
`evaluation/invoice/prepare_pt_probe.py`; the source and prepared hashes are in
the manifest. The prompt and schema were frozen before the run.

## Headline

The shipped 0.3.0 merge **double-counted VAT on tax-inclusive receipts**: when
the reader returned a printed total and a VAT amount, `reconcile_totals`
rewrote the total as `subtotal + tax`. On this probe that turned 50 correct
printed totals into wrong ones.

| Path | Field accuracy | Documents fully correct | All six correct (n=80) |
|---|---:|---:|---:|
| Raw reader (no merge) | 428/577 = **74.2%** | 18/100 = **18%** | 13/80 = **16.3%** |
| 0.3.0 shipped merge | 374/577 = **64.8%** | 3/100 = **3%** | 2/80 = **2.5%** |
| Unreleased merge fix (same raws) | 428/577 = **74.2%** | 18/100 = **18%** | 13/80 = **16.3%** |

Exact 95% intervals (Clopper-Pearson):

| Path | Field accuracy | Fully correct |
|---|---:|---:|
| 0.3.0 | 60.8–68.7% | 0.6–8.5% |
| Merge fix | 70.4–77.7% | 11.0–26.9% |

The field interval treats fields as independent and is optimistic because fields
cluster within documents. Cite the document-level interval for fully-correct
claims.

The merge fix changes **no model output**. It was applied by rebuilding the
saved page raws with `rebuild_pagewise.py`; it preserves a printed total and a
printed tax, uses a derived line sum only for subtotal checks, and repairs a
total only when it is structurally impossible (less than 95% of a printed
subtotal). One correction remains in the 100 documents; 84/100 documents carry
at least one arithmetic review flag, so inconsistencies are surfaced rather
than silently rewritten.

## Per field

| Field | 0.3.0 merge | Merge fix | 95% CI (merge fix) |
|---|---:|---:|---:|
| Currency | 96/100 (96.0%) | 96/100 (96.0%) | 90.1–98.9% |
| Invoice date | 93/97 (95.9%) | 93/97 (95.9%) | 89.8–98.9% |
| Invoice number | 44/98 (44.9%) | 44/98 (44.9%) | 34.8–55.3% |
| Tax | 67/87 (77.0%) | 68/87 (78.2%) | 68.0–86.3% |
| Total | 19/100 (19.0%) | 72/100 (72.0%) | 62.1–80.5% |
| Vendor | 55/95 (57.9%) | 55/95 (57.9%) | 47.3–68.0% |

By source, merge fix: company 345/458 fields (75.3%), personal 83/119 (69.7%).

## Residual categories (0.3.0 merge)

| Category | Count | Meaning |
|---|---:|---|
| `total_double_counted_tax` | 50 | printed total rewritten as subtotal + tax |
| `vendor_other` | 27 | wrong seller (customer, bank, operator, or unrelated text) |
| `number_other` | 26 | wrong document number shape or unrelated number |
| `total_other` | 24 | misread total, often a line amount or a digit error |
| `tax_other` | 19 | misread VAT amount |
| `number_digit_or_shape` | 17 | one or two wrong digits in a long number |
| `vendor_alias_contains` | 12 | brand vs legal entity (e.g. `jumbo` vs `JUMBO AMOREIRAS`) |
| `number_empty` | 11 | no number returned |
| `total_dot_decimal_x1000` | 5 | `8.88` read as `8880` (Portuguese dot-decimal) |
| `currency_wrong_iso` | 4 | 2 `BRL`, 1 `E`, 1 empty |
| `date_wrong` | 4 | day/month or digit errors |
| other | 4 | one missing tax, one missing total, one line amount, one empty vendor |

After the merge fix, the `total_double_counted_tax` category is gone; the
remaining total errors are the reader's (dot-decimal, line amount, missing) and
the remaining fields are unchanged.

## What this says about the product

- **The merge was the largest real-world defect.** A deterministic rewrite that
  looked safe on tax-exclusive synthetic invoices was wrong on tax-inclusive
  receipts. Preserving printed totals recovers 53 documents without a new model
  call.
- **The reader is usable on dates and currency, weak on identifiers and
  vendors.** Dates 95.9%, currency 96.0%, but invoice numbers 44.9% and vendors
  57.9%. Long numeric identifiers are misread at 1400 px, and the transcription
  labels often use a brand while the page prints a legal entity or operator.
- **Totals are still not solved.** Even with the merge fixed, 28/100 totals are
  wrong: five dot-decimal ×1000 errors, nine model-computed totals that add tax
  to a tax-inclusive amount, and the rest line/amount/digit errors. A
  receipt-aware prompt (header-only pass, explicit tax-inclusive rule,
  Portuguese dot-decimal rule) is the next experiment.
- **One runaway failure.** A Zara Home receipt generated line items until the
  2048-token cap (`done_reason: length`); the JSON was truncated and the
  document scored zero. The line-item pass is a liability on receipts with many
  SKUs.

## What must not be claimed

- Not a full nine-field invoice benchmark, not a supplier-invoice benchmark,
  not an external leaderboard, and not comparable to the synthetic benchmark's
  key-header numbers.
- No line-item, due-date, flag-quality, or confidence-calibration result.
- The 100 documents are a development set. The merge fix was designed after
  seeing their failures, so its 74.2% is a **post-hoc** result. The prepared
  seed-43 holdout is the validation set; until it is run, do not present 74.2%
  as a held-out estimate.
- The 0.3.0 wheel on PyPI does **not** contain the merge fix.

## Artifacts

- Rows: `results/rows_invoice_pt_probe100_1400.jsonl` (shipped merge) and
  `results/rows_invoice_pt_probe100_1400_mergefix5.jsonl` (same raws, fixed merge)
- Scores: `results/scores_invoice_pt_probe100_1400_full.json` (0.3.0),
  `results/scores_invoice_pt_probe100_1400_mergefix5.json` (fix)
- Summaries: the matching `summary_invoice_pt_probe100_1400_*.json`
- Audit: `results/audit_invoice_pt_probe100_1400.json`
- Residuals: `results/residuals_pt_probe100_1400.json` / `.md`
- Prep: `results/pt_real_probe_prep_1400.json`

## Reproduce

```bash
export PATH="/opt/homebrew/bin:$PATH"
ROOT=/var/folders/6m/l_wd40y91jqbj36ty4nz2skm0000gn/T/opencode/datasets/pt_invoices

# Inference (multi-document run; get approval first)
scripts/run_pt_probe.sh pt_probe100_1400 \
    evaluation/invoice/manifest_pt_real_probe100_1400.jsonl 2048 0

# Full predictions (keep parsed docs even when the merge-sanity flag fires)
python3 evaluation/invoice/preds_from_pagewise_rows.py \
    --rows evaluation/invoice/results/rows_invoice_pt_probe100_1400.jsonl \
    --out evaluation/invoice/results/preds_invoice_pt_probe100_1400_full.json \
    --raw-out evaluation/invoice/results/preds_invoice_pt_probe100_1400_raw.json

# Score and audit
python3 evaluation/invoice/pt_score.py \
    --manifest evaluation/invoice/manifest_pt_real_probe100_1400.jsonl \
    --preds evaluation/invoice/results/preds_invoice_pt_probe100_1400_full.json \
    --rows evaluation/invoice/results/rows_invoice_pt_probe100_1400.jsonl \
    --scores evaluation/invoice/results/scores_invoice_pt_probe100_1400_full.json \
    --summary evaluation/invoice/results/summary_invoice_pt_probe100_1400_full.json
python3 evaluation/invoice/audit_pt_rows.py \
    --manifest evaluation/invoice/manifest_pt_real_probe100_1400.jsonl \
    --rows evaluation/invoice/results/rows_invoice_pt_probe100_1400.jsonl \
    --out evaluation/invoice/results/audit_invoice_pt_probe100_1400.json \
    --allow-failed 1

# Merge fix from the same raws (no model calls)
python3 evaluation/invoice/rebuild_pagewise.py \
    --rows evaluation/invoice/results/rows_invoice_pt_probe100_1400.jsonl \
    --out-rows evaluation/invoice/results/rows_invoice_pt_probe100_1400_mergefix5.jsonl \
    --out-preds evaluation/invoice/results/preds_invoice_pt_probe100_1400_mergefix5.json
```
