# Real Portuguese probe — results (2026-11-12)

**Status:** measured on a seed-42 development set and validated on a disjoint
seed-43 holdout. Both are 100-document, five-field real-document probes. This is
not the nine-field synthetic invoice benchmark, not a B2B supplier-invoice
benchmark, and not an external leaderboard.

## Setup

| | |
|---|---|
| Documents | 100 real phone photos per split (80 company, 20 personal) |
| Source | [PT_REAL.md](PT_REAL.md), CC BY 4.0, Zenodo 6371710 + 7213544 |
| Model | `qwen3-vl:2b-instruct` via Ollama |
| Prompt | canonical page prompt v1, page 1 of 1 |
| Input | long side 1400 px (the product's page cap), JPEG q92 |
| Context / output cap | 8192 / 2048 tokens |
| Scored fields | vendor, invoice number, date, currency, tax, total |
| Development set | seed 42, 577 scored fields |
| Holdout | seed 43, disjoint, 582 scored fields |

The images are resized before inference with
`evaluation/invoice/prepare_pt_probe.py`; source and prepared hashes are in the
manifests. The prompt and schema were frozen before each run.

## Headline

The shipped 0.3.0 merge **double-counted VAT on tax-inclusive receipts**: when
the reader returned a printed total and a VAT amount, `reconcile_totals`
rewrote the total as `subtotal + tax`. On the development set that turned 50
correct printed totals into wrong ones.

| Development set (same model outputs) | Field accuracy | Fully correct | All six (n=80) |
|---|---:|---:|---:|
| 0.3.0 shipped merge | 374/577 = **64.8%** | 3/100 = **3%** | 2/80 = **2.5%** |
| Preserve-printed-total merge | 428/577 = **74.2%** | 18/100 = **18%** | 13/80 = **16.3%** |

The fix changes **no model output**. It was applied by rebuilding the saved page
raws with `rebuild_pagewise.py`: a printed total and a printed tax are
authoritative, a derived line sum cannot drive a tax overwrite, and a total is
repaired only when it is missing or structurally impossible. One correction
remains in the 100 development documents; 84/100 carry an arithmetic review
flag, so inconsistencies are surfaced rather than silently rewritten.

## Holdout validation

The seed-43 holdout was frozen before the merge fix was designed and run once
with the fixed merge. It did **not** reuse any development document.

| Holdout (fixed merge) | Field accuracy | Fully correct | All six (n=84) |
|---|---:|---:|---:|
| Seed-43, 100 documents | 419/582 = **72.0%** | 22/100 = **22%** | 15/84 = **17.9%** |

| Exact 95% interval (Clopper-Pearson) | Field accuracy | Fully correct |
|---|---:|---:|
| Development | 70.4–77.7% | 11.0–26.9% |
| Holdout | 68.2–75.6% | 14.3–31.4% |

Development → holdout: field accuracy **−2.2 points**, fully-correct rate
**+4.0 points**, all-six **+1.6 points**. The field intervals overlap heavily;
the merge fix generalizes. **72.0% / 22-of-100 is the held-out real-document
headline.** 74.2% remains the development-set number.

The field interval treats fields as independent and is optimistic because fields
cluster within documents. Cite the document-level interval for fully-correct
claims.

## Prompt pilot (negative)

A 30-document development pilot tested two receipt-aware prompts: header-only
(v3) and v1 plus receipt rules (v4). v3 gained **+2.9 field points** but **2/30
responses looped to the 2048-token cap**; v4 lost **1.2 points**. The canonical
**v1 prompt remains the default**; no full-development or holdout run was
justified. Details and artifacts: [PT_PROMPT_PILOT.md](PT_PROMPT_PILOT.md).

## Resolution pilot (promising)

On the 30-document development pilot, raising the input from 1400 px to 2400 px
with prompt v1 adds **+4.6 field points** (141/174 vs 133/174) and 4 more
fully-correct documents, at **4.1× latency**. On a fixed **50-document holdout
subset**, the gain is smaller but still positive: **73.0% → 76.5%** (+3.5
points, +1 fully-correct document), at 4.4× latency. The field pattern is mixed:
tax **+8** and invoice number **+5**, but total **−2** and vendor **−3**. A fixed
field hybrid reaches 78.2% but needs both runs. This is a candidate for a
**targeted high-resolution retry for tax and identifiers**, not a new default.
Details: [PT_RESOLUTION_PILOT.md](PT_RESOLUTION_PILOT.md).

## Per field

| Field | Development | Holdout | Δ (pp) |
|---|---:|---:|---:|
| Currency | 96/100 (96.0%) | 96/100 (96.0%) | 0.0 |
| Invoice date | 93/97 (95.9%) | 91/98 (92.9%) | −3.0 |
| Invoice number | 44/98 (44.9%) | 41/98 (41.8%) | −3.1 |
| Tax | 68/87 (78.2%) | 59/91 (64.8%) | −13.3 |
| Total | 72/100 (72.0%) | 76/100 (76.0%) | +4.0 |
| Vendor | 55/95 (57.9%) | 56/95 (58.9%) | +1.1 |

Tax is the least stable field across splits. The merge fix preserves printed
tax; the remaining tax errors are the reader's (wrong VAT line, multi-rate
receipts, or a total/tax confusion). Invoice numbers and vendors remain the
weakest fields on both splits.

## Incidents and audit

Development: 99/100 parsed; 1 receipt generated line items until the
2048-token cap (`done_reason: length`) and produced no JSON. 84/100 carry at
least one arithmetic review flag after the fix.

Holdout: 98/100 stopped cleanly. Three documents were incomplete:

- `ptreal-personal-20210215_165018` — runaway line items, hit the 2048-token
  cap, no JSON (`done_reason: length`).
- `ptreal-company-20210323_184527` — truncated response with **no
  `done_reason`**; partial JSON had a vendor and number but no total. This is a
  runner/transport truncation, not a clean model stop.
- `ptreal-company-20210429_192628` — valid JSON but the model omitted the total
  (`done_reason: stop`); other fields are still scored.

The audit allows the two hard failures and lists all three merge-sanity flags.
Latency: development p50 8.7 s / p90 11.0 s; holdout p50 8.4 s / p90 10.4 s.

## Residuals (holdout, fixed merge)

| Category | Count | Meaning |
|---|---:|---|
| `tax_other` | 30 | misread VAT line/amount, multi-rate receipt, or total/tax confusion |
| `number_other` | 27 | wrong document number shape or unrelated number |
| `vendor_other` | 25 | wrong seller (customer, bank, operator, or unrelated text) |
| `number_empty` | 19 | no number returned |
| `vendor_alias_contains` | 13 | brand vs legal entity (e.g. `jumbo` vs `JUMBO AMOREIRAS`) |
| `total_double_counted_tax` | 11 | **reader-computed** total adds tax to a tax-inclusive amount |
| `number_digit_or_shape` | 11 | one or two wrong digits in a long number |
| `total_other` | 6 | misread/line amount |
| `date_wrong` | 7 | day/month or digit errors |
| `total_missing` | 3 | no total returned |
| `currency_wrong_iso` | 4 | wrong ISO code |
| `total_dot_decimal_x1000` | 1 | Portuguese dot-decimal ×1000 |
| other | 6 | selected line amount (3), missing tax (2), empty vendor (1) |

The merge fix removed the **merge-caused** double count; the remaining 11 are in
the model's raw text. A receipt-aware prompt must state that a printed total is
tax-inclusive and that Portuguese `4.190` means `4.19`, and should stop the
line-item pass from running away.

## What this says about the product

- **The merge bug is fixed and the fix generalizes.** The held-out product path
  moves from a 19% total accuracy regime (0.3.0 development) to 76% total
  accuracy, with 22/100 fully correct.
- **The reader is usable on dates and currency, weak on identifiers and
  vendors.** Dates 92.9%, currency 96.0%, but invoice numbers 41.8%, vendors
  58.9%, and tax 64.8% on holdout.
- **Tax is the next target.** It is the only field that moved materially
  between splits; the model often picks a line's VAT or computes VAT from a
  misread subtotal.
- **Output control still matters.** One runaway per 100 documents; one
  transport truncation with no completion reason. A retry/continuation policy
  and a header-first pass are warranted.

## What must not be claimed

- Not a full nine-field invoice benchmark, not a B2B supplier-invoice
  benchmark, not an external leaderboard, and not comparable to the synthetic
  benchmark's key-header numbers.
- No line-item, due-date, flag-quality, or confidence-calibration result.
- The holdout is 100 documents in one country (Portugal, 2019–2021), two
  sources, with human labels that contain brand/legal-entity ambiguity and
  occasional tax noise.
- The 0.3.0 wheel on PyPI does **not** contain the merge fix; 72.0% is a
  measure of the unreleased 0.3.1-dev path.
- Do not quote 72.0% without the document-level interval (14.3–31.4% fully
  correct), and do not present the field interval as a document-level rate.

## Artifacts

Development:
`results/rows_invoice_pt_probe100_1400.jsonl` (0.3.0 merge),
`results/rows_invoice_pt_probe100_1400_mergefix5.jsonl` (fixed merge from the
same raws), matching `scores_*` / `summary_*`, `results/audit_invoice_pt_probe100_1400.json`,
`results/residuals_pt_probe100_1400.json`.

Holdout:
`results/rows_invoice_pt_holdout100_1400.jsonl`,
`results/scores_invoice_pt_holdout100_1400_full.json`,
`results/summary_invoice_pt_holdout100_1400_full.json`,
`results/audit_invoice_pt_holdout100_1400.json`,
`results/residuals_pt_holdout100_1400.json`,
`results/compare_pt_dev_vs_holdout.json`.

## Reproduce

```bash
export PATH="/opt/homebrew/bin:$PATH"
ROOT=/var/folders/6m/l_wd40y91jqbj36ty4nz2skm0000gn/T/opencode/datasets/pt_invoices

# Inference (multi-document run; get approval first)
scripts/run_pt_probe.sh pt_probe100_1400 \
    evaluation/invoice/manifest_pt_real_probe100_1400.jsonl 2048 0
scripts/run_pt_probe.sh pt_holdout100_1400 \
    evaluation/invoice/manifest_pt_real_holdout100_1400.jsonl 2048 0

# Score both (the holdout run already uses the fixed merge)
python3 evaluation/invoice/preds_from_pagewise_rows.py \
    --rows evaluation/invoice/results/rows_invoice_pt_holdout100_1400.jsonl \
    --out evaluation/invoice/results/preds_invoice_pt_holdout100_1400_full.json \
    --raw-out evaluation/invoice/results/preds_invoice_pt_holdout100_1400_raw.json
python3 evaluation/invoice/pt_score.py \
    --manifest evaluation/invoice/manifest_pt_real_holdout100_1400.jsonl \
    --preds evaluation/invoice/results/preds_invoice_pt_holdout100_1400_full.json \
    --rows evaluation/invoice/results/rows_invoice_pt_holdout100_1400.jsonl \
    --scores evaluation/invoice/results/scores_invoice_pt_holdout100_1400_full.json \
    --summary evaluation/invoice/results/summary_invoice_pt_holdout100_1400_full.json
python3 evaluation/invoice/audit_pt_rows.py \
    --manifest evaluation/invoice/manifest_pt_real_holdout100_1400.jsonl \
    --rows evaluation/invoice/results/rows_invoice_pt_holdout100_1400.jsonl \
    --out evaluation/invoice/results/audit_invoice_pt_holdout100_1400.json \
    --allow-failed 2
```
