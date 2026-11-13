# Targeted high-resolution tax retry

**Status:** validated on a fresh 100-document set (seed 44) after development
replay. See "Fresh validation" below.
**Headline:** the retry is a structural, opt-in correction. It replays at
+10 field points on tax (30/46 → 40/46) on the 50-document subset with no
field lost, at the cost of one extra 2400 px page read on 15/50 documents.
It is not part of the published held-out headline (72.0% / 22-of-100, seed 43)
until the fresh validation lands.

## Why this lever

The 2400 px whole-document pilot and the held-out 50-document check
([PT_RESOLUTION_PILOT.md](PT_RESOLUTION_PILOT.md)) showed that higher
resolution helps tax (+8 on the 50 subset) and invoice number (+5) but costs
4.4× latency (p50 36.6 s vs 8.4 s) and regresses total (−2) and vendor (−3).
Running every document at 2400 px is not a good default.

The tax errors on Portuguese receipts are not random: they are
**tax-inclusive** receipts where the model returns the gross amount, the VAT
rate (23.0), or a truncated digit instead of the printed IVA amount. Those
cases are structurally detectable in the first-pass prediction, and the
printed IVA is small, labelled, and readable at 2400 px. A targeted retry can
fix exactly those cases without touching totals or vendors.

## Trigger (structural, no gold, no country lookup)

`tax_retry_reason` fires on the first-pass merged prediction when the printed
tax is implausible:

- `tax` is missing or non-positive;
- `tax >= 0.5 * total` (the gross amount or a VAT rate, not a VAT amount);
- `tax < 0.01 * total` (a truncated leading digit, e.g. 0.1 for 3.72).

Everything else is left alone. The retry is a single read of the final page at
`retry_px` (default 2400). Only `tax` is replaced; totals, vendors, and all
other fields stay at 1400 px. A retry that fails or is declined leaves the
first-pass value untouched, so the retry can never fail an extraction.

## Acceptance (structural prior)

The candidate is accepted only when it looks like a VAT portion of the printed
total: `0 < tax < total` and the implied rate `tax / (total - tax) * 100`
is between 1.0% and 27.5%. The upper bound is the highest standard VAT rate in
the EU; the lower bound rejects zero and noise. This rejects the 2400 px
regressions on the 50 subset (for example 17.2 read as 1000% of the base, or
1535.0 read as 432%) while accepting the genuine fixes (14.37 at 23%, 0.43 at
13%).

## Replay on saved predictions (development evidence)

No model calls. The replay re-applies the trigger to the saved 1400 px
predictions, takes the candidate from the saved 2400 px predictions, and scores
both with `pt_score.py`. The rule was designed on these same documents, so this
is development evidence, not a held-out result.

| Set | Triggered | Applied | Tax before | Tax after | Field before | Field after | Fully before → after | Lost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 30-document pilot | 6/30 | 1/6 | 21/25 | 22/25 | 133/174 | 134/174 | 7 → 7 | 0 |
| 50-document holdout subset | 15/50 | 11/15 | 30/46 | 40/46 | 211/289 | 221/289 | 11 → 13 | 0 |

Per field, the retry changes tax only: every other field is identical before
and after. On the 50 subset the retry also improves all-six from 8/40 to
10/40. The 10 applied fixes are all tax-inclusive receipts where the first
pass reported the gross amount (9.51 → 1.78), the VAT rate (23.0 → 1.87), or
a truncated digit (0.1 → 3.72). The 4 declined cases are 2400 px reads that
are structurally impossible (17.2 for a 18.92 total, 1535.0 for a 1890.0
total) and are correctly rejected.

Artifacts: `results/retry_replay_pt_pilot30.json`,
`results/retry_replay_pt_holdout50.json`, produced by
`analyze_tax_retry.py`. Both are checked by `recompute_scores.py`.

## Product path

- `InvoicePipeline.extract(..., retry_tax_px=0)` (opt-in; 0 disables). Set it to
  `2400` to enable the retry. The CLI exposes `--retry-tax-px PX` (default 0).
- `tax_retry_reason` and `retry_tax_is_plausible` are exported from
  `tinydoc.invoice` and are the single source of truth for both the SDK and the
  evaluation runner.
- `InvoiceResult.retry` records the trigger, candidate, accepted value, and the
  raw retry response. `InvoiceResult.method` gains `+high_res_tax_retry` only
  when a value was actually applied.
- The retry re-renders only the final page (PDFs: `render_pdf_pages(...,
  page_numbers=[last])`), so the cost is one extra 2400 px read per triggered
  document, not per page.
- The OCR total reader (`ocr_totals.py`) still runs first; the retry only fires
  when OCR did not already produce a consistent block.

## Fresh validation (seed 44, 100 documents)

A fresh 100-document set disjoint from both the development probe and the
seed-43 holdout (80 company / 20 personal; 570 scored fields; seed 44). The
first pass and the retry were captured in one run; the pre-retry predictions
were rebuilt with `preds_from_pagewise_rows.py --pred-key pred_first_pass`.

| Metric | First pass (1400 px) | + targeted 2400 tax retry | Δ |
|---|---:|---:|---:|
| Field accuracy | 401/570 = **70.4%** | 411/570 = **72.1%** | +10 fields (+1.8 pp) |
| Fully correct | 19/100 | **21/100** | +2 |
| All six | 12/74 | **13/74** | +1 |
| Tax | 51/82 = 62.2% | **61/82 = 74.4%** | **+10** |

Per field, only tax moves: currency 94/100, date 91/98, invoice number 43/96,
total 68/100, vendor 54/94 are identical before and after (paired comparison:
401 both, 0 first-pass-only, 10 retry-only, 159 neither). 23/100 documents
triggered (12 applied); 0 retry errors. Fully-correct documents went from 19/100
to 21/100 with no regression.

The fresh numbers match the development replay almost exactly (development:
+10 tax, +10 field, +2 fully correct, +1 all-six, 0 lost), so the trigger and
the 1.0–27.5% acceptance rule are not overfit to the 50-document subset. The
fresh set is harder overall (70.4% first pass vs 73.0% on the subset), so the
absolute lift is the right quantity to compare, not the level.

Hard failures on this set: 7/100 (3 pages with no parseable JSON, 4 pages with
no total), matching the 2–7% failure rate seen on the earlier real-document
runs. None of those 7 were retry failures.

## Decision

The retry is validated and is the chosen structural lever for 0.3.1: it adds
~1.8 field points and 2 fully-correct documents at the cost of one 2400 px read
on ~23% of documents (~1.3× average latency), with no field regression and a
deterministic acceptance rule. It does not change totals, vendors, or
identifiers, so the existing real-document headline (72.0% / 22-of-100 on the
seed-43 holdout, 1400 px) remains the comparable baseline; the retry number is
reported separately. The retry should ship opt-in behind `--retry-tax-px` /
`retry_tax_px`, with the product default decided at release time.
