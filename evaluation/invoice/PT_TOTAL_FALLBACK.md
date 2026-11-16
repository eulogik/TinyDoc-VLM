# Total recovery: rejected model retry, shipped single-amount fallback

**Status:** the model-based total retry was rejected by a 7-document development
probe. A deterministic single-amount fallback is implemented and replayed on the
seed-44 fresh set; its seed-45 validation is in progress.
**Headline:** when a receipt prints exactly one amount and the reader puts it in
`tax` or `subtotal` instead of `total`, reclassify that amount as the total. On
the seed-44 set this recovers **4 fields and 4 fully-correct documents** with no
field lost. It makes no model call and does not guess when two amounts exist.

## Why the model retry was rejected

The fresh seed-44 run left 7/100 documents without a total: three toll receipts
and one card slip with no printed `Total` line, plus three pages that produced no
JSON at 1400 px (a handwritten taxi invoice, a long IKEA receipt, and a long El
Corte Inglés receipt).

Two bounded retry protocols were measured on those seven documents at 2400 px:

1. **Full-page re-read with the canonical invoice prompt** (7 calls). It produced
   **0/7 correct totals**: the three toll receipts still had no total, the card
   slip still had no total, the taxi invoice returned 2716.0 for 15.60, and the
   two long receipts returned VAT-double-counted totals (59.00 → 70.03;
   2.51 → 2.66).
2. **Total-only prompt** (7 calls): *"Return only the final amount the customer
   paid... Prefer Total, Total a pagar, Valor, Taxa, Compra..."*. It recovered
   4.37, 59.00, and 2.51 correctly, but returned **75,000.00** for all three toll
   receipts (the capital-social line) and **1560** for the handwritten 15,60.
   That is 3/7 correct and three dangerous magnitudes; it cannot be accepted
   without an independent anchor, and no anchor exists on those pages.

Artifacts: `results/total_prompt_probe_fail7.json` and
`results/rows_invoice_pt_fail7_2400.jsonl` (7 documents, 2400 px full-page read).
This is a development probe on the seed-44 failures, not a held-out result.

**Decision:** do not ship a model total retry. Higher resolution does not create
a printed total line, and a second full extraction re-derives totals with the
same VAT-inclusive confusion the preserve-printed-total merge fix removed.

## Deterministic fallback (shipped)

`apply_single_amount_total_fallback` in `tinydoc.invoice` runs after the optional
tax retry and after OCR reconciliation, and only when the final page parsed.

- If `total` is missing/non-positive, `subtotal` is absent, and `tax > 0`: the
  single printed amount was labelled `Taxa`/fee and mapped to tax. Set
  `total = tax` and clear `tax`.
- If `total` is missing/non-positive, `tax` is absent, and `subtotal > 0`: the
  single printed amount was labelled `Compra`/charged and mapped to subtotal. Set
  `total = subtotal`.
- If both `subtotal` and `tax` are present, or neither is, do nothing. There is
  no arithmetic guess between two amounts; the preserve-printed-total rule still
  applies.

The mutation is recorded in `corrections` (`total_from_single_amount_tax` or
`total_from_single_amount_subtotal`), in `review_flags`, and in
`InvoiceResult.total_fallback` with the before/after values. `method` gains
`+single_amount_total`. Nothing is silent.

## Seed-44 replay (development evidence)

No model calls. The rule is applied to the saved first-pass predictions and both
sides are scored with `pt_score.py`.

| Metric | First pass | + single-amount fallback | Δ |
|---|---:|---:|---:|
| Field accuracy | 401/570 = **70.4%** | 405/570 = **71.1%** | +4 |
| Total | 68/100 | **72/100** | +4 |
| Fully correct | 19/100 | **23/100** | +4 |
| Every other field | unchanged | unchanged | 0 |

Four documents triggered, all four correct: three toll receipts where the printed
`Taxa` (1.35, 1.35, 0.35) was in `tax`, and one card slip where the printed
`Compra` (4.37) was in `subtotal`. The tax field is not scored on those documents
(gold tax is null), so clearing it costs nothing. No document in the 30-document
pilot or the 50-document holdout subset triggers the rule.

Artifacts: `results/total_fallback_replay_pt_retry100.json`, produced by
`analyze_total_fallback.py`.

## Fresh validation (seed 45, 100 documents)

The rule was found on the seed-44 failures, so the seed-44 replay is development
evidence. A new 100-document set (seed 45, disjoint from seeds 42/43/44, 80
company / 20 personal, 572 scored fields) was frozen before the run. It is run at
1400 px with the shipped fallback; the pre-fallback predictions are reconstructed
from each row's `total_fallback` block and scored separately. Results and the
0.3.2 scope decision will be appended when the run completes and is audited.

## Product path

- `InvoiceResult.total_fallback` records the trigger and the before/after values.
- The CLI and `extract()` need no new flag: the fallback is deterministic and
  runs whenever a total is missing and exactly one printed amount remains.
- The evaluation runner imports the same function from the SDK, so the measured
  path cannot drift from the product path.
