# Total recovery: rejected model retry, validated single-amount fallback

**Status:** the model-based total retry was rejected. The deterministic lone-tax
fallback shipped in 0.3.2 and was validated on a targeted toll/parking stress
set.
**Headline:** on Portuguese toll and parking receipts that print one `Taxa`/fee
amount and no total, the reader often puts that amount in `tax`. Reclassifying it
as `total` recovers **3 fields and 3 fully-correct documents on the seed-44
set** and **3 fields and 3 fully-correct documents on a 28-document targeted
toll/parking stress set**, with **6/6 triggers correct and 0 fields lost**. It
makes no model call and does not guess when more than one amount exists.

## Why the model retry was rejected

The fresh seed-44 run left 7/100 documents without a total: three toll receipts
and one card slip with no printed `Total` line, plus three pages that produced no
JSON at 1400 px (a handwritten taxi invoice, a long IKEA receipt, and a long El
Corte Inglés receipt).

Two bounded retry protocols were measured on those seven documents at 2400 px:

1. **Full-page re-read with the canonical invoice prompt** (7 calls). It produced
   **0/7 correct totals**: the toll receipts and card slip still had no total, the
   taxi invoice returned 2716.0 for 15.60, and the two long receipts returned
   VAT-double-counted totals (59.00 → 70.03; 2.51 → 2.66).
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

- Trigger: `total` is missing/non-positive, `subtotal` is absent, `line_items`
  is empty, and `tax > 0`. The page has exactly one candidate amount, labelled
  `Taxa`/fee, and the reader put it in `tax`. Set `total = tax` and clear `tax`.
- Everything else is left alone. A lone `subtotal` is **not** reclassified:
  the targeted probe found a parking receipt with a 29.40 subtotal against a
  4.20 printed total, so subtotal is ambiguous. When two amounts exist, there is
  no arithmetic guess; the preserve-printed-total rule still applies.

The mutation is recorded in `corrections` (`total_from_single_amount_tax`), in
`review_flags`, and in `InvoiceResult.total_fallback` with the before/after
values. `method` gains `+single_amount_total`. Nothing is silent.

## Seed-44 replay (development evidence)

No model calls. The rule is applied to the saved first-pass predictions and both
sides are scored with `pt_score.py`.

| Metric | First pass | + single-amount fallback | Δ |
|---|---:|---:|---:|
| Field accuracy | 401/570 = **70.4%** | 404/570 = **71.0%** | +3 |
| Total | 68/100 | **71/100** | +3 |
| Fully correct | 19/100 | **22/100** | +3 |
| Every other field | unchanged | unchanged | 0 |

Three documents triggered, all three correct: toll receipts where the printed
`Taxa` (1.35, 1.35, 0.35) was in `tax`. The tax field is not scored on those
documents (gold tax is null), so clearing it costs nothing. No document in the
30-document pilot or the 50-document holdout subset triggers the rule.

Artifacts: `results/total_fallback_replay_pt_retry100.json`, produced by
`analyze_total_fallback.py`.

## Seed-45 fresh sample: no applicable document

A new 100-document set (seed 45, disjoint from seeds 42/43/44, 80 company / 20
personal, 572 scored fields) was frozen before the run and run at 1400 px with
the shipped fallback. It scores **429/572 = 75.0% field accuracy, 29/100 fully
correct, 2 hard failures** (two pages with no parseable JSON). The rule
**triggered 0 times**: every parsed receipt already had a total. Pre- and
post-fallback scores are identical, so this sample neither validates nor refutes
the rule; it shows only that shipping it changed nothing on a representative
fresh set.

## Targeted toll/parking stress probe (28 documents)

The relevant stratum is rare in a random sample, so a targeted stress set was
frozen: the 28 remaining real Portuguese toll, parking, and transport receipts
in the pool, all with no scored tax and disjoint from every earlier set.
Selection uses the vendor annotation only; the fallback rule uses no vendor or
gold values. This is **not** a population sample.

| Metric | First pass | + single-amount fallback | Δ |
|---|---:|---:|---:|
| Field accuracy | 111/139 = **79.9%** | 114/139 = **82.0%** | +3 |
| Fully correct | 4/28 | **7/28** | +3 |
| Triggers | — | **3/28** | 3/3 correct |

All three triggers are Brisa toll receipts whose printed `Taxa` was in `tax`
(0.35, 1.35, 1.35); all three match the printed total. The probe also found the
subtotal counterexample that narrowed the rule: a Bragaparques receipt had
`subtotal = 29.40`, `total = null`, and a printed total of **4.20**. An earlier
version of the rule reclassified the subtotal to 29.40, which was wrong; the
shipped rule now requires an empty `subtotal` and an empty `line_items` list, so
that document is left with a missing total instead of a wrong one.

Artifacts: `manifest_pt_toll28.jsonl`, `results/pt_toll28_selection.json`,
`results/rows_invoice_pt_toll28_1400_narrowed.jsonl`,
`results/total_fallback_replay_pt_toll28_narrowed.json`. The earlier run
`rows_invoice_pt_toll28_1400.jsonl` is kept as the counterexample that narrowed
the rule.

**Residual:** this is 6/6 correct triggers across development and stress sets,
not a guarantee. A single-amount receipt that is genuinely VAT (not a fee) and
has no line items could still be misclassified; the correction is recorded and
flagged so a review queue can catch it.

## Product path

- `InvoiceResult.total_fallback` records the trigger and the before/after values.
- The CLI and `extract()` need no new flag: the fallback is deterministic and
  runs whenever a total is missing and a lone tax amount remains.
- The evaluation runner imports the same function from the SDK, so the measured
  path cannot drift from the product path.
