# Prompt pilot — receipt rules (negative result, 2026-11-12)

**Status:** rejected. This is a 30-document development pilot, not a public
result and not a held-out estimate. The canonical v1 prompt remains the default.

## Question

The real Portuguese holdout leaves three weak fields: tax (64.8%), invoice
number (41.8%), and vendor (58.9%). Can a receipt-aware prompt improve them
without changing the model?

## Design

- 30 development documents: 24 company, 6 personal, seed 7, 174 scored fields.
- v1 baseline: predictions filtered from the existing 100-document development
  run; no new inference.
- v3: header-first prompt. No line items; adds `prices_include_tax`, the
  tax-inclusive total rule, Portuguese dot-decimal, identifier and vendor rules.
- v4: v1's proven output shape (including line items) plus the same receipt
  rules, as focused addenda.
- Same model (`qwen3-vl:2b-instruct`), 1400 px, 8192 context, 2048-token cap.
- Paired on the same 30 documents.

## Result

| Prompt | Field accuracy | Fully correct | All six | p50 | Cap hits |
|---|---:|---:|---:|---:|---:|
| v1 (canonical) | 133/174 = **76.4%** | 7/30 | 6/24 | 8.3 s | 0/30 |
| v3 (header-first) | 138/174 = **79.3%** | 7/30 | 6/24 | 8.4 s | **2/30** |
| v4 (v1 + rules) | 131/174 = **75.3%** | 6/30 | 5/24 | 5.4 s | 0/30 |

Paired field counts:

- v1 → v3: 125 both, 8 v1-only, 13 v3-only (+5 fields); documents 2 v3-only,
  2 v1-only.
- v1 → v4: 125 both, 8 v1-only, 6 v4-only (−2 fields); documents 1 v4-only,
  2 v1-only.

| Field | v1 | v3 | v4 |
|---|---:|---:|---:|
| Currency | 28/30 | 30/30 | 29/30 |
| Invoice date | 30/30 | 30/30 | 30/30 |
| Invoice number | 15/30 | 16/30 | 15/30 |
| Tax | 21/25 | 21/25 | 19/25 |
| Total | 19/30 | 21/30 | 17/30 |
| Vendor | 20/29 | 20/29 | 21/29 |

## Why v3 is rejected

- The +5 fields (+2.9 points) is within noise at n=30 and does not raise the
  fully-correct count.
- **2/30 responses repeated the JSON object until the 2048-token cap** (56.5 s
  and 45.9 s). `_extract_json` salvaged the first object, but the loop rate is
  worse than v1's single runaway in 100 documents. A prompt that trades a small
  accuracy gain for a 6.7% cap-hit rate is not shippable.
- The loop is a new failure mode: header-only output gives the model too little
  structure to stop. A retry or strict single-object constraint might fix it,
  but that is a separate experiment.

## Why v4 is rejected

- The receipt rules alone did not help: −2 fields, −1 fully-correct document.
  Tax and total each regressed by 2, while currency and vendor each gained 1.
- It is faster (5.4 s p50), but latency is not the binding constraint.
- The rules may distract the model from the printed values, or the 30-document
  pilot is simply too small to resolve a 1–3 point effect.

## Conclusion

**Keep prompt v1.** No prompt change is justified by this pilot. The next levers
are structural, not wording:

1. A second OCR pass over the printed total/tax block (`with_ocr_totals`) to
   correct tax-inclusive and multi-rate receipts.
2. A higher-resolution or cropped input for long invoice numbers.
3. A retry for the no-`done_reason` transport truncation.
4. A bounded second pass for line items, so the header pass cannot run away.

A future prompt candidate needs a materially larger gain before it earns a full
100-document development run and a new holdout.

## Artifacts

- `manifest_pt_pilot30_1400.jsonl`
- `results/rows_invoice_pt_pilot30_v1.jsonl` (filtered development rows)
- `results/rows_invoice_pt_pilot30_v3.jsonl`, `..._v4.jsonl`
- `results/scores_invoice_pt_pilot30_v1.json`, `..._v3.json`, `..._v4.json`
- `results/summary_invoice_pt_pilot30_*.json`
- `results/compare_pt_pilot30_v1_vs_v3.json`, `..._v1_vs_v4.json`
