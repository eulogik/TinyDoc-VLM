# Resolution pilot — 1400 px vs 2400 px (promising, not shippable default)

**Status:** 30-document development pilot. The 2400 px input improves the weak
fields but is ~4× slower and memory-heavy. It is a candidate for an optional
high-resolution retry, not a new default.

## Design

- Same 30 development documents as the prompt pilot (24 company, 6 personal,
  seed 7, 174 scored fields).
- Same model (`qwen3-vl:2b-instruct`), prompt v1, 8192 context, 2048-token cap.
- Only the input changes: long side 1400 px vs 2400 px (both JPEG q92,
  prepared with `prepare_pt_probe.py`).
- Paired on the same documents.

## Result

| Input | Field accuracy | Fully correct | All six | p50 | p90 |
|---|---:|---:|---:|---:|---:|
| 1400 px (product cap) | 133/174 = **76.4%** | 7/30 | 6/24 | 8.3 s | 9.1 s |
| 2400 px | 141/174 = **81.0%** | **11/30** | 9/24 | **33.8 s** | 38.2 s |

Paired field counts: 130 both, 3 only-1400, 11 only-2400 (+8 fields).
Paired documents: 7 both, **0 only-1400**, **4 only-2400**.

| Field | 1400 px | 2400 px | Δ |
|---|---:|---:|---:|
| Currency | 28/30 | 29/30 | +1 |
| Invoice date | 30/30 | 30/30 | 0 |
| Invoice number | 15/30 | **18/30** | +3 |
| Tax | 21/25 | 21/25 | 0 |
| Total | 19/30 | **21/30** | +2 |
| Vendor | 20/29 | **22/29** | +2 |

## What improved

- Three more invoice numbers, including long identifiers with leading zeros and
  slashes.
- Two more vendor names: the higher resolution favors the printed brand over a
  small-print legal entity (`Intermarché` instead of `DISTRIBUIDORA
  SUPERMERCADOS LDA`).
- Two more totals and one more currency.
- Four more fully-correct documents; no document lost full correctness.

The gains are concentrated exactly in the fields the 1400 px holdout showed as
weak (number 41.8%, vendor 58.9%). That is a coherent mechanism, not noise.

## Costs and regressions

- **4.1× slower** (p50 33.8 s vs 8.3 s) and materially more memory; the run
  pushed swap but completed with the watchdog.
- One document returned valid JSON with no total (`stop`, 116 output tokens),
  so 29/30 parsed for scoring.
- Some individual regressions: `20210322_163546` fixed the vendor brand but
  read the total as 47.72 instead of 1.72; `20210428_183109` changed a correct
  total to 12.80. The net is positive but not monotonic.
- n=30 is underpowered: +4.6 field points has a wide interval, and the
  fully-correct gain is 4 documents.

## Held-out 50-document check

The 2400 effect was re-tested on a fixed **50-document subset of the seed-43
holdout** (40 company, 10 personal, seed 99, 289 scored fields). The 1400 px
predictions already existed; only the 2400 px run was new.

| Input | Field accuracy | Fully correct | All six | p50 |
|---|---:|---:|---:|---:|
| 1400 px | 211/289 = **73.0%** | 11/50 | 8/40 | 8.4 s |
| 2400 px | 221/289 = **76.5%** | 12/50 | 11/40 | 36.6 s |
| Field hybrid (1400 total/vendor + 2400 other) | 226/289 = **78.2%** | 13/50 | 11/40 | both runs |

2400 vs 1400: **+3.46 field points**, +1 fully-correct document, +3 all-six;
23 fields gained, 13 lost. The field pattern is not uniform:

| Field | 1400 | 2400 | Hybrid |
|---|---:|---:|---:|
| Currency | 49/50 | 50/50 | 50/50 |
| Invoice date | 44/48 | 45/48 | 45/48 |
| Invoice number | 19/48 | **24/48** | 24/48 |
| Tax | 30/46 | **38/46** | 38/46 |
| Total | **41/50** | 39/50 | 41/50 |
| Vendor | **28/47** | 25/47 | 28/47 |

**Tax +8 and invoice number +5 are real; total −2 and vendor −3 are the cost.**
The hybrid is a diagnostic upper bound for a two-resolution ensemble, not a
shippable default: it requires both runs (~5.4× latency).

## Conclusion

**Do not make 2400 px the default.** The held-out 50-document check confirms a
modest net gain (+3.5 field points, +1 fully-correct document) at ~4.4× latency,
with tax and identifiers improving while totals and vendors regress.

The promising product shape is an **optional high-resolution retry targeted at
tax and invoice numbers**, not a whole-document re-run: try 1400 px first, then
retry at 2400 px only when the tax or identifier path is weak. A fixed
field-level hybrid reaches 78.2% but needs both runs. A real retry needs a
reliable trigger; invoice-field confidence is not calibrated, so use a
structural trigger for now (empty/short identifier, repeated digits, tax missing
or implausible). Validate that trigger on a fresh set before shipping.

## Artifacts

- `manifest_pt_pilot30_2400.jsonl`
- `results/rows_invoice_pt_pilot30_2400_v1.jsonl`
- `results/scores_invoice_pt_pilot30_2400_v1.json`
- `results/summary_invoice_pt_pilot30_2400_v1.json`
- `results/compare_pt_pilot30_1400_vs_2400.json`
- Prep stats: `results/pt_real_probe_prep_2400.json`
- Holdout-50: `manifest_pt_holdout50_2400.jsonl`,
  `results/scores_invoice_pt_holdout50_1400.json`,
  `results/scores_invoice_pt_holdout50_2400_v1.json`,
  `results/scores_invoice_pt_holdout50_hybrid.json`,
  `results/compare_pt_holdout50_1400_vs_2400.json`
