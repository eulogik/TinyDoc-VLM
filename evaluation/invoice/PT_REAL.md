# Real Portuguese invoice/receipt probe (field subset)

**Status:** source downloaded, checksum-verified, deduplicated, and frozen into a
100-document development probe and a disjoint 100-document holdout. Both have
been measured once. Results: [PT_REAL_RESULTS.md](PT_REAL_RESULTS.md). Any number
produced from it is a real-world transfer probe, not an external leaderboard
result and not a full nine-field invoice benchmark.

## Source

| | |
|---|---|
| Documents | 1,003 phone photos (813 company + 190 personal), 2019–2021 |
| Label source | Human transcription, shipped with the dataset |
| License | CC BY 4.0 |
| Company set | [Zenodo 6371710](https://doi.org/10.5281/zenodo.6371710) — invoices/receipts of a private Portuguese company |
| Personal set | [Zenodo 7213544](https://doi.org/10.5281/zenodo.7213544) — one author's personal invoices/receipts |
| Authors | Francisco Cruz, Mauro Castelli |
| Images | 3000×4000 JPEG phone photos, mostly thermal receipts and simplified invoices |

The archives are checked against the MD5 values published by Zenodo before
extraction. The builder is `evaluation/invoice/build_pt_manifest.py`; images are
**not** committed.

## Why this probe exists

Every invoice number in the repository before this probe came from a synthetic,
vendor-made corpus. This is the first real-document evidence path: real
Portuguese receipts and invoices, real paper, real phone photos, human labels.
It is deliberately narrow: five fields, one country, two sources. It cannot
replace a real supplier-invoice benchmark.

## Field mapping

| TinyDoc field | Gold field | Probe coverage (n=100) |
|---|---|---:|
| `vendor_name` | `company` | 95 |
| `invoice_number` | `invoice_number` | 98 |
| `invoice_date` | `date` (normalized to ISO, day-first) | 97 |
| `currency` | fixed `EUR` | 100 |
| `tax` | `iva_amount` | 87 |
| `total` | `total` | 100 |

**Not available, therefore not scored:** `due_date`, `subtotal`, `line_items`,
`document_type`, review flags. The corpus has no line-item labels and no
due-date column.

## Cleaning rules

- **Amounts.** The transcriptions mix `1.234,56`, `6.950` (six euros
  ninety-five), and `13,00`. A single separator is treated as decimal; with both,
  the last separator is decimal. This matches the printed receipts in the set.
- **Tax.** `iva_amount` is scored only when `0 <= tax < total` and
  `tax / total <= 0.35`. Fourteen rows in the full set were excluded rather than
  corrected; a few are clearly transcription errors (for example IVA larger than
  the total).
- **Missing fields.** A blank or unparseable gold value is left out of
  `expected.scored` for that document; it is never filled with a guess.
- **Duplicates.** 1,003 photos collapse to **943 unique documents** after
  deduplication on `(vendor, invoice number, date, total)`, keeping the copy with
  the most scorable fields. Sixty repeat photos are listed in
  `evaluation/invoice/results/pt_real_dataset_stats.json`.
- **Privacy.** The public manifest carries only the five scored fields. Buyer and
  seller tax IDs and addresses from the upstream annotations are not copied.

## Frozen probe

`evaluation/invoice/manifest_pt_real_probe100.jsonl`

- 100 documents, seed 42, proportional stratified sample: **80 company / 20
  personal**.
- 80 documents have all six fields available; the rest are scored on the fields
  their labels provide.
- The sample was fixed before any model run.

## Protocol

The baseline run used the canonical page prompt (v1), one page per document,
`qwen3-vl:2b-instruct`, long side 1400 px (the product cap), 8192 context, and a
2048-token output cap. The scorer uses the benchmark's MIT rules with Portuguese
legal-form stopwords. The merge was measured twice from the same saved raws:
once with the shipped 0.3.0 merge and once with the preserve-printed-total fix.

The seed-43 holdout (`manifest_pt_real_holdout100.jsonl`) is disjoint from the
development 100 and was run once with the fixed merge. Its result is the
held-out estimate: **72.0% field, 22/100 fully correct**. Do not tune on it
further without labeling a new experiment post-hoc; a new prompt needs a new
holdout.

Label it **"real Portuguese receipt/invoice field-subset probe"** in every
public surface. Never merge it into the synthetic benchmark's key-header
number.

## Limitations

- Portuguese, 2019–2021, two sources, one company plus one individual.
- Mostly receipts and simplified invoices; not representative of B2B supplier
  invoices or other countries.
- Human labels contain noise; implausible tax values are excluded, not repaired.
- Phone photos have dark backgrounds, perspective, glare, and curled thermal
  paper. The product's 1400 px cap may be the limiting factor.
- Each split is n=100, giving a 95% interval of roughly ±6–8 points around a
  90% field accuracy; do not quote a point estimate without the interval.
- This probe says nothing about line items, due dates, flags, confidence
  calibration, or grounding.

## Reproduce

```bash
export TINYDOC_PT_ROOT=/var/folders/6m/l_wd40y91jqbj36ty4nz2skm0000gn/T/opencode/datasets/pt_invoices

# Download, verify MD5, extract (about 525 MB)
python3 evaluation/invoice/build_pt_manifest.py --download \
    --root "$TINYDOC_PT_ROOT" \
    --out evaluation/invoice/manifest_pt_real.jsonl \
    --probe evaluation/invoice/manifest_pt_real_probe100.jsonl \
    --stats evaluation/invoice/results/pt_real_dataset_stats.json
```

The builder is deterministic: the same archives produce the same 943 rows and
the same seed-42 probe.
