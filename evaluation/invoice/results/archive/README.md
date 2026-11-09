# Archived multipage attempts

These are **not results**. They document why the output budget and failure
accounting changed.

## `rows_invoice_qwen3-vl_2b_mp_num768_truncated.jsonl`

First multipage pass (2026-11-08) with the page-1 default `num_predict=768`.
Every long document was cut off mid-JSON (~2,000 characters); the 3-, 5-, and
6-page rows failed to parse. The output budget was too small for 98–117 line
items, not a model failure.

## 8-page retry (`../rows_invoice_qwen3-vl_2b_mp_8p.jsonl`)

Two 8-page invoices with 110–117 expected line items were retried with
`num_ctx=24576` and `num_predict=8192`. Both still stopped at the output limit,
but the cause was **runaway generation**: the raw output contains 213 and 212
descriptions with fabricated repeated “(batch N)” items, not a truncated valid
JSON. A larger budget would make it worse. These rows remain visible failures in
the canonical 10-document result; they were not merged or hidden.

## Superseded page-wise merge (2026-11-09)

The first page-wise pass wrote `*_pagewise_raw.*` with the initial merge. It was
valid inference but the merge used printed totals without the disclosed
internal-consistency repairs. The canonical `*_pagewise.*` artifacts were rebuilt
from the saved page raws with `rebuild_pagewise.py` (no model calls) and score
**6/10 fully correct, 84/90 key-header, 0.848 line F1**. The raw merge scored
**5/10, 82/90, 0.848**. The difference is the subtotal/tax repairs documented in
provenance; the raw artifacts stay for audit and must not be cited as the
canonical result.
