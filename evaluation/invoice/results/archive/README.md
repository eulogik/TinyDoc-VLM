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
