# Archived runs

`rows_invoice_qwen3-vl_2b_mp_num768_truncated.jsonl` — first multipage pass
(2026-11-08) with the page-1 default `num_predict=768`. Every long document was
cut off mid-JSON (~2,000 characters); the 6-, 5-, and 3-page rows failed to
parse. This is **not a result**: the output budget was too small for 98–117 line
items, not a model failure. The rerun uses `num_predict=5120` and is the only
multipage evidence that may be cited.
