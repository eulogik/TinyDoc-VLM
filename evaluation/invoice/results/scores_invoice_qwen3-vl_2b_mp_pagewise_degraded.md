# TinyDoc engine eval (qwen3-vl:2b-instruct) (2026-11-12)

**Partial run: 8 of 181 documents scored.**

| Measure | Result |
|---|---|
| Documents with every key header field right | 62.5% (5 of 8) |
| Key header fields right | 91.7% (66 of 72) |
| Line items, mean F1 | 0.795 (8 documents; all lines right on 0) |
| Invoices whose figures do not add up, flagged | 0 of 0 |
| Files holding several invoices, flagged | 0 of 0 |
| Consistent documents flagged (false alarms) | 0 of 8 |

## Fields

| Field | Right | Scored | % |
|---|---|---|---|
| vendor_name | 7 | 8 | 87.5 |
| invoice_number | 8 | 8 | 100 |
| invoice_date | 8 | 8 | 100 |
| due_date | 8 | 8 | 100 |
| currency | 7 | 8 | 87.5 |
| subtotal | 7 | 8 | 87.5 |
| tax | 6 | 8 | 75 |
| total | 7 | 8 | 87.5 |
| document_type | 8 | 8 | 100 |
| line_items | 0 | 8 | 0 |
| flag | 8 | 8 | 100 |

## By category

| Category | Documents | Fully correct | % |
|---|---|---|---|
| multipage | 8 | 5 | 62.5 |

## By difficulty

| Difficulty | Documents | Fully correct | % |
|---|---|---|---|
| medium | 3 | 1 | 33.3 |
| hard | 5 | 4 | 80 |

## By photo/scan severity

| Photo/scan severity | Documents | Fully correct | % |
|---|---|---|---|
