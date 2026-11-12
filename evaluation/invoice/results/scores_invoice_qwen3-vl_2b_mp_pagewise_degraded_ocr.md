# TinyDoc engine eval (qwen3-vl:2b-instruct) (2026-11-12)

**Partial run: 8 of 181 documents scored.**

| Measure | Result |
|---|---|
| Documents with every key header field right | 87.5% (7 of 8) |
| Key header fields right | 98.6% (71 of 72) |
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
| currency | 8 | 8 | 100 |
| subtotal | 8 | 8 | 100 |
| tax | 8 | 8 | 100 |
| total | 8 | 8 | 100 |
| document_type | 8 | 8 | 100 |
| line_items | 0 | 8 | 0 |
| flag | 8 | 8 | 100 |

## By category

| Category | Documents | Fully correct | % |
|---|---|---|---|
| multipage | 8 | 7 | 87.5 |

## By difficulty

| Difficulty | Documents | Fully correct | % |
|---|---|---|---|
| medium | 3 | 3 | 100 |
| hard | 5 | 4 | 80 |

## By photo/scan severity

| Photo/scan severity | Documents | Fully correct | % |
|---|---|---|---|
