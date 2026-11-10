# TinyDoc engine eval (qwen3-vl:2b-instruct) (2026-11-10)

**Partial run: 10 of 181 documents scored.**

| Measure | Result |
|---|---|
| Documents with every key header field right | 80% (8 of 10) |
| Key header fields right | 95.6% (86 of 90) |
| Line items, mean F1 | 0.848 (10 documents; all lines right on 1) |
| Invoices whose figures do not add up, flagged | 0 of 0 |
| Files holding several invoices, flagged | 0 of 0 |
| Consistent documents flagged (false alarms) | 0 of 10 |

## Fields

| Field | Right | Scored | % |
|---|---|---|---|
| vendor_name | 10 | 10 | 100 |
| invoice_number | 10 | 10 | 100 |
| invoice_date | 10 | 10 | 100 |
| due_date | 10 | 10 | 100 |
| currency | 9 | 10 | 90 |
| subtotal | 9 | 10 | 90 |
| tax | 9 | 10 | 90 |
| total | 9 | 10 | 90 |
| document_type | 10 | 10 | 100 |
| line_items | 1 | 10 | 10 |
| flag | 10 | 10 | 100 |

## By category

| Category | Documents | Fully correct | % |
|---|---|---|---|
| multipage | 10 | 8 | 80 |

## By difficulty

| Difficulty | Documents | Fully correct | % |
|---|---|---|---|
| medium | 4 | 2 | 50 |
| hard | 6 | 6 | 100 |

## By photo/scan severity

| Photo/scan severity | Documents | Fully correct | % |
|---|---|---|---|
