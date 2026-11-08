# TinyDoc engine eval (qwen3-vl:2b-instruct) (2026-11-08)

**Partial run: 10 of 181 documents scored.**

| Measure | Result |
|---|---|
| Documents with every key header field right | 40% (4 of 10) |
| Key header fields right | 71.1% (64 of 90) |
| Line items, mean F1 | 0.322 (10 documents; all lines right on 0) |
| Invoices whose figures do not add up, flagged | 0 of 0 |
| Files holding several invoices, flagged | 0 of 0 |
| Consistent documents flagged (false alarms) | 0 of 10 |

## Fields

| Field | Right | Scored | % |
|---|---|---|---|
| vendor_name | 8 | 10 | 80 |
| invoice_number | 8 | 10 | 80 |
| invoice_date | 8 | 10 | 80 |
| due_date | 8 | 10 | 80 |
| currency | 6 | 10 | 60 |
| subtotal | 5 | 10 | 50 |
| tax | 6 | 10 | 60 |
| total | 5 | 10 | 50 |
| document_type | 10 | 10 | 100 |
| line_items | 0 | 10 | 0 |
| flag | 10 | 10 | 100 |

## By category

| Category | Documents | Fully correct | % |
|---|---|---|---|
| multipage | 10 | 4 | 40 |

## By difficulty

| Difficulty | Documents | Fully correct | % |
|---|---|---|---|
| medium | 4 | 3 | 75 |
| hard | 6 | 1 | 16.7 |

## By photo/scan severity

| Photo/scan severity | Documents | Fully correct | % |
|---|---|---|---|
