# TinyDoc engine eval (qwen3-vl:2b-instruct) (2026-11-09)

**Partial run: 10 of 181 documents scored.**

| Measure | Result |
|---|---|
| Documents with every key header field right | 50% (5 of 10) |
| Key header fields right | 90% (81 of 90) |
| Line items, mean F1 | 0.844 (10 documents; all lines right on 1) |
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
| currency | 7 | 10 | 70 |
| subtotal | 8 | 10 | 80 |
| tax | 8 | 10 | 80 |
| total | 8 | 10 | 80 |
| document_type | 10 | 10 | 100 |
| line_items | 1 | 10 | 10 |
| flag | 10 | 10 | 100 |

## By category

| Category | Documents | Fully correct | % |
|---|---|---|---|
| multipage | 10 | 5 | 50 |

## By difficulty

| Difficulty | Documents | Fully correct | % |
|---|---|---|---|
| medium | 4 | 2 | 50 |
| hard | 6 | 3 | 50 |

## By photo/scan severity

| Photo/scan severity | Documents | Fully correct | % |
|---|---|---|---|
