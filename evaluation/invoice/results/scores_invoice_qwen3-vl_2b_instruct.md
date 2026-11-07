# TinyDoc engine eval (qwen3-vl:2b-instruct) (2026-11-07)

| Measure | Result |
|---|---|
| Documents with every key header field right | 71.8% (130 of 181) |
| Key header fields right | 93.5% (1474 of 1576) |
| Line items, mean F1 | 0.904 (179 documents; all lines right on 151) |
| Invoices whose figures do not add up, flagged | 0 of 5 |
| Files holding several invoices, flagged | 0 of 2 |
| Consistent documents flagged (false alarms) | 0 of 172 |

## Fields

| Field | Right | Scored | % |
|---|---|---|---|
| vendor_name | 178 | 181 | 98.3 |
| invoice_number | 170 | 179 | 95 |
| invoice_date | 175 | 179 | 97.8 |
| due_date | 139 | 140 | 99.3 |
| currency | 161 | 179 | 89.9 |
| subtotal | 153 | 179 | 85.5 |
| tax | 160 | 179 | 89.4 |
| total | 159 | 179 | 88.8 |
| document_type | 179 | 181 | 98.9 |
| line_items | 151 | 179 | 84.4 |
| flag | 172 | 179 | 96.1 |
| references | 20 | 25 | 80 |

## By category

| Category | Documents | Fully correct | % |
|---|---|---|---|
| clean | 31 | 27 | 87.1 |
| currency | 19 | 13 | 68.4 |
| tax | 14 | 8 | 57.1 |
| credit | 10 | 10 | 100 |
| multipage | 10 | 0 | 0 |
| freight | 16 | 9 | 56.2 |
| scan | 15 | 15 | 100 |
| photo | 45 | 38 | 84.4 |
| handwritten | 8 | 3 | 37.5 |
| hard | 13 | 7 | 53.8 |

## By difficulty

| Difficulty | Documents | Fully correct | % |
|---|---|---|---|
| easy | 62 | 56 | 90.3 |
| medium | 48 | 36 | 75 |
| hard | 71 | 38 | 53.5 |

## By photo/scan severity

| Photo/scan severity | Documents | Fully correct | % |
|---|---|---|---|
| mild | 20 | 19 | 95 |
| moderate | 20 | 19 | 95 |
| severe | 20 | 15 | 75 |
