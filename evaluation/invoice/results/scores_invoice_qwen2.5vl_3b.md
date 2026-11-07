# TinyDoc engine eval (qwen2.5vl:3b) (2026-11-06)

| Measure | Result |
|---|---|
| Documents with every key header field right | 65.7% (119 of 181) |
| Key header fields right | 91.9% (1448 of 1576) |
| Line items, mean F1 | 0.929 (179 documents; all lines right on 158) |
| Invoices whose figures do not add up, flagged | 0 of 5 |
| Files holding several invoices, flagged | 0 of 2 |
| Consistent documents flagged (false alarms) | 2 of 172 |

## Fields

| Field | Right | Scored | % |
|---|---|---|---|
| vendor_name | 163 | 181 | 90.1 |
| invoice_number | 167 | 179 | 93.3 |
| invoice_date | 170 | 179 | 95 |
| due_date | 135 | 140 | 96.4 |
| currency | 170 | 179 | 95 |
| subtotal | 151 | 179 | 84.4 |
| tax | 156 | 179 | 87.2 |
| total | 157 | 179 | 87.7 |
| document_type | 179 | 181 | 98.9 |
| line_items | 158 | 179 | 88.3 |
| flag | 170 | 179 | 95 |
| references | 1 | 25 | 4 |

## By category

| Category | Documents | Fully correct | % |
|---|---|---|---|
| clean | 31 | 23 | 74.2 |
| currency | 19 | 10 | 52.6 |
| tax | 14 | 7 | 50 |
| credit | 10 | 8 | 80 |
| multipage | 10 | 0 | 0 |
| freight | 16 | 7 | 43.8 |
| scan | 15 | 15 | 100 |
| photo | 45 | 39 | 86.7 |
| handwritten | 8 | 6 | 75 |
| hard | 13 | 4 | 30.8 |

## By difficulty

| Difficulty | Documents | Fully correct | % |
|---|---|---|---|
| easy | 62 | 53 | 85.5 |
| medium | 48 | 31 | 64.6 |
| hard | 71 | 35 | 49.3 |

## By photo/scan severity

| Photo/scan severity | Documents | Fully correct | % |
|---|---|---|---|
| mild | 20 | 20 | 100 |
| moderate | 20 | 18 | 90 |
| severe | 20 | 16 | 80 |
