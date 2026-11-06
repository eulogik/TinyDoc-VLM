# TinyDoc engine eval (qwen2.5vl:3b) (2026-11-06)

| Measure | Result |
|---|---|
| Documents with every key header field right | 55.8% (101 of 181) |
| Key header fields right | 90.6% (1428 of 1576) |
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
| currency | 150 | 179 | 83.8 |
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
| scan | 15 | 11 | 73.3 |
| photo | 45 | 26 | 57.8 |
| handwritten | 8 | 5 | 62.5 |
| hard | 13 | 4 | 30.8 |

## By difficulty

| Difficulty | Documents | Fully correct | % |
|---|---|---|---|
| easy | 62 | 47 | 75.8 |
| medium | 48 | 26 | 54.2 |
| hard | 71 | 28 | 39.4 |

## By photo/scan severity

| Photo/scan severity | Documents | Fully correct | % |
|---|---|---|---|
| mild | 20 | 14 | 70 |
| moderate | 20 | 13 | 65 |
| severe | 20 | 10 | 50 |
