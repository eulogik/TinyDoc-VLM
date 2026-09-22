# Phase 0 — SROIE receipt field extraction baselines

Eval: 100 held-out SROIE images (`results/sroie_eval.json`), fields: company/date/address/total.

| Engine | n | Field P | Field R | Field F1 | ANLS | Schema-valid | Exact-4/4 | Avg latency ms |
|--------|---|---------|---------|----------|------|--------------|-----------|----------------|
| ollama | 100 | 0.870 | 0.870 | **0.870** | 0.955 | 1.000 | 0.610 | 2573 |
| smolvlm2 | 100 | 0.332 | 0.328 | **0.330** | 0.544 | 0.960 | 0.010 | 9133 |
| ocr_regex | 100 | 0.247 | 0.210 | **0.227** | 0.303 | 0.600 | 0.000 | 1623 |

Notes:
- `field_f1` is micro over (example × field) binary matches with normalize + type-aware rules (money/date).
- company/address match is **punctuation-insensitive** (alnum tokens; same rule as evidence.py) — 2026-09-22 metric fix for OCR comma/period variance. Re-scored from existing preds (`rescore_preds.py`).
- `ocr_regex` is a floor (weak OCR heuristics), not a competitive parser.
- Layout-family split: `results/layout_families.json` + `scores_*_unseen.json` (excluded from this table).
- Ollama engine: unconstrained JSON + salvage (no `format=json` — see engines.py).
- Marked numbers are measured on this machine (MPS/CPU), not copied from papers.
