# TinyDoc — local-first grounded document extraction

**Status (2026-09-23):** Phase 1 product spine. Free `ollama:qwen2.5vl:3b` · SROIE field F1 **0.870** (n=100 holdout, `ReceiptPipeline` e2e + engine-only, prompt-v1) · schema-valid **1.000** · unseen-layout F1 **0.955** (n=11). Evidence-on: same F1, mean conf **0.694**, coverage **0.668**. Second vertical FUNSD n=50: F1 **0.352**, schema **0.880** (different task; measured, not tuned). Address residual **0.72** (postprocess tried; 0.80 **not** claimed). Unit tests: `tests/test_pipeline.py` 30 passed. Local demo: `demo/app.py --share`. No API key. Training skipped for Phase 1.

> Extract receipt/invoice fields as **schema-validated JSON** with per-field **evidence** (quote + bbox) and **confidence**. Runs on a laptop.

## Install

```bash
pip install ./sdk          # from repo root
# or editable for development:
pip install -e ./sdk
```

Core deps: `pydantic`, `pillow`, `jsonschema`. Optional:

```bash
pip install './sdk[ocr]'        # pytesseract for evidence + ocr_regex engine
pip install './sdk[smolvlm2]'   # local HF SmolVLM2 weights (optional engine)
pip install './sdk[all]'
```

Also needs a local [Ollama](https://ollama.com) with `qwen2.5vl:3b` for the default engine (`ollama pull qwen2.5vl:3b`). System `tesseract` for OCR evidence.

## Quick start

```python
from tinydoc import ReceiptPipeline

pipe = ReceiptPipeline("auto")   # ollama if up; OCR fills empty fields only if schema fails
r = pipe.extract("receipt.jpg")
print(r.fields)          # {"company": ..., "date": ..., "address": ..., "total": ...}
print(r.schema_valid)    # True/False (JSON Schema)
print(r.confidence)      # 0..1 aggregate
for fr in r.field_results:
    print(fr.name, fr.value, fr.confidence, fr.evidence
```

CLI:

```bash
tinydoc extract ./receipts/ --engine ollama --out results.jsonl --overlay overlays/
```

Overlay PNGs draw OCR-span evidence boxes + field labels (`tinydoc.draw_overlay`).

## Engines (measured — `evaluation/phase0/results/baseline_table.md`)

| Engine | Field F1 | Schema | Notes |
|--------|----------|--------|-------|
| **ollama:qwen2.5vl:3b** | **0.870** | 1.000 | default; free, local (unconstrained + salvage + prompt-v1) |
| ReceiptPipeline e2e | 0.870 | 1.000 | sanitize + jsonschema path (`run_pipeline_eval.py`) |
| FUNSD forms (n=50) | 0.352 | 0.880 | second vertical; form NER → receipt-shaped keys (`run_funsd_eval.py`) |
| auto (`RoutedEngine`) | — | — | ollama if up; OCR fills **empty** fields only when schema fails |
| ocr_regex | 0.227 | 0.600 | floor / offline fallback |
| smolvlm2 (base 2.2B) | 0.330 | 0.960 | optional local weights |

**Do not** set Ollama `format: "json"` — constrained decoding truncates long addresses and can drop `total`. The SDK uses unconstrained prompts + JSON salvage + prompt-v1 (char-by-character address rules, `num_predict=512`).

## Evidence & honesty

- Evidence kind is always `ocr_span` (Tesseract word boxes) — **not** VLM-predicted boxes.
- Schema is packaged (`tinydoc/schemas/sroie_receipt.schema.json`); validation via `jsonschema`.
- Contaminated training rows for the eval holdout were removed (`data/training/manifest_train_clean.jsonl`). Free-engine numbers were never trained on this data.

## Layout

```
sdk/tinydoc/
  pipeline.py   # engines, schema, sanitize, confidence, ReceiptPipeline
  evidence.py   # OCR-span evidence
  overlay.py    # draw bboxes + labels
  cli.py        # tinydoc extract
  schemas/      # JSON Schema per doc type
```

## License

Apache 2.0
