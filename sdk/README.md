# TinyDoc 0.2.0 — local-first grounded document extraction

Extract receipt/invoice fields as **schema-validated JSON** with per-field
**evidence** (quote + OCR bbox) and **confidence**. Runs locally with Ollama;
no API key, no cloud, no telemetry.

> **Status (2026-11-06):** the SDK and evaluation harness are measured. The
> first-party 256M model is retired (OCRBench 0.0%) and the SmolVLM2-2.2B SFT
> track is closed (no gain over base). The working product is the engine +
> trust layer, not the old checkpoint. Full analysis:
> [docs/v3_plan.md](https://github.com/eulogik/TinyDoc-VLM/blob/main/docs/v3_plan.md).

## Measured results (SROIE public-train-derived clean-100, n=100)

| Engine | Field F1 | Address | Schema | Median latency | Errors |
|---|---:|---:|---:|---:|---:|
| **Qwen2.5-VL-3B** (`ollama:qwen2.5vl:3b`) | **0.9475** | 0.86 | 1.00 | 14.1 s | 0 |
| **Qwen3-VL-2B-Instruct** (`ollama:qwen3-vl:2b-instruct`) | **0.9200** | 0.84 | 1.00 | 7.0 s | 0 |
| Hybrid v2 first-party (`hybrid_2p2b`) | 0.603 | 0.37 | 0.93 | ~8 s | — |
| PP-OCR + heuristics | 0.376 | 0.03 | — | — | — |
| Tesseract + regex | 0.227 | — | 0.60 | — | — |

> **Caveat:** these are **not external benchmark numbers**. The 100-doc set is a
> duplicate-filtered split of the public SROIE **train** pool; Qwen may have seen
> it during pretraining. The original 100-doc set is 60% leaked. Use the numbers
> for same-document engine comparisons, not as a generalization claim. The next
> eval adds CORD/FUNSD/invoices and a private wild set.

## Install

From PyPI (use this once 0.2.0 is published):

```bash
pip install tinydoc
```

From the repo:

```bash
pip install ./sdk
# editable for development:
pip install -e ./sdk
```

If `pip install tinydoc` resolves to **0.1.2**, that is the legacy 256M API; use
the source install above until 0.2.0 is uploaded. Core deps: `pydantic`,
`pillow`, `jsonschema`. Optional extras:

```bash
pip install './sdk[ocr]'      # pytesseract for evidence + ocr_regex
pip install './sdk[ocr-pp]'   # RapidOCR ONNX evidence
pip install './sdk[smolvlm2]' # optional local HF SmolVLM2 engine
```

For the default engine, install [Ollama](https://ollama.com) and pull a base:

```bash
ollama pull qwen2.5vl:3b            # quality default
ollama pull qwen3-vl:2b-instruct    # fast tier (use -instruct, not the thinking tag)
```

System `tesseract` is needed for OCR-span evidence (`brew install tesseract` on macOS).

## Quick start

```python
from tinydoc import ReceiptPipeline

# Quality default: Qwen2.5-VL-3B
pipe = ReceiptPipeline("ollama:qwen2.5vl:3b")

# Fast tier: Qwen3-VL-2B-Instruct, ~2x faster
# pipe = ReceiptPipeline("ollama:qwen3-vl:2b-instruct")

result = pipe.extract("receipt.jpg")
print(result.fields)        # {"company": ..., "date": ..., "address": ..., "total": ...}
print(result.schema_valid)  # True/False (JSON Schema)
print(result.confidence)    # 0..1 aggregate
for field in result.field_results:
    print(field.name, field.value, field.confidence, field.evidence)
```

CLI:

```bash
tinydoc extract ./receipts/ --engine ollama:qwen2.5vl:3b \
    --out results.jsonl --overlay overlays/
```

Overlay PNGs draw OCR-span evidence boxes and field labels (`tinydoc.draw_overlay`).

## Engines

| Engine id | What it does |
|---|---|
| `ollama` / `ollama:qwen2.5vl:3b` | Quality default; free, local |
| `ollama:qwen3-vl:2b-instruct` | Fast tier; 1.9 GB Q4, ~2x faster |
| `auto` | Ollama if reachable; OCR floor if the schema would fail |
| `ocr_regex` | Offline floor (Tesseract + rules) |
| `mlx_2p2b` | First-party v2 edge engine (MLX SmolVLM2-2.2B 4-bit) |
| `hybrid_2p2b` | v2 product path: MLX fields + frozen F3 OCR address |

**Do not** set Ollama `format: "json"` — constrained decoding truncates long
addresses and can drop `total`. The SDK uses unconstrained prompts, JSON salvage,
and prompt-v1.

### Environment

- `TINYDOC_OLLAMA_TIMEOUT` (default 180 s) — raise to 600 for slow local runs.
- `TINYDOC_OLLAMA_NUM_CTX` (default 8192) — Ollama context window. The implicit
  4096 rejected 2481×3508 scans with HTTP 400; keep 8192 for full pages.

## Evidence and honesty

- Evidence kind is always `ocr_span` (Tesseract word boxes), **not** VLM-predicted
  boxes. Treat it as provenance, not as model grounding.
- The schema is packaged (`tinydoc/schemas/sroie_receipt.schema.json`); validation
  uses `jsonschema`.
- Every published number must recompute from `evaluation/phase0/results/`:
  `python3 evaluation/phase0/recompute_scores.py` must exit 0.
- The 256M model card and the v2 decision are kept for reproducibility; neither is
  a current performance claim.

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
