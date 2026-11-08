# TinyDoc 0.2.0 — local-first grounded document extraction

**TinyDoc turns receipt and invoice images into schema-validated JSON with
per-field evidence (quote + OCR bbox) and confidence. It runs local Qwen VLMs
through Ollama; no API key, no cloud, no telemetry.**

```bash
pip install tinydoc==0.2.0
ollama pull qwen2.5vl:3b            # quality tier
ollama pull qwen3-vl:2b-instruct    # fast tier
```

> **Status (2026-11-07):** 0.2.0 is published on PyPI. The SDK and evaluation
> harness are measured. The first-party 256M model is retired (OCRBench 0.0%) and
> the SmolVLM2-2.2B SFT track is closed (no gain over base). The product is the
> engine + trust layer, not the old checkpoint. Full analysis:
> [docs/v3_plan.md](https://github.com/eulogik/TinyDoc-VLM/blob/main/docs/v3_plan.md).

![TinyDoc architecture](https://raw.githubusercontent.com/eulogik/TinyDoc-VLM/main/docs/assets/architecture.png)

## Measured results

Receipts: SROIE public-train-derived clean-100, n=100.

| Engine | Field F1 | Address | Schema | Median latency | Errors |
|---|---:|---:|---:|---:|---:|
| **Qwen2.5-VL-3B** (`ollama:qwen2.5vl:3b`) | **0.9475** | 0.86 | 1.00 | 14.1 s | 0 |
| **Qwen3-VL-2B-Instruct** (`ollama:qwen3-vl:2b-instruct`) | **0.9200** | 0.84 | 1.00 | 7.0 s | 0 |
| Hybrid v2 first-party (`hybrid_2p2b`) | 0.603 | 0.37 | 0.93 | ~8 s | — |
| PP-OCR + heuristics | 0.376 | 0.03 | — | — | — |
| Tesseract + regex | 0.227 | — | 0.60 | — | — |

Invoices: public synthetic benchmark, page 1 only, n=181.

| Engine | Key-header | Fully correct | p50 latency |
|---|---:|---:|---:|
| Qwen2.5-VL-3B | 91.9% | 65.7% | 26.5 s |
| Qwen3-VL-2B-Instruct | **93.5%** | **71.8%** | **16.1 s** |

Multipage probe (2B, 10 synthetic invoices, 46 pages, every page sent at
1400 px): **4/10 fully correct** versus **0/10** with page 1 only; key-header
accuracy **71.1%** versus **63.3%**. Paired tests are not significant (n=10),
two 8-page invoices with 110–117 line items failed with runaway line-item
generation, and mean line-item F1 fell from 0.418 to 0.322. The SDK extraction
path remains single-image.

> **Caveat:** these are **not external benchmark numbers**. The 100-doc set is a
> duplicate-filtered split of the public SROIE **train** pool; Qwen may have seen
> it during pretraining. The original 100-doc set is 60% leaked. The invoice set
> is synthetic and vendor-made, and the 181-document probe is page 1 only; real
> supplier invoices are unmeasured and flagging fails (0/7). The multipage probe
> is 10 documents, 2B only, descriptive, and not a shipped SDK path. Use the
> numbers for same-document engine comparisons, not as generalization claims.

## Install

```bash
pip install tinydoc==0.2.0
```

Core deps: `pydantic`, `pillow`, `jsonschema`. Optional extras:

```bash
pip install 'tinydoc[ocr]'      # pytesseract for evidence + ocr_regex
pip install 'tinydoc[ocr-pp]'   # RapidOCR ONNX evidence
pip install 'tinydoc[smolvlm2]' # optional local HF SmolVLM2 engine
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

# Fast tier: Qwen3-VL-2B-Instruct, ~2x faster on receipts
# pipe = ReceiptPipeline("ollama:qwen3-vl:2b-instruct")

result = pipe.extract("receipt.jpg")
print(result.fields)        # {"company": ..., "date": ..., "address": ..., "total": ...}
print(result.schema_valid)  # True/False (JSON Schema)
print(result.confidence)    # 0..1 aggregate, for review routing only
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
| `ollama:qwen3-vl:2b-instruct` | Fast tier; 1.9 GB Q4, ~2x faster on receipts |
| `auto` | Ollama if reachable; OCR floor if the schema would fail |
| `ocr_regex` | Offline floor (Tesseract + rules) |
| `mlx_2p2b` | Legacy v2 edge engine (MLX SmolVLM2-2.2B 4-bit) |
| `hybrid_2p2b` | Legacy v2 product path: MLX fields + frozen F3 OCR address |

**Do not** set Ollama `format: "json"`: constrained decoding truncates long
addresses and can drop `total`. The SDK uses unconstrained prompts, JSON salvage,
and prompt-v1.

### Environment

- `TINYDOC_OLLAMA_TIMEOUT` (default 180 s): raise to 600 for slow local runs.
- `TINYDOC_OLLAMA_NUM_CTX` (default 8192): Ollama context window. The implicit
  4096 rejected 2481×3508 scans with HTTP 400; keep 8192 for full pages.

## Evidence and honesty

- Evidence kind is always `ocr_span` (RapidOCR word boxes), **not** VLM-predicted
  boxes. Treat it as provenance, not model grounding. A 100-box probe measured
  VLM bbox hit@0.5 of **0.20** (2B) and **0.04** (3B), below the 0.70 gate, so
  OCR spans remain the shipped path.
- The schema is packaged (`tinydoc/schemas/sroie_receipt.schema.json`); validation
  uses `jsonschema`.
- Confidence is a routing signal, not a probability. Evidence-off runs are
  constant (0.4025; field ECE 0.52–0.55). The default evidence-on path measures
  field ECE **0.115–0.135** raw and **0.005–0.014** after document-grouped
  isotonic OOF; document-level confidence is **not** calibrated (ECE 0.09–0.10).
- Every published number recomputes from `evaluation/phase0/results/`:
  `python3 evaluation/phase0/recompute_scores.py` must exit 0.
- System card: [docs/MODEL_CARD.md](https://github.com/eulogik/TinyDoc-VLM/blob/main/docs/MODEL_CARD.md).

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
