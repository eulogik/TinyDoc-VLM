<div align="center">
  <h1>TinyDoc</h1>
  <p><b>Local-first grounded document extraction</b></p>
  <p>Schema-valid JSON · per-field evidence · confidence · Apache 2.0</p>
  <p>Measured <b>0.9475</b> field F1 on our hardest receipt set with Qwen2.5-VL-3B;<br>
     <b>0.9200</b> with Qwen3-VL-2B-Instruct at <b>2× the speed</b>.</p>

[![PyPI](https://img.shields.io/pypi/v/tinydoc?color=blue&label=tinydoc)](https://pypi.org/project/tinydoc/)
[![GitHub](https://img.shields.io/badge/GitHub-Source-181717?logo=github)](https://github.com/eulogik/TinyDoc-VLM)
[![Model Status](https://img.shields.io/badge/Model%20status-v3%20plan-yellow)](docs/v3_plan.md)
[![HF Space](https://img.shields.io/badge/🤗-Demo-yellow)](https://huggingface.co/spaces/eulogik/TinyDoc-VLM)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/eulogik/TinyDoc-VLM/actions/workflows/ci.yml/badge.svg)](https://github.com/eulogik/TinyDoc-VLM/actions)

---

**Built by [eulogik](https://eulogik.com)** — AI infrastructure for document intelligence.

[PyPI](https://pypi.org/project/tinydoc/) · [Source](https://github.com/eulogik/TinyDoc-VLM) · [Model status](docs/v3_plan.md) · [Benchmarks](docs/BENCHMARKS.md) · [Demo](https://huggingface.co/spaces/eulogik/TinyDoc-VLM)
</div>

## What is TinyDoc?

TinyDoc is a Python SDK and CLI that turns document images into **schema-validated
JSON**. It runs a local vision-language model through
[Ollama](https://ollama.com), then adds JSON Schema validation, per-field
OCR-span evidence, and confidence scoring. No API key, no cloud, no telemetry.

The base model is third-party (Qwen2.5-VL / Qwen3-VL); the product is the
grounded extraction layer around it. The first-party 256M checkpoint is retired
(OCRBench 0.0%) and the SmolVLM2-2.2B SFT track is closed (no gain over base).
See [`docs/v3_plan.md`](docs/v3_plan.md) for the evidence.

## Measured results

Same 100 receipts, same prompt, same scorer, same 8k context, Ollama Q4,
temperature 0:

| System | Field F1 | Address | Schema | Median latency | Errors |
|---|---:|---:|---:|---:|---:|
| **Qwen2.5-VL-3B** — quality default | **0.9475** | 0.86 | 1.00 | 14.1 s | 0 |
| **Qwen3-VL-2B-Instruct** — fast tier | **0.9200** | 0.84 | 1.00 | **7.0 s** | 0 |
| Hybrid v2 — first-party, shipped | 0.603 | 0.37 | 0.93 | ~8 s | — |
| PP-OCR + heuristics | 0.376 | 0.03 | — | — | — |
| Tesseract + regex | 0.227 | — | 0.60 | — | — |

> ⚠️ **These are not external benchmark numbers.** The 100-doc set is a
> duplicate-filtered split of the public SROIE **train** pool; Qwen may have seen
> it during pretraining. The original 100-doc set is 60% leaked. Use these numbers
> for same-document engine comparisons, not as a generalization claim. The next
> eval adds CORD/FUNSD/invoices and a private wild set. Every published number
> must recompute from `evaluation/phase0/results/` via
> `python3 evaluation/phase0/recompute_scores.py`.

## Quick start

```bash
# 0.2.0 is currently a source install; PyPI 0.1.2 is the legacy 256M API.
pip install "tinydoc @ git+https://github.com/eulogik/TinyDoc-VLM.git#subdirectory=sdk"
# or, from a clone:
pip install ./sdk

# Default quality engine (free, local)
ollama pull qwen2.5vl:3b
# Fast tier (1.9 GB Q4, ~2x faster)
ollama pull qwen3-vl:2b-instruct
```

```python
from tinydoc import ReceiptPipeline

pipe = ReceiptPipeline("ollama:qwen2.5vl:3b")   # or "ollama:qwen3-vl:2b-instruct"
result = pipe.extract("receipt.jpg")

print(result.fields)        # {"company": ..., "date": ..., "address": ..., "total": ...}
print(result.schema_valid)  # True/False (JSON Schema)
print(result.confidence)    # 0..1 aggregate
for field in result.field_results:
    print(field.name, field.value, field.confidence, field.evidence)
```

```bash
tinydoc extract ./receipts/ --engine ollama:qwen2.5vl:3b \
    --out results.jsonl --overlay overlays/
```

Overlay PNGs draw OCR-span evidence boxes and field labels (`tinydoc.draw_overlay`).
The local Gradio demo is `python demo/app.py --share` (requires Ollama).

### Engines

| Engine id | What it does |
|---|---|
| `ollama` / `ollama:qwen2.5vl:3b` | Quality default; free, local |
| `ollama:qwen3-vl:2b-instruct` | Fast tier; use the `-instruct` tag, not the thinking tag |
| `auto` | Ollama if reachable; OCR floor if the schema would fail |
| `ocr_regex` | Offline floor (Tesseract + rules) |
| `mlx_2p2b` | First-party v2 edge engine (MLX SmolVLM2-2.2B 4-bit) |
| `hybrid_2p2b` | v2 product path: MLX fields + frozen F3 OCR address |

**Do not** set Ollama `format: "json"` — constrained decoding truncates long
addresses and can drop `total`. The SDK uses unconstrained prompts, JSON salvage,
and prompt-v1. `TINYDOC_OLLAMA_NUM_CTX` defaults to 8192; the old implicit 4096
rejected 2481×3508 scans with HTTP 400.

## Model status

| Artifact | Status | Evidence |
|---|---|---|
| **TinyDoc 3.0 engine** (Qwen2.5-VL-3B / Qwen3-VL-2B + trust layer) | ✅ **Current** | 0.9475 / 0.9200 clean-100; zero errors; schema 1.00 |
| Hybrid v2 (`hybrid_2p2b`, MLX + Tesseract F3) | 🚢 Shipped 2026-10-07, superseded | 0.603 clean-100; address 0.37; paired address 31/7/62 |
| SmolVLM2-2.2B SFT / vision LoRA | ❌ Closed | Base 0.536; clean-100 LoRA 0.535–0.540; vision LoRA 0.0395 / 0.4277 |
| TinyDoc-VLM-256M | ⚠️ Retired research checkpoint | OCRBench **0.0%** (n=999 scored); no performance claims |
| TinyDoc-2B grounded LoRA (v3.1) | 🧪 Conditional | Only if v3.0 leaves a measured invoice/grounding residual |

The honest one-liner: **the evaluation harness and product layer are real; the
old first-party models are not competitive.** The current product is a strong
third-party reader behind a grounded extraction layer. The v3.1 plan is the path
to a first-party adapter.

## How it works

```
Receipt / invoice image
    ↓
Engine (Ollama Qwen2.5-VL-3B, Qwen3-VL-2B, MLX v2, or OCR floor)
    ↓
Raw output → JSON extraction → repetition collapse → sanitize
    ↓
JSON Schema validation  ── schema_valid: true/false
    ↓
OCR-span evidence (word boxes → best quote + bbox per field)
    ↓
Confidence (evidence quality × schema × field rules)
    ↓
DocumentResult — fields + per-field {quote, bbox, score} + confidence
```

Evidence is **OCR-span**, not VLM-predicted boxes. Treat it as provenance, not
as model grounding. A 100-box probe measured VLM bbox hit@0.5 of **0.20** (2B) and
**0.04** (3B), below the 0.70 product gate, so OCR spans remain the shipped path.

## Evaluation and honesty

- **SROIE clean-100** is a duplicate-filtered split of the public SROIE train
  pool (`source: sroie_public_train_clean`), not an external test set.
- The original 100-doc SROIE set is **60% leaked** (61/100 also appear in the
  public train split; perceptual match across two resamplers). Its 0.870 headline
  is retained only as a historical, caveated number.
- **FUNSD** is a transfer probe with nonstandard gold mapping; never cite it as a
  leaderboard number.
- **Confidence is only partly calibrated.** With evidence off (the fast eval path) the
  score is a constant 0.4025 and field ECE is 0.52–0.55; the SDK default attaches
  evidence, where field ECE is **0.115–0.135** raw and **0.005–0.014** after
  document-grouped isotonic OOF. Document-level exact-4 confidence is **not**
  reliable (ECE 0.09–0.10). Use field-level scores for review routing, not as a
  document probability.
- **Invoices are measured only on a synthetic public benchmark.** On Invoice
  Extraction Benchmark v1 (181 docs, page 1 only, vendor-made), Qwen2.5-VL-3B
  reaches **90.6% key-header field accuracy** but only **55.8% documents fully
  correct**; flagging is 0/7 and multipage is 0% fully correct. This is not real
  supplier-invoice accuracy. Real invoices remain unmeasured.
- **DocVQA / CORD** are not yet measured. No claims.
- The claim audit recomputes every published pair from committed artifacts:
  `python3 evaluation/phase0/recompute_scores.py` must exit 0.

Measured artifacts live in [`evaluation/phase0/results/`](evaluation/phase0/results/);
methodology and provenance are in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).
Machine-readable provenance and SHA-256 hashes for every set are in
[`eval_provenance.json`](evaluation/phase0/results/eval_provenance.json), guarded by
`tests/test_eval_provenance.py`.

## Development

```bash
python3 -m pytest tests/ -q                 # 116 passed
python3 -m ruff check tinydoc_vlm/ tests/   # CI-scope lint
python3 evaluation/phase0/recompute_scores.py
node scripts/verify-product.mjs audit
```

Build the SDK:

```bash
cd sdk && python3 -m build
```

The release checklist is in [`docs/RELEASE.md`](docs/RELEASE.md).

## Repository layout

| Path | What it is |
|---|---|
| `sdk/tinydoc/` | Python SDK: engines, schema, evidence, confidence, CLI |
| `evaluation/phase0/` | Measured SROIE/FUNSD/OCRBench harness + artifacts |
| `training/` | QLoRA and Kaggle experiment code (historical; tracks closed) |
| `tinydoc_vlm/` | From-scratch 256M model code (retired research artifact) |
| `demo/` | Local Gradio demo + static HF Space card |
| `docs/` | Benchmarks, pivot plan, v3 plan, release checklist |

## Research archive

The original TinyDoc-VLM was a 256M from-scratch model: SigLIP-B/16 vision
encoder + PixelShuffle 3× compressor + SmolLM2-135M decoder. It measured
**0.0% OCRBench** (n=999 scored, 962/999 unique outputs) and is retired from all
performance claims. The 768 retrain had a dead vision tower.

The SmolVLM2-2.2B QLoRA track was also closed: base 0.5357 clean-100 F1, every
clean-100 language-LoRA config 0.535–0.540, vision-LoRA 0.0395 at 5e-5 and
0.4277 at 1e-6. The failure was perception, not JSON formatting.

Read the full evidence and the v3 plan in [`docs/v3_plan.md`](docs/v3_plan.md);
the older decision record is [`docs/pivot_plan.md`](docs/pivot_plan.md).

## Citation

```bibtex
@software{eulogik_tinydoc_2026,
  author = {eulogik},
  title = {TinyDoc: Local-First Grounded Document Extraction},
  year = {2026},
  url = {https://github.com/eulogik/TinyDoc-VLM}
}
```

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache 2.0. See [LICENSE](LICENSE) for details.

---

<div align="center">
  <b>Made with by <a href="https://eulogik.com">eulogik</a></b>
</div>
