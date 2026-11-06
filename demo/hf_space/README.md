---
title: TinyDoc
emoji: 📄
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---
# TinyDoc — local-first grounded document extraction

This is a **static project card**, not a live model demo. The retired 256M
checkpoint measured **0.0% on OCRBench (n=999 scored)** and is no longer served.
The working 2B/3B engines run locally through Ollama.

Measured on a SROIE public-train-derived clean-100 (not an external benchmark):

| Engine | Field F1 | Address | Schema | Median |
|---|---:|---:|---:|---:|
| Qwen2.5-VL-3B — quality | **0.9475** | 0.86 | 1.00 | 14.1 s |
| Qwen3-VL-2B-Instruct — fast | **0.9200** | 0.84 | 1.00 | 7.0 s |
| Hybrid v2 — first-party | 0.603 | 0.37 | 0.93 | ~8 s |
| PP-OCR + heuristics | 0.376 | 0.03 | — | — |

See [the repository](https://github.com/eulogik/TinyDoc-VLM) for install
instructions and [docs/v3_plan.md](https://github.com/eulogik/TinyDoc-VLM/blob/main/docs/v3_plan.md)
for the full evidence.
