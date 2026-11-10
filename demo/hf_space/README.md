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

Invoices: 181 synthetic documents, page 1 only.

| Reader | Key-header | Fully correct | p50 |
|---|---:|---:|---:|
| Qwen2.5-VL-3B | 91.9% | 65.7% | 26.5 s |
| Qwen3-VL-2B-Instruct | **93.5%** | **71.8%** | **16.1 s** |

Multipage invoices (2B, 10 synthetic documents, 46 pages, 1400 px):

| Method | Key-header | Fully correct | Line F1 |
|---|---:|---:|---:|
| All pages in one request (VLM) | 71.1% | 4/10 | 0.322 |
| One page per request (VLM) | 93.3% | 6/10 | 0.848 |
| + deterministic reconciliation | 95.6% | 8/10 | 0.848 |
| + OCR printed totals | **100% (90/90)** | **10/10** | 0.848 |

The 10/10 is a **pipeline result, not a pure VLM score**: OCR reads the printed
subtotal/tax/total and the VLM reads headers and line items. It uses no gold
answers, but line-item F1 is 0.848 and all lines are right on only 1/10. The
exact 95% CI for 10/10 fully correct is 69.2–100% at n=10.

DocVQA reading probe: 200 public images, one question per image, 768 px mirror,
one reference answer. Qwen3-VL-2B-Instruct scores **0.829 ANLS / 0.765 exact**,
0 errors, 3.36 s p50. The set may be in pretraining; official DocVQA uses up to
10 references, so this is not a leaderboard number.

> **Caveats that travel with every number.** The 100-doc receipt set is a
> duplicate-filtered split of the public SROIE **train** pool; Qwen may have seen
> it during pretraining. The original 100-doc set is 60% leaked. Invoice results
> are synthetic and vendor-made; real supplier invoices are unmeasured and
> flagging fails (0/7). The multipage probe is 10 documents at 1400 px, 2B only,
> is descriptive, and is not a shipped SDK path (the SDK is single-image). These
> are same-document engine comparisons, not generalization claims.

See [the repository](https://github.com/eulogik/TinyDoc-VLM) for install
instructions and [docs/v3_plan.md](https://github.com/eulogik/TinyDoc-VLM/blob/main/docs/v3_plan.md)
for the full evidence.
