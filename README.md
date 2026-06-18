# TinyDoc-VLM

TinyDoc-VLM is a 150M–300M parameter vision-language model trained exclusively for document understanding — invoices, receipts, forms, tables, charts, ID cards, and business documents. It runs efficiently on resource-constrained devices, such as a Raspberry Pi or local CPU, with a VRAM footprint under 1GB.

## Features

- **Tiny and Specialized**: Focused exclusively on document understanding, outperforming general VLMs of similar size.
- **Layout-Aware**: Incorporates segment-level 2D layout embeddings and region-aware visual representations.
- **Efficient Token Compression**: Custom pixel shuffle connector reduces dense visual sequences by 4x to 9x, saving sequence budget for long document reasoning.
- **Fast and Local**: Runs at >100 tokens/second on standard CPUs using ONNX/GGUF/CoreML optimization.

## Installation

Install the package in editable mode:

```bash
pip install -e .
```

For GPU acceleration and flash-attention:

```bash
pip install flash-attn --no-build-isolation
```

## Structure

- `tinydoc_vlm/`: Core model code (architecture, configurations, custom processors).
- `data/`: Synthetic document generator and real-dataset loaders.
- `training/`: Autoregressive trainers, schedulers, and multi-stage configs.
- `evaluation/`: Benchmark scripts (DocVQA ANLS, CORD F1, TEDS Table similarity).
- `export/`: ONNX/GGUF/AWQ compilation scripts.
- `sdk/`: Lightweight client SDK for production integrations.
- `demo/`: Gradio dashboard for interactive visualization.
