# DocVQA reading probe

DocVQA is a public document-reading benchmark: a question and a short answer per
document image. This probe measures whether the base reader can resolve document
text at all. It is **not** an extraction metric, **not** a product claim, and
**not** an external test set: DocVQA is public and the Qwen base models may have
seen it during pretraining. Label every reported number that way.

## What is measured

- **ANLS** (average normalized Levenshtein similarity, threshold 0.5) — the
  standard DocVQA metric.
- **Exact match** after lowercasing, punctuation removal, and article stripping.
- **Errors and latency** per question.

One question per image is sampled deterministically (seed 42) so the probe stays
CPU-affordable and the manifest is reproducible.

**Single-reference caveat.** The mirror carries one target per question, while
official DocVQA scoring takes the best ANLS over up to 10 human answers. This
probe's ANLS is therefore a **single-reference** reading signal and is **not
comparable to DocVQA leaderboard numbers**. Report it as an internal reading
probe on a public subset, nothing more.

## Build the manifest

```bash
python3 evaluation/docvqa/build_manifest.py \
    --pairs data/datasets/docvqa/docvqa.json \
    --data-dir data/datasets/docvqa \
    --out evaluation/docvqa/manifest_docvqa_200.jsonl \
    --limit 200
```

The pair file is the lmms-lab mirror used by this repo (JSONL with
`image_path`, `prompt`, `target`). The builder writes `id`, `image`, `question`,
`target`, and `source` per row. No model is called.

## Run the probe

Inference over many documents needs explicit approval under `AGENTS.md`.
With approval:

```bash
scripts/run_docvqa_probe.sh qwen3-vl:2b-instruct qwen3-vl_2b_instruct
scripts/run_docvqa_probe.sh qwen2.5vl:3b qwen2.5vl_3b
```

Or directly:

```bash
python3 evaluation/docvqa/run_docvqa_probe.py \
    --manifest evaluation/docvqa/manifest_docvqa_200.jsonl \
    --model qwen3-vl:2b-instruct \
    --out evaluation/docvqa/results/rows_docvqa_qwen3-vl_2b_instruct.jsonl \
    --scores evaluation/docvqa/results/scores_docvqa_qwen3-vl_2b_instruct.json \
    --resume
```

The runner checks Ollama first, appends one row per question (so a crash does
not lose completed work), and writes a compact scores JSON. Keep
`TINYDOC_OLLAMA_NUM_CTX=8192` and `TINYDOC_OLLAMA_TIMEOUT=600` for full pages.

## Report it honestly

- Say **DocVQA reading probe**, never “DocVQA score” as a leaderboard claim.
- Always include: public set, possibly in pretraining, one question per image.
- Do not compare it to the SROIE field-F1 numbers; they measure different tasks.
- No number enters `docs/BENCHMARKS.md` until the rows and scores are committed
  and `python3 evaluation/phase0/recompute_scores.py` passes.
