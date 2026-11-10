#!/usr/bin/env bash
# Run the page-wise multipage invoice probe, then score it with the benchmark's
# own scorer. Inference over many documents needs explicit approval (AGENTS.md).
# Usage: scripts/run_invoice_pagewise.sh [model] [tag] [prompt_version]
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="/opt/homebrew/bin:$PATH"
export TINYDOC_OLLAMA_TIMEOUT="${TINYDOC_OLLAMA_TIMEOUT:-600}"
export TINYDOC_OLLAMA_NUM_CTX="${TINYDOC_OLLAMA_NUM_CTX:-8192}"
export TINYDOC_OLLAMA_NUM_PREDICT="${TINYDOC_OLLAMA_NUM_PREDICT:-2048}"

MODEL="${1:-qwen3-vl:2b-instruct}"
TAG="${2:-qwen3-vl_2b_mp_pagewise}"
PROMPT_VERSION="${3:-1}"
MANIFEST="${INVOICE_MANIFEST:-evaluation/invoice/manifest_multipage.jsonl}"
BENCH="${INVOICE_BENCHMARK:-/Volumes/KIOXIA 1TB/tinydoc/invoice-extraction-benchmark}"
ROOT="$(pwd)"
RESULTS="evaluation/invoice/results"
mkdir -p "$RESULTS" logs

if [ ! -f "$MANIFEST" ]; then
  echo "ERROR: manifest missing: $MANIFEST" >&2
  exit 2
fi
if ! curl -sf --max-time 10 "http://127.0.0.1:11434/api/tags" >/dev/null; then
  echo "ERROR: Ollama is not running; start it before the probe" >&2
  exit 3
fi

echo "START $MODEL prompt=$PROMPT_VERSION manifest=$MANIFEST $(date '+%Y-%m-%d %H:%M:%S')"
python3 evaluation/invoice/run_invoice_pagewise.py \
  --manifest "$MANIFEST" \
  --model "$MODEL" \
  --prompt-version "$PROMPT_VERSION" \
  --resume \
  --out "$RESULTS/rows_invoice_${TAG}.jsonl" \
  --preds "$RESULTS/preds_invoice_${TAG}.json" \
  >> "logs/invoice_${TAG}.log" 2>&1 &
pid=$!
scripts/watch_mem.sh "$pid" 6 10 &
wpid=$!
wait "$pid"; rc=$?
kill "$wpid" 2>/dev/null || true
echo "END inference rc=$rc $(date '+%Y-%m-%d %H:%M:%S')"

if [ "$rc" -eq 0 ]; then
  (
    cd "$BENCH" || exit 1
    python3 -m scorer.evaluate "$ROOT/$RESULTS/preds_invoice_${TAG}.json" \
      --out "$ROOT/$RESULTS/scores_invoice_${TAG}.json" \
      --markdown "$ROOT/$RESULTS/scores_invoice_${TAG}.md" \
      --allow-missing \
      >> "$ROOT/logs/invoice_${TAG}.log" 2>&1
  )
  echo "SCORE rc=$? $(date '+%Y-%m-%d %H:%M:%S')"
fi
exit "$rc"
