#!/usr/bin/env bash
# Run the invoice transfer probe for one model, then score with the benchmark's
# own scorer. Inference is long-running; watch_mem.sh guards system memory.
# Usage: scripts/run_invoice_probe.sh [model] [tag]
# Env: INVOICE_MANIFEST (default page-1 manifest), INVOICE_ALLOW_MISSING=1 for a
#      subset manifest (e.g. the 10-document multipage probe).
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="/opt/homebrew/bin:$PATH"
export TINYDOC_OLLAMA_TIMEOUT="${TINYDOC_OLLAMA_TIMEOUT:-600}"
export TINYDOC_OLLAMA_NUM_CTX="${TINYDOC_OLLAMA_NUM_CTX:-8192}"

MODEL="${1:-qwen2.5vl:3b}"
TAG="${2:-qwen2.5vl_3b}"
MANIFEST="${INVOICE_MANIFEST:-evaluation/invoice/manifest_v1.jsonl}"
NUM_PREDICT="${INVOICE_NUM_PREDICT:-768}"
BENCH="${INVOICE_BENCHMARK:-/Volumes/KIOXIA 1TB/tinydoc/invoice-extraction-benchmark}"
ROOT="$(pwd)"
RESULTS="evaluation/invoice/results"
mkdir -p "$RESULTS" logs

if [ ! -f "$MANIFEST" ]; then
  echo "ERROR: manifest missing: $MANIFEST" >&2
  exit 2
fi
echo "START $MODEL manifest=$MANIFEST num_predict=$NUM_PREDICT $(date '+%Y-%m-%d %H:%M:%S')"
if ! curl -sf --max-time 10 "http://127.0.0.1:11434/api/tags" >/dev/null; then
  echo "ERROR: Ollama is not running; start it before the probe" >&2
  exit 3
fi
python3 evaluation/invoice/run_invoice_eval.py \
  --manifest "$MANIFEST" \
  --model "$MODEL" \
  --num-predict "$NUM_PREDICT" \
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
  ALLOW_MISSING=""
  if [ "${INVOICE_ALLOW_MISSING:-0}" = "1" ]; then
    ALLOW_MISSING="--allow-missing"
  fi
  (
    cd "$BENCH" || exit 1
    python3 -m scorer.evaluate "$ROOT/$RESULTS/preds_invoice_${TAG}.json" \
      --out "$ROOT/$RESULTS/scores_invoice_${TAG}.json" \
      --markdown "$ROOT/$RESULTS/scores_invoice_${TAG}.md" \
      $ALLOW_MISSING \
      >> "$ROOT/logs/invoice_${TAG}.log" 2>&1
  )
  echo "SCORE rc=$? $(date '+%Y-%m-%d %H:%M:%S')"
fi
exit "$rc"
