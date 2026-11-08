#!/usr/bin/env bash
# Run the DocVQA reading probe for one model. Inference over many documents
# needs explicit approval first (AGENTS.md). CPU-only; no model download here.
# Usage: scripts/run_docvqa_probe.sh [model] [tag] [limit]
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="/opt/homebrew/bin:$PATH"
export TINYDOC_OLLAMA_TIMEOUT="${TINYDOC_OLLAMA_TIMEOUT:-600}"
export TINYDOC_OLLAMA_NUM_CTX="${TINYDOC_OLLAMA_NUM_CTX:-8192}"

MODEL="${1:-qwen3-vl:2b-instruct}"
TAG="${2:-qwen3-vl_2b_instruct}"
LIMIT="${3:-200}"
MANIFEST="${DOCVQA_MANIFEST:-evaluation/docvqa/manifest_docvqa_200.jsonl}"
RESULTS="evaluation/docvqa/results"
mkdir -p "$RESULTS" logs

if [ ! -f "$MANIFEST" ]; then
  echo "ERROR: manifest missing: $MANIFEST (run build_manifest.py first)" >&2
  exit 2
fi
if ! curl -sf --max-time 10 "http://127.0.0.1:11434/api/tags" >/dev/null; then
  echo "ERROR: Ollama is not running; start it before the probe" >&2
  exit 3
fi

echo "START $MODEL $(date '+%Y-%m-%d %H:%M:%S')"
python3 evaluation/docvqa/run_docvqa_probe.py \
  --manifest "$MANIFEST" \
  --model "$MODEL" \
  --limit "$LIMIT" \
  --resume \
  --out "$RESULTS/rows_docvqa_${TAG}.jsonl" \
  --scores "$RESULTS/scores_docvqa_${TAG}.json" \
  >> "logs/docvqa_${TAG}.log" 2>&1 &
pid=$!
scripts/watch_mem.sh "$pid" 6 10 &
wpid=$!
wait "$pid"; rc=$?
kill "$wpid" 2>/dev/null || true
echo "END rc=$rc $(date '+%Y-%m-%d %H:%M:%S')"
exit "$rc"
