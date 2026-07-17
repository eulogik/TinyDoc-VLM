#!/usr/bin/env bash
# Launch the TinyDoc-VLM 768 retrain on a Colab T4 via the Google Colab CLI.
# Requires: pip install google-colab-cli  (free; uses your Colab account tier)
#
# Usage:
#   ./training/run_colab_cli.sh            # default: T4, 8000 steps, batch 8
#   STEPS=12000 ./training/run_colab_cli.sh
#   GPU=L4 ./training/run_colab_cli.sh
#
# The CLI provisions a fresh T4 VM, runs training/colab_train.py headlessly,
# and tears the VM down on completion. Checkpoints are saved to Google Drive
# (mounted inside colab_train.py), so you can re-run to resume across sessions.
set -euo pipefail

GPU="${GPU:-T4}"
STEPS="${STEPS:-8000}"
BATCH="${BATCH:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
MAX_SEQ="${MAX_SEQ:-512}"

HERE="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT="$HERE/training/colab_train.py"

if ! command -v colab >/dev/null 2>&1; then
    echo "Install the Colab CLI first:  pip install google-colab-cli" >&2
    exit 1
fi

# Ephemeral job: provision T4, run script with args, release VM on finish.
colab run --gpu "$GPU" "$SCRIPT" \
    --steps "$STEPS" \
    --batch-size "$BATCH" \
    --grad-accum "$GRAD_ACCUM" \
    --max-seq-length "$MAX_SEQ"
