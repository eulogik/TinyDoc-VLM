#!/usr/bin/env bash
# Push + run the TinyDoc-VLM 768 retrain on Kaggle GPU (free 30h/week).
# Requires:
#   1. kaggle CLI:      pip install kaggle  +  ~/.kaggle/kaggle.json
#   2. HF_TOKEN secret: https://www.kaggle.com/code/eulogik/tinydoc-vlm-768-retrain/settings
#      (Add-ons -> Secrets -> HF_TOKEN)
#   3. Data uploaded once: python training/upload_data_to_hf.py
#
# Usage:
#   ./training/run_kaggle.sh            # default: 8000 steps, batch 8
#   STEPS=12000 ./training/run_kaggle.sh
#   BATCH=4 STEPS=30000 ./training/run_kaggle.sh
#
# Sessions are resumable: checkpoints sync to the HF model repo during
# training, so re-running after a 9-12h session kill continues from the
# last saved step.
set -euo pipefail

STEPS="${STEPS:-8000}"
BATCH="${BATCH:-8}"
KERNEL="eulogik/tinydoc-vlm-768-retrain"

HERE="$(cd "$(dirname "$0")/.." && pwd)"

if ! command -v kaggle >/dev/null 2>&1; then
    echo "Install the Kaggle CLI first:  pip install kaggle" >&2
    echo "Then put your API credentials at ~/.kaggle/kaggle.json" >&2
    exit 1
fi

# Stage a copy of the notebook with STEPS/BATCH baked in (env vars do not
# propagate into the Kaggle VM).
STAGE="$(mktemp -d)"
cp "$HERE/training/kaggle/kernel-metadata.json" "$STAGE/"
sed "s/STEPS = os.environ.get('STEPS', '8000')/STEPS = '$STEPS'/; \
     s/STEPS = os.environ.get('STEPS', '[0-9]*')/STEPS = '$STEPS'/; \
     s/BATCH = os.environ.get('BATCH', '8')/BATCH = '$BATCH'/; \
     s/BATCH = os.environ.get('BATCH', '[0-9]*')/BATCH = '$BATCH'/" \
    "$HERE/training/kaggle/kaggle_notebook.ipynb" > "$STAGE/kaggle_notebook.ipynb"

kaggle kernels push -p "$STAGE"
rm -rf "$STAGE"
echo "Kernel submitted. Track progress:"
echo "  kaggle kernels status $KERNEL"
echo "  kaggle kernels output $KERNEL -p /tmp/tinydoc-kaggle-out   # logs"
