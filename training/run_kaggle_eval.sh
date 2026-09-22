#!/usr/bin/env bash
# Push + run the TinyDoc-VLM 768 eval on Kaggle GPU.
# Evaluates the current latest/ checkpoint from eulogik/TinyDoc-VLM-768-checkpoints
# on the 12-page eval set; results -> eulogik/TinyDoc-VLM-runtime/eval/eval_768_results.json
set -euo pipefail

KERNEL="eulogikdevelopers/tinydoc-vlm-768-eval"
HERE="$(cd "$(dirname "$0")/.." && pwd)"

if [ -z "${HF_TOKEN:-}" ] && [ -f "$HERE/training/.hf_token" ]; then
    HF_TOKEN="$(cat "$HERE/training/.hf_token")"
fi

STAGE="$(mktemp -d)"
cp "$HERE/training/kaggle/eval/kernel-metadata.json" "$STAGE/"
cp "$HERE/training/kaggle/eval/kaggle_eval_notebook.ipynb" "$STAGE/"

if [ -n "${HF_TOKEN:-}" ]; then
    python3 - "$STAGE/kaggle_eval_notebook.ipynb" "$HF_TOKEN" <<'PYEOF'
import json, sys
path, token = sys.argv[1], sys.argv[2]
nb = json.load(open(path))
src = "".join(nb["cells"][1]["source"])
src = src.replace("import subprocess, sys, os, time\n",
                  f"import subprocess, sys, os, time\nos.environ['HF_TOKEN'] = '{token}'\n", 1)
nb["cells"][1]["source"] = src.splitlines(keepends=True)
json.dump(nb, open(path, "w"))
print("HF_TOKEN injected into staged eval notebook (not committed).")
PYEOF
fi

kaggle kernels push -p "$STAGE"
rm -rf "$STAGE"
echo "Eval kernel submitted. Track progress:"
echo "  kaggle kernels status $KERNEL"
echo "  kaggle kernels output $KERNEL -p /tmp/eval-out   # logs"