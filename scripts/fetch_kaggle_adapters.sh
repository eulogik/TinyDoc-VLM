#!/usr/bin/env bash
# Fetch Kaggle-trained vision adapters from HF Hub to KIOXIA for local eval.
# Torch PEFT adapters (adapter_model.safetensors) are evaluated with
# transformers+MPS locally (exact arch match; no MLX conversion needed).
# Usage: HF_TOKEN=$(cat training/.hf_token) bash scripts/fetch_kaggle_adapters.sh [hub-id]
set -euo pipefail
HUB_ID="${1:-eulogik/SmolVLM2-TinyDoc-distill-vision}"
OUT="/Volumes/KIOXIA 1TB/tinydoc/adapters/kaggle_vision"
: "${HF_TOKEN:?set HF_TOKEN}"
export PATH="/opt/homebrew/bin:$PATH"
mkdir -p "$OUT"
for ckpt in $(HF_TOKEN="$HF_TOKEN" python3 -c "
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ['HF_TOKEN'])
files = api.list_repo_files(repo_id='$HUB_ID', repo_type='model')
ckpts = sorted({f.split('/')[0] for f in files if f.startswith('checkpoint-')}, key=lambda c: int(c.split('-')[1]))
print(' '.join(ckpts))
"); do
  if [ -d "$OUT/$ckpt" ]; then echo "$ckpt present, skipping"; continue; fi
  echo "downloading $ckpt ..."
  HF_TOKEN="$HF_TOKEN" hf download "$HUB_ID" --include "$ckpt/*" --local-dir "$OUT/tmp" >/dev/null 2>&1
  mkdir -p "$OUT/$ckpt" && mv "$OUT/tmp/$ckpt"/* "$OUT/$ckpt"/ && rm -rf "$OUT/tmp"
done
echo "--- fetched: ---"
ls "$OUT"
