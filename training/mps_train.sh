#!/bin/bash
# Overnight MPS training supervisor: runs full_train.py on the Mac's M4 GPU
# in a retry loop. full_train.py resumes from checkpoints/full768/latest
# (step.txt) on every restart, so a crash/reboot just continues.
# Logs: /tmp/mps_train.log  |  Checkpoints: /Volumes/KIOXIA 1TB/tinydoc_mps
cd /Users/eulogikdeveloper/Documents/TinyDoc-VLM || exit 1

export PYTORCH_ENABLE_MPS_FALLBACK=1

while true; do
  echo "[mps-supervisor] $(date -u +%H:%M:%S) launching full_train..."
  python3 -u -c "
import torch.multiprocessing as mp
mp.set_start_method('fork')
import sys
sys.path.insert(0, 'training')
sys.argv = ['full_train.py',
  '--model-id', '/Volumes/KIOXIA 1TB/huggingface_cache/hub/models--eulogik--TinyDoc-VLM-768-init/snapshots/a72a7b27e9b890d9a32f7cbdc4dfa416130fa70c',
  '--manifest', '/Volumes/KIOXIA 1TB/mpsdata/manifest.jsonl',
  '--steps', '8000', '--batch-size', '1', '--grad-accum', '4',
  '--warmup', '500', '--lr', '1e-4', '--max-seq-length', '512',
  '--device', 'mps', '--bf16', '--grad-checkpoint',
  '--output-dir', '/Volumes/KIOXIA 1TB/tinydoc_mps/checkpoints/full768',
  '--max-samples', '2500', '--save-every', '500', '--save-latest-every', '50',
  '--log-every', '10']
import full_train
full_train.main()
" >> /tmp/mps_train.log 2>&1
  rc=$?
  echo "[mps-supervisor] $(date -u +%H:%M:%S) full_train exited rc=$rc; restarting in 30s"
  sleep 30
done
