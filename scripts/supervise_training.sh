#!/usr/bin/env bash
# Supervisor: ratchet LoRA training to 1200 global iters across watchdog kills
# and NIRNAY memory waves. Each segment warm-starts from the latest checkpoint
# (--adapter-path); checkpoints (every 25 iters) are archived with GLOBAL iter
# identity so val-selection can pick among them later.
#
# Safety: every segment runs under watch_mem.sh (avail<=10% or growing swap
# aborts). The supervisor itself is footprint-free (bash + sleep).
# Provenance: all decisions appended to $SCRATCH/supervisor.log.
#
# Usage: nohup bash scripts/supervise_training.sh >/dev/null 2>&1 &
set -u

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SCRATCH="/var/folders/6m/l_wd40y91jqbj36ty4nz2skm0000gn/T/opencode"
OUT="$SCRATCH/adapters/full"
TARGET=1200
SUPLOG="$SCRATCH/supervisor.log"

log() { echo "[$(date -u +%FT%TZ)] $*" >> "$SUPLOG"; }

avail_mb() {
  vm_stat | awk '/page size of/{ps=$8} /Pages free/{f=$3} /Pages inactive/{i=$3} /Pages speculative/{s=$3} END{gsub(/\./,"",f);gsub(/\./,"",i);gsub(/\./,"",s); printf "%.0f", (f+i+s)*ps/1048576}'
}

nir_cpu() {
  ps aux | grep "NIRNAY/.venv" | grep -v grep | awk '{print int($3)}' | head -1
}

done_global() {
  ls "$OUT/banked"/global_*_adapters.safetensors 2>/dev/null \
    | grep -oE 'global_[0-9]+' | grep -oE '[0-9]+' | sort -n | tail -1
}

# archive this segment's numbered saves with global identity ($1 = seg start)
archive_seg() {
  local start="$1" f nnn g
  mkdir -p "$OUT/banked"
  for f in "$OUT"/0000*_adapters.safetensors; do
    [ -e "$f" ] || continue
    nnn=$(basename "$f" | grep -oE '^[0-9]+')
    g=$(($start + 10#$nnn))
    mv "$f" "$OUT/banked/global_${g}_adapters.safetensors"
    echo "global_${g} <= segment(start=$start)+seg-iter $((10#$nnn)) [supervisor $(date -u +%FT%TZ)]" >> "$OUT/banked/MAPPING.txt"
  done
  echo "$start" > "$OUT/banked/CURRENT_START"
}

wait_window() {
  local i nir a
  for i in $(seq 1 100); do
    nir=$(nir_cpu); a=$(avail_mb)
    if { [ -z "$nir" ] || [ "$nir" -lt 50 ]; } && [ "$a" -ge 5000 ]; then
      log "window open (nir=${nir:-gone} avail=${a}MB)"
      return 0
    fi
    sleep 30
  done
  return 1
}

log "supervisor start (target=$TARGET)"
while true; do
  DONE=$(done_global); DONE=${DONE:-0}
  if [ "$DONE" -ge "$TARGET" ]; then
    log "COMPLETE: banked $DONE >= $TARGET"
    break
  fi
  if ! wait_window; then
    log "no quiet window in 50min at done=$DONE — supervisor exiting (relaunch me)"
    exit 3
  fi
  SEGLOG="$SCRATCH/train_seg_${DONE}.log"
  rm -f "$SEGLOG"
  log "launching segment from $DONE (remaining $((TARGET - DONE)))"
  if [ "$DONE" -eq 0 ]; then
    bash "$REPO/scripts/train_smolvlm_lora.sh" full > "$SEGLOG" 2>&1 &
  else
    bash "$REPO/scripts/train_smolvlm_lora.sh" resume "$DONE" > "$SEGLOG" 2>&1 &
  fi
  TPID=$!
  bash "$SCRATCH/watch_mem.sh" "$TPID" none "$SCRATCH/watchdog_seg_${DONE}.log" >/dev/null 2>&1 &
  wait "$TPID"; CODE=$?
  LASTITER=$(grep -a -oE "Iter [0-9]+" "$SEGLOG" | tail -1 || echo "Iter ?")
  log "segment from=$DONE exit=$CODE last=($LASTITER)"
  archive_seg "$DONE"
  sleep 5
done
