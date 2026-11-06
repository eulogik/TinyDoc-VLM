#!/usr/bin/env bash
# macOS memory guard for long inference jobs.
# Usage: scripts/watch_mem.sh <pid> [min_free_pct] [interval_s]
# Logs to $WATCH_MEM_LOG (default: logs/watch_mem.log). Terminates the target
# if system-wide free memory drops below min_free_pct.
set -euo pipefail

PID="${1:?usage: watch_mem.sh <pid> [min_free_pct] [interval_s]}"
MIN_FREE="${2:-8}"
INTERVAL="${3:-5}"
LOG="${WATCH_MEM_LOG:-logs/watch_mem.log}"
mkdir -p "$(dirname "$LOG")"

while kill -0 "$PID" 2>/dev/null; do
  FREE="$(memory_pressure 2>/dev/null | awk -F': ' '/System-wide memory free percentage/ {gsub(/%/,"",$2); print $2}' | tail -1)"
  RSS="$(ps -o rss= -p "$PID" 2>/dev/null | awk '{print int($1/1024)}')"
  printf '%s pid=%s free_pct=%s rss_mb=%s\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')" "$PID" "${FREE:-?}" "${RSS:-?}" >> "$LOG"

  if [ -n "${FREE:-}" ] && [ "$FREE" -lt "$MIN_FREE" ]; then
    printf 'MEMORY GUARD: free %s%% < %s%%; terminating %s\n' \
      "$FREE" "$MIN_FREE" "$PID" >> "$LOG"
    kill -TERM "$PID" 2>/dev/null || true
    exit 1
  fi
  sleep "$INTERVAL"
done
