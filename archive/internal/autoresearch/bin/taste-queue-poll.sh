#!/usr/bin/env bash
# taste-queue-poll.sh — poll the autoresearch taste queue.
# Cron'd every 3 min. Logs to /tmp/taste-queue.log. If items are in
# `needs-taste`, prints one line per item. No side effects — does not
# claim or flip anything (the taste-reviewer prompt does that).
set -uo pipefail

ROOT="/Users/vukrosic/my-life/llm-research-kit-scaling"
LOG="/tmp/taste-queue.log"

cd "$ROOT" || { echo "$(date -u +%FT%TZ) [ERR] cd failed" >> "$LOG"; exit 1; }

TS=$(date -u +%FT%TZ)
echo "--- $TS ---" >> "$LOG"

TASTE=$(grep -l "status: needs-taste" autoresearch/ideas/*/idea.md 2>/dev/null | sed 's|autoresearch/ideas/||;s|/idea.md||')

if [ -n "$TASTE" ]; then
  echo "needs-taste:" >> "$LOG"
  for f in $TASTE; do echo "  $f" >> "$LOG"; done
else
  echo "queue empty" >> "$LOG"
fi
