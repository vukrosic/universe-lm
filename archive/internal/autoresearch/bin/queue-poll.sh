#!/usr/bin/env bash
# queue-poll.sh — poll the autoresearch code-impl queue.
# Cron'd every 3 min (see CronCreate in the main session). Logs to
# /tmp/code-impl-queue.log. If items are found in `needs-recode`
# (re-code loop) or `needs-plan` (new work), prints one line per item.
# No side effects — does not claim or flip anything.
set -uo pipefail

ROOT="/Users/vukrosic/my-life/llm-research-kit-scaling"
LOG="/tmp/code-impl-queue.log"

cd "$ROOT" || { echo "$(date -u +%FT%TZ) [ERR] cd failed" >> "$LOG"; exit 1; }

TS=$(date -u +%FT%TZ)
echo "--- $TS ---" >> "$LOG"

RECODE=$(grep -l "status: needs-recode" autoresearch/ideas/*/idea.md 2>/dev/null | sed 's|autoresearch/ideas/||;s|/idea.md||')
PLAN=$(grep -l "status: needs-plan" autoresearch/ideas/*/idea.md 2>/dev/null | sed 's|autoresearch/ideas/||;s|/idea.md||')

if [ -n "$RECODE" ]; then
  echo "needs-recode:" >> "$LOG"
  for f in $RECODE; do echo "  $f" >> "$LOG"; done
fi
if [ -n "$PLAN" ]; then
  echo "needs-plan:" >> "$LOG"
  for f in $PLAN; do echo "  $f" >> "$LOG"; done
fi
if [ -z "$RECODE" ] && [ -z "$PLAN" ]; then
  echo "queue empty" >> "$LOG"
fi
