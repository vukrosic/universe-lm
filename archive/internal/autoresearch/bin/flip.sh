#!/usr/bin/env bash
# flip.sh — change an idea's pipeline status in ONE call:
#   1. rewrites idea.md frontmatter (status + updated, optional round)
#   2. appends one event line to that idea's log.jsonl
# Use this instead of hand-editing both files (avoids status/log desync).
#
# Round cap: a `needs-recode` flip on an idea whose `round` already hit
# MAX_RECODE_ROUNDS (default 3) is auto-closed to `rejected` + logged in
# closed.md instead — stops the recode->run->diverge->recode loop. See below.
#
# Usage:
#   autoresearch/bin/flip.sh <idea-slug> <new-status> <agent> [note] [round]
#
# Examples:
#   autoresearch/bin/flip.sh 002-cautious-adamw reviewing reviewer "claimed"
#   autoresearch/bin/flip.sh 002-cautious-adamw needs-review reviser "applied 4 findings" 2
set -euo pipefail

idea="${1:?idea-slug required}"
to="${2:?new-status required}"
agent="${3:?agent required}"
note="${4:-}"
round_arg="${5:-}"

root="$(cd "$(dirname "$0")/../.." && pwd)"
f="$root/autoresearch/ideas/$idea/idea.md"
log="$root/autoresearch/ideas/$idea/log.jsonl"
[ -f "$f" ] || { echo "no such idea: $f" >&2; exit 1; }

ts="$(date -u +%FT%TZ)"
from="$(awk -F': *' '/^status:/{print $2; exit}' "$f")"

# --- Round cap: stop the recode -> run -> diverge -> recode loop ---
# When something would bounce an idea back to `needs-recode` but it has already
# exhausted its recode budget (round >= MAX_RECODE_ROUNDS), close it instead of
# retrying forever: flip to `rejected` and record the close in closed.md. Every
# needs-recode write (runner, run-button, orchestrate reclaim) routes through
# here, so this one check covers them all. Idempotent — once rejected the idea
# is terminal and orchestrate skips it, so it can never be bounced again.
MAX_RECODE_ROUNDS="${MAX_RECODE_ROUNDS:-3}"
capped=0
if [ "$to" = "needs-recode" ]; then
  cap_round="$(awk -F': *' '/^round:/{print $2; exit}' "$f")"
  cap_round="${cap_round:-0}"
  if [ "$cap_round" -ge "$MAX_RECODE_ROUNDS" ] 2>/dev/null; then
    to="rejected"; capped=1
  fi
fi

# Rewrite only the YAML frontmatter (first --- ... --- block).
awk -v to="$to" -v ts="$ts" -v rnd="$round_arg" '
  NR==1 && $0=="---"{infm=1; print; next}
  infm && $0=="---"{infm=0; print; next}
  infm && /^status:/{print "status: " to; next}
  infm && /^updated:/{print "updated: " ts; next}
  infm && /^round:/{ if(rnd!=""){print "round: " rnd} else {print}; next}
  {print}
' "$f" > "$f.tmp" && mv "$f.tmp" "$f"

round="${round_arg:-$(awk -F': *' '/^round:/{print $2; exit}' "$f")}"
note="${note//\\/\\\\}"; note="${note//\"/\\\"}"   # escape \ and " for JSON

printf '{"ts":"%s","agent":"%s","idea":"%s","from":"%s","to":"%s","round":%s,"note":"%s"}\n' \
  "$ts" "$agent" "$idea" "$from" "$to" "$round" "$note" >> "$log"

# Round-cap close: append one greppable line under closed.md's append marker
# (newest first, matching how the reviewer/evidence steps write closes). Guard
# on the slug so a re-run never double-appends.
if [ "$capped" = 1 ]; then
  closed="$root/autoresearch/closed.md"
  marker='<!-- reviewer/evidence step appends one line per close here -->'
  cline="- $idea — reject: exhausted $cap_round recode rounds (MAX_RECODE_ROUNDS=$MAX_RECODE_ROUNDS), axis abandoned — $(date -u +%F)"
  if [ -f "$closed" ] && ! grep -qF -- "$idea — reject: exhausted" "$closed"; then
    awk -v marker="$marker" -v line="$cline" '
      {print}
      index($0, marker) && !done {print line; done=1}
    ' "$closed" > "$closed.tmp" && mv "$closed.tmp" "$closed"
  fi
  echo "$idea: round cap hit ($cap_round >= $MAX_RECODE_ROUNDS) — closed to closed.md"
fi

echo "$idea: $from -> $to (round $round) logged"
