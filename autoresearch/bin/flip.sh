#!/usr/bin/env bash
# flip.sh — change an idea's pipeline status in ONE call:
#   1. rewrites idea.md frontmatter (status + updated, optional round)
#   2. appends one event line to that idea's log.jsonl
# Use this instead of hand-editing both files (avoids status/log desync).
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

echo "$idea: $from -> $to (round $round) logged"
