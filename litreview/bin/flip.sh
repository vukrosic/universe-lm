#!/usr/bin/env bash
# flip.sh — change a paper's lit-review status in ONE call:
#   1. rewrites paper.md frontmatter (status + updated, optional round)
#   2. appends one event line to that paper's log.jsonl
#
# Usage:
#   litreview/bin/flip.sh <paper-slug> <new-status> <agent> [note] [round]
set -euo pipefail

paper="${1:?paper-slug required}"
to="${2:?new-status required}"
agent="${3:?agent required}"
note="${4:-}"
round_arg="${5:-}"

root="$(cd "$(dirname "$0")/../.." && pwd)"
f="$root/litreview/papers/$paper/paper.md"
log="$root/litreview/papers/$paper/log.jsonl"
[ -f "$f" ] || { echo "no such paper: $f" >&2; exit 1; }

ts="$(date -u +%FT%TZ)"
from="$(awk -F': *' '/^status:/{print $2; exit}' "$f")"

awk -v to="$to" -v ts="$ts" -v rnd="$round_arg" '
  NR==1 && $0=="---"{infm=1; print; next}
  infm && $0=="---"{infm=0; print; next}
  infm && /^status:/{print "status: " to; next}
  infm && /^updated:/{print "updated: " ts; next}
  infm && /^round:/{ if(rnd!=""){print "round: " rnd} else {print}; next}
  {print}
' "$f" > "$f.tmp" && mv "$f.tmp" "$f"

round="${round_arg:-$(awk -F': *' '/^round:/{print $2; exit}' "$f")}"
note="${note//\\/\\\\}"; note="${note//\"/\\\"}"

printf '{"ts":"%s","agent":"%s","paper":"%s","from":"%s","to":"%s","round":%s,"note":"%s"}\n' \
  "$ts" "$agent" "$paper" "$from" "$to" "$round" "$note" >> "$log"

echo "$paper: $from -> $to (round $round) logged"
