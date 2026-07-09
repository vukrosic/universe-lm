#!/usr/bin/env bash
# flip-brief.sh — change a brief's pipeline status in ONE call:
#   1. rewrites brief.md frontmatter (status + updated, optional round)
#   2. appends one event line to that brief's log.jsonl
# Brief-pipeline twin of flip.sh (see autoresearch/briefs/PIPELINE.md).
#
# Usage:
#   autoresearch/bin/flip-brief.sh <brief-slug> <new-status> <agent> [note] [round]
#
# Examples:
#   autoresearch/bin/flip-brief.sh 002-pe-zoo scoping brief-reviewer "claimed"
#   autoresearch/bin/flip-brief.sh 002-pe-zoo active vuk "blessed"
set -euo pipefail

brief="${1:?brief-slug required}"
to="${2:?new-status required}"
agent="${3:?agent required}"
note="${4:-}"
round_arg="${5:-}"

root="$(cd "$(dirname "$0")/../.." && pwd)"
f="$root/autoresearch/briefs/$brief/brief.md"
log="$root/autoresearch/briefs/$brief/log.jsonl"
[ -f "$f" ] || { echo "no such brief: $f" >&2; exit 1; }

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

printf '{"ts":"%s","agent":"%s","brief":"%s","from":"%s","to":"%s","round":%s,"note":"%s"}\n' \
  "$ts" "$agent" "$brief" "$from" "$to" "$round" "$note" >> "$log"

echo "$brief: $from -> $to (round $round) logged"
