#!/usr/bin/env bash
# orchestrate.sh — one idempotent lit-review orchestration tick.
#
# Same pattern as autoresearch/bin/orchestrate.sh:
#   1. RECLAIM stale -ing locks
#   2. FAN OUT one cmf worker per actionable needs-* paper
#   3. Optionally launch global scout if screen queue is thin
#   4. REPORT queue depths
#
# Usage:  litreview/bin/orchestrate.sh [--dry-run]
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PAPERS="$ROOT/litreview/papers"
FLIP="$ROOT/litreview/bin/flip.sh"
PROMPTS="$ROOT/litreview/prompts"
PDIR="/tmp/litreview-orch-prompts"; mkdir -p "$PDIR"
STALE_MIN="${STALE_MIN:-7}"
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1

now=$(date -u +%s)

recover_to () { case "$1" in
  scouting) echo needs-scout;; screening) echo needs-screen;;
  digesting) echo needs-digest;; digestreviewing) echo needs-digestreview;;
  redigesting) echo needs-redigest;; *) echo "";; esac; }

prompt_file () { case "$1" in
  needs-scout|needs-rescout) echo scout.md;;
  needs-screen) echo screener.md;;
  needs-digest|needs-redigest) echo digester.md;;
  needs-digestreview) echo digest-reviewer.md;;
  *) echo "";; esac; }

role () { case "$1" in
  needs-scout|needs-rescout) echo scout;; needs-screen) echo screen;;
  needs-digest|needs-redigest) echo digest;; needs-digestreview) echo digest-review;;
  *) echo "";; esac; }

is_busy () {
  tmux capture-pane -t "$1" -p 2>/dev/null | grep -v '^[[:space:]]*$' | tail -4 \
    | grep -qE "Puzzling|Thinking|Considering|Calculat|Crunch|Cook|esticulat|searched|Did [0-9]|Shenanigan|Hatching|Working|Forging|tokens · "
}

iso_to_epoch () { date -j -u -f "%Y-%m-%dT%H:%M:%SZ" "$1" +%s 2>/dev/null || echo 0; }

launched=0; reclaimed=0; busy=0; skipped=0

echo "=== litreview orchestrate $(date -u +%FT%TZ) (stale=${STALE_MIN}m dry=$DRY) ==="

for f in "$PAPERS"/*/paper.md; do
  [ -f "$f" ] || continue
  slug="$(basename "$(dirname "$f")")"
  [[ "$slug" == _closed ]] && continue
  status="$(awk -F': *' '/^status:/{print $2; exit}' "$f")"
  updated="$(awk '/^updated:/{sub(/^updated:[[:space:]]*/,""); print; exit}' "$f")"
  num="${slug%%-*}"
  sess="lr_${num}"

  case "$status" in
    done|rejected) continue;;
  esac

  rec="$(recover_to "$status")"
  if [ -n "$rec" ]; then
    age=$(( (now - $(iso_to_epoch "$updated")) / 60 ))
    if tmux has-session -t "$sess" 2>/dev/null && is_busy "$sess"; then
      busy=$((busy+1)); echo "  $slug: $status (worker $sess BUSY) — leave"; continue
    fi
    if [ "$age" -ge "$STALE_MIN" ]; then
      echo "  $slug: $status stale ${age}m -> reclaim to $rec"
      [ "$DRY" = 0 ] && "$FLIP" "$slug" "$rec" orchestrator "reclaim stale ${status} lock (${age}m)" >/dev/null
      status="$rec"; reclaimed=$((reclaimed+1))
    else
      echo "  $slug: $status (lock fresh ${age}m, worker idle) — wait"; continue
    fi
  fi

  pf="$(prompt_file "$status")"; rl="$(role "$status")"
  [ -z "$pf" ] && { echo "  $slug: $status (no handler) — skip"; skipped=$((skipped+1)); continue; }

  if tmux has-session -t "$sess" 2>/dev/null; then
    if is_busy "$sess"; then busy=$((busy+1)); echo "  $slug: $status (worker $sess BUSY) — leave"; continue
    else tmux kill-session -t "$sess" 2>/dev/null; fi
  fi

  cat > "$PDIR/$sess.txt" <<EOF
You are the $rl agent for the lit-review pipeline. Authoritative protocol: read $PROMPTS/$pf and follow it EXACTLY. Use $ROOT/litreview/bin/flip.sh for EVERY status change (never hand-edit frontmatter).

TARGET (only this one): litreview/papers/$slug/ (current status: $status). Do the work for this gate, write your artifact file, then flip the status via flip.sh per the prompt's routing.

Read litreview/brief.md and litreview/seen.md before any search or verdict.

CRITICAL — prior workers may have exited WITHOUT saving. You MUST write your artifact AND run flip.sh. Before you stop, run: grep -H "status:" litreview/papers/$slug/paper.md and confirm the status changed away from "$status". Do not stop until verified.
EOF

  echo "  $slug: $status -> launch worker $sess ($rl)"
  if [ "$DRY" = 0 ]; then
    tmux new-session -d -s "$sess" -x 200 -y 50
    sleep 0.3
    tmux send-keys -t "$sess" -l "cmf \"\$(cat $PDIR/$sess.txt)\""
    tmux send-keys -t "$sess" Enter
  fi
  launched=$((launched+1))
done

# Global scout when screen queue is thin (WIP gate from scout.md)
screen_ct=$(grep -l "status: needs-screen" "$PAPERS"/*/paper.md 2>/dev/null | grep -v _closed | wc -l | tr -d ' ')
upstream=$(grep -L "status: \(done\|rejected\)" "$PAPERS"/*/paper.md 2>/dev/null | grep -v _closed | wc -l | tr -d ' ')
scout_sess="lr_scout"
if [ "$screen_ct" -lt 3 ] && [ "$upstream" -lt 12 ]; then
  if tmux has-session -t "$scout_sess" 2>/dev/null && is_busy "$scout_sess"; then
    echo "  scout: global worker BUSY — leave"
  else
    [ "$DRY" = 0 ] && tmux has-session -t "$scout_sess" 2>/dev/null && tmux kill-session -t "$scout_sess" 2>/dev/null || true
    n=$((3 - screen_ct)); [ "$n" -gt 3 ] && n=3
    cat > "$PDIR/scout.txt" <<EOF
You are the scout agent for the lit-review pipeline. Read $PROMPTS/scout.md and follow it EXACTLY.

File up to $n new papers at needs-screen this pass (WIP gate permitting). Use litreview/bin/flip.sh only when re-scouting existing papers.

Read litreview/brief.md and litreview/seen.md first. Do not ask the human — act and stop when done or SKIP.
EOF
    echo "  scout: screen queue thin ($screen_ct) -> launch global $scout_sess (allowance=$n)"
    if [ "$DRY" = 0 ]; then
      tmux new-session -d -s "$scout_sess" -x 200 -y 50
      sleep 0.3
      tmux send-keys -t "$scout_sess" -l "cmf \"\$(cat $PDIR/scout.txt)\""
      tmux send-keys -t "$scout_sess" Enter
    fi
    launched=$((launched+1))
  fi
else
  echo "  scout: skip (screen=$screen_ct upstream=$upstream)"
fi

echo "--- digest queue (needs-digest / needs-digestreview) ---"
grep -l "status: \(needs-digest\|needs-digestreview\)" "$PAPERS"/*/paper.md 2>/dev/null \
  | sed "s#$PAPERS/##;s#/paper.md##" | grep -v _closed | sed 's/^/  /' || echo "  (none)"

echo "--- summary: launched=$launched reclaimed=$reclaimed busy=$busy skipped=$skipped ---"
