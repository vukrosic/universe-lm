#!/usr/bin/env bash
# orchestrate.sh — one idempotent, self-healing orchestration tick.
#
# Replaces hand-launching cmf workers. Each run:
#   1. RECLAIM: any idea stuck in an `-ing` lock with a stale `updated`
#      timestamp (and no live busy worker) is flipped back to its `needs-*`.
#   2. FAN OUT: launch ONE dedicated cmf worker per actionable `needs-*` idea,
#      fully in parallel, skipping ideas that already have a live busy worker.
#   3. REPORT: print needs-run/running ideas (the GPU's queue) and a summary.
#
# Safe to run repeatedly (cron/loop). Flakiness self-heals: a worker that
# stalls leaves a lock that the NEXT tick reclaims and relaunches.
#
# Usage:  autoresearch/bin/orchestrate.sh [--dry-run]
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
IDEAS="$ROOT/autoresearch/ideas"
FLIP="$ROOT/autoresearch/bin/flip.sh"
PROMPTS="$ROOT/autoresearch/prompts"
PDIR="/tmp/orch-prompts"; mkdir -p "$PDIR"
STALE_MIN="${STALE_MIN:-7}"        # an -ing lock older than this (min) is dead
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1

now=$(date -u +%s)

# status (-ing lock) -> needs-* to recover to. $2 = slug (for implementing,
# which the code-implementer reuses for BOTH a fresh build and a post-failure
# retry: route by whether a failed run's evidence.md exists).
recover_to () { case "$1" in
  tasting) echo needs-taste;; repitching) echo needs-repitch;;
  reviewing) echo needs-review;; revising) echo needs-revision;;
  planning) echo needs-plan;; recoding) echo needs-recode;;
  implementing) [ -f "$IDEAS/$2/evidence.md" ] && echo needs-recode || echo needs-plan;;
  *) echo "";; esac; }

# needs-* -> canonical prompt file (the worker reads & follows it EXACTLY)
prompt_file () { case "$1" in
  needs-taste) echo idea-taste.md;; needs-repitch) echo idea-miner.md;;
  needs-review) echo idea-reviewer.md;; needs-revision) echo idea-reviser.md;;
  needs-plan|needs-recode) echo code-implementer.md;;
  *) echo "";; esac; }

# needs-* -> short role label (for the worker prompt + session name)
role () { case "$1" in
  needs-taste) echo taste;; needs-repitch) echo repitch;;
  needs-review) echo review;; needs-revision) echo revise;;
  needs-plan) echo plan;; needs-recode) echo recode;;
  *) echo "";; esac; }

# Is a worker live?  return 0 (busy) means a running worker holds the lock;
# return 1 means idle/dead. Lockfile-based, not TUI-grep — worker_run.sh runs
# the agent HEADLESS (no spinner words to capture-pane for), so liveness is
# tracked by the worker's own PID via a lockfile it holds for its lifetime.
# A lock whose PID is gone (worker was kill -9'd) is stale -> clean it, idle.
is_busy () {
  local lock="${ORCH_LOCKDIR:-/tmp/orch-locks}/$1.lock" pid
  [ -f "$lock" ] || return 1
  pid="$(awk '{print $1}' "$lock" 2>/dev/null)"
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then return 0; fi
  rm -f "$lock"; return 1
}

iso_to_epoch () { date -j -u -f "%Y-%m-%dT%H:%M:%SZ" "$1" +%s 2>/dev/null || echo 0; }

launched=0; reclaimed=0; busy=0; skipped=0

echo "=== orchestrate tick $(date -u +%FT%TZ) (stale=${STALE_MIN}m dry=$DRY) ==="

for f in "$IDEAS"/*/idea.md; do
  slug="$(basename "$(dirname "$f")")"
  status="$(awk -F': *' '/^status:/{print $2; exit}' "$f")"
  updated="$(awk '/^updated:/{sub(/^updated:[[:space:]]*/,""); print; exit}' "$f")"
  sess="w_${slug%%-*}"   # e.g. w_016  (short, unique per idea number)

  case "$status" in
    done|rejected|needs-run|running) continue;;  # terminal or GPU-owned
  esac

  # --- 1. RECLAIM stale -ing locks ---
  rec="$(recover_to "$status" "$slug")"
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

  # --- 2. FAN OUT: ensure a worker for this needs-* idea ---
  pf="$(prompt_file "$status")"; rl="$(role "$status")"
  [ -z "$pf" ] && { echo "  $slug: $status (no handler) — skip"; skipped=$((skipped+1)); continue; }

  if tmux has-session -t "$sess" 2>/dev/null; then
    if is_busy "$sess"; then busy=$((busy+1)); echo "  $slug: $status (worker $sess BUSY) — leave"; continue
    else tmux kill-session -t "$sess" 2>/dev/null; fi   # idle/stalled -> relaunch fresh
  fi

  cat > "$PDIR/$sess.txt" <<EOF
You are the $rl agent for the autoresearch pipeline. Authoritative protocol: read $PROMPTS/$pf and follow it EXACTLY. Use $ROOT/autoresearch/bin/flip.sh for EVERY status change (never hand-edit frontmatter).

TARGET (only this one): autoresearch/ideas/$slug/ (current status: $status). Do the work for this gate, write your artifact file, then flip the status via flip.sh per the prompt's routing.

COORDINATION: before editing models/layers.py or configs/llm_config.py, run git diff/status to check conflicts; never push.

CRITICAL — prior workers exited WITHOUT saving. You MUST write your artifact AND run flip.sh. Before you stop, run: grep -H "status:" autoresearch/ideas/$slug/idea.md and confirm the status changed away from "$status". Do not stop until verified.
EOF

  echo "  $slug: $status -> launch worker $sess ($rl)"
  if [ "$DRY" = 0 ]; then
    # Run the worker headless via worker_run.sh: holds a liveness lock (for
    # is_busy) and runs the prompt through the MiniMax->Codex rate-limit
    # fallback, so a MiniMax outage no longer stalls this gate. tmux is kept
    # only so the UI log viewer can tail the pane.
    tmux new-session -d -s "$sess" -x 200 -y 50
    sleep 0.3
    tmux send-keys -t "$sess" -l "$ROOT/autoresearch/bin/worker_run.sh $sess $PDIR/$sess.txt"
    tmux send-keys -t "$sess" Enter
  fi
  launched=$((launched+1))
done

echo "--- GPU queue (needs-run / running) ---"
grep -l "status: \(needs-run\|running\)" "$IDEAS"/*/idea.md 2>/dev/null \
  | sed "s#$IDEAS/##;s#/idea.md##" | sed 's/^/  /' || echo "  (none)"

echo "--- summary: launched=$launched reclaimed=$reclaimed busy=$busy skipped=$skipped ---"

# Snapshot ideas so a worker that rm's an idea.md can't cause real loss (115
# was nearly lost this way; only git history saved it). Best-effort, never
# blocks the tick. Commit ONLY the ideas dir, never push.
if [ "$DRY" = 0 ]; then
  git -C "$ROOT" add -A autoresearch/ideas >/dev/null 2>&1 \
    && git -C "$ROOT" diff --cached --quiet autoresearch/ideas 2>/dev/null \
    || git -C "$ROOT" commit -q -m "orchestrate: idea snapshot $(date -u +%FT%TZ)" -- autoresearch/ideas >/dev/null 2>&1 || true
fi
