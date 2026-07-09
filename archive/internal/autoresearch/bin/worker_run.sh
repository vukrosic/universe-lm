#!/usr/bin/env bash
# worker_run.sh — run ONE orchestrate gate worker, headless, with a live
# MiniMax->Codex rate-limit fallback and a deterministic liveness lockfile.
#
# orchestrate.sh launches this inside a detached tmux pane (so the UI log
# viewer can still tail it). Two jobs:
#   1. Hold a lockfile for as long as the worker is alive, so orchestrate's
#      is_busy() can tell "live worker" from "idle/dead pane" WITHOUT grepping
#      TUI spinner words (headless mode emits none). The trap removes it on
#      normal exit AND when orchestrate tmux-kills an idle pane.
#   2. Run the prompt through agent_with_fallback.sh, which live-kills MiniMax
#      the instant it 429s / runs out of tokens and re-runs on Codex — so a
#      MiniMax outage no longer freezes the gates for ~6min per worker.
#
# Usage: worker_run.sh <session-name> <prompt-file>
set -uo pipefail

sess="${1:?usage: worker_run.sh <session> <prompt-file>}"
prompt_file="${2:?prompt-file required}"

LOCKDIR="${ORCH_LOCKDIR:-/tmp/orch-locks}"; mkdir -p "$LOCKDIR"
LOCK="$LOCKDIR/$sess.lock"
echo "$$ $(date -u +%s)" > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

# Live rate-limit fallback wrapper (built in voidspark). Override via env.
FB="${AGENT_FALLBACK_SCRIPT:-/Users/vukrosic/my-life/voidspark/scripts/agent_with_fallback.sh}"

# Primary = MiniMax headless print mode (exits on completion; `< /dev/null`
# gives an immediate stdin EOF). Fallback = Codex headless exec. The prompt is
# passed as a single trailing positional arg, never string-interpolated.
"$FB" \
  'claude-minimax-free -p --output-format stream-json --verbose < /dev/null' \
  "codex exec -m ${CODEX_MODEL:-gpt-5.4-mini} --dangerously-bypass-approvals-and-sandbox" \
  "$(cat "$prompt_file")"
