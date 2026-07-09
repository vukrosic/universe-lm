#!/usr/bin/env bash
# Capped, auto-killing launcher for token2science simulation agents.
#
# Enforces a GLOBAL cap on concurrent codex sessions (named simu-*) so that
# recursive spawning - agents launching agents - can NEVER explode. Both the
# coordinator and each contributor spawn ONLY through this script, so the cap
# is global no matter who calls it.
#
# It wraps the real fire-and-forget launcher, which already auto-kills a
# session the moment the agent prints `FINAL:` (or on timeout). So sessions
# clean themselves up; this script only gates how many run at once.
#
# Usage:
#   capped_launch.sh <name> "<prompt>" [timeout_seconds]
# Env:
#   T2S_SIM_MAX   max concurrent simu-* sessions (default 4)
#
# Exit 5 = cap stayed full past the wait budget, nothing launched.
set -euo pipefail

NAME="${1:?usage: capped_launch.sh <name> <prompt> [timeout]}"
PROMPT="${2:?missing prompt}"
TIMEOUT="${3:-600}"
MAX="${T2S_SIM_MAX:-4}"
INNER="/Users/vukrosic/.claude/skills/launch-codex-tmux/scripts/launch_and_wait.sh"

count_sessions() { tmux ls 2>/dev/null | grep -c '^simu-' || true; }

# Soft global cap: wait for a free slot, up to ~10 minutes.
waited=0
while [ "$(count_sessions)" -ge "$MAX" ]; do
  if [ "$waited" -ge 600 ]; then
    echo "CAP-TIMEOUT: $MAX simu-* sessions busy for 600s; did not launch '$NAME'" >&2
    exit 5
  fi
  sleep 5
  waited=$((waited + 5))
done

# Launch as simu-<name>; the inner launcher auto-kills on FINAL or timeout.
exec "$INNER" "simu-$NAME" "$PROMPT" "$TIMEOUT"
