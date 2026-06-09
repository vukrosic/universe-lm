#!/usr/bin/env bash
# Launch a MiniMax-M3 (cmf) Claude Code agent in a detached tmux session.
#
# Usage:
#   scripts/launch_minimax.sh <session-name> "<prompt>"
#
# Each rule below exists because skipping it silently breaks the session.
# See ~/.claude/skills/launch-minimax-tmux/SKILL.md for full context.

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <session-name> \"<prompt for MiniMax>\"" >&2
  exit 2
fi

NAME="$1"
shift
PROMPT="$*"

# Refuse to clobber an existing session — prompts into a live pane always
# corrupt it. If you really want to reuse the name, kill the old one first.
if tmux has-session -t "$NAME" 2>/dev/null; then
  echo "tmux session '$NAME' already exists. Kill it first:" >&2
  echo "  tmux kill-session -t $NAME" >&2
  exit 3
fi

# Detached, wide enough that the TUI doesn't wrap weirdly.
tmux new-session -d -s "$NAME" -x 200 -y 50

# 1) Send the cmf command with the prompt as a single arg.
#    We use `claude-minimax-free` (the launcher that scrubs the parent-agent
#    env and points at api.minimaxi.com). The prompt is passed via -p so the
#    agent starts working immediately instead of waiting at an empty prompt.
# We use `claude-minimax-free` (the launcher that scrubs the parent-agent
#    env and points at api.minimaxi.com). The prompt is passed via -p so the
#    agent starts working immediately instead of waiting at an empty prompt.
#    NOTE: use literal send (-l) for the whole command line, not printf %q —
#    printf %q over-escapes em-dashes / apostrophes / spaces, leaving visible
#    backslashes in the pane. send-keys -l already does the right escaping.
tmux send-keys -t "$NAME" -l "claude-minimax-free -p \"$PROMPT\""

# 2) Send Enter as a SEPARATE send-keys call. Bundling it with the previous
#    call (or sending keys while the agent is mid-turn) registers as an
#    interrupt and aborts the agent. This is the #1 most common mistake.
tmux send-keys -t "$NAME" Enter

echo "Launched: tmux session '$NAME'"
echo "  attach:   tmux attach -t $NAME"
echo "  tail:     tmux capture-pane -t $NAME -p | grep -v '^\s*\$' | tail"
echo "  kill:     tmux kill-session -t $NAME"
