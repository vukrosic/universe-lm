#!/usr/bin/env bash
# Launch a real-Anthropic Sonnet Claude Code agent in a detached tmux session.
# Sibling of launch_minimax.sh — use when MiniMax is rate-limited. Same
# prompt-via-file and separate-Enter rules (see that script for the why).
#
# Usage:
#   scripts/launch_sonnet.sh <session-name> "<prompt>"

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <session-name> \"<prompt for Sonnet>\"" >&2
  exit 2
fi

NAME="$1"
shift
PROMPT="$*"

if tmux has-session -t "$NAME" 2>/dev/null; then
  echo "tmux session '$NAME' already exists. Kill it first:" >&2
  echo "  tmux kill-session -t $NAME" >&2
  exit 3
fi

tmux new-session -d -s "$NAME" -x 200 -y 50
sleep 0.3

PROMPT_FILE="$(mktemp /tmp/sonnet-prompt.XXXXXX)"
printf '%s' "$PROMPT" > "$PROMPT_FILE"
tmux send-keys -t "$NAME" -l "claude-sonnet-clean \"\$(cat $PROMPT_FILE)\""
tmux send-keys -t "$NAME" Enter

echo "Launched: tmux session '$NAME'"
echo "  attach:   tmux attach -t $NAME"
echo "  tail:     tmux capture-pane -t $NAME -p | grep -v '^\s*\$' | tail"
echo "  kill:     tmux kill-session -t $NAME"
