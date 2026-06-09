#!/usr/bin/env bash
# Launch a MiniMax-M3 (cmf) Claude Code agent in a detached tmux session.
# Usage: launch_minimax.sh <session-name> "<prompt>" [extra cmf args...]
# Example: launch_minimax.sh docs-pass "summarize README" --strict-mcp-config
#
# Robustness notes (learned the hard way):
#  - Long prompts sent via `send-keys -l` trigger the terminal's bracketed-paste
#    mode, which swallows the Enter and/or truncates the line (zsh drops into
#    `quote>`). So we write the prompt to a temp file and type only a SHORT
#    command that reads it at runtime — the typed line stays tiny.
#  - Enter must be a SEPARATE send-keys call; bundling it (or sending keys into
#    a busy cmf) registers as an interrupt and aborts the agent.
set -euo pipefail

NAME="${1:?usage: launch_minimax.sh <session-name> \"<prompt>\" [extra cmf args...]}"
PROMPT="${2:?prompt required}"
shift 2 || true
EXTRA="$*"   # optional flags passed through to cmf (e.g. --strict-mcp-config)

PROMPT_FILE="$(mktemp "/tmp/cmf_prompt_${NAME}.XXXXXX.txt")"
printf '%s' "$PROMPT" > "$PROMPT_FILE"

tmux kill-session -t "$NAME" 2>/dev/null || true
tmux new-session -d -s "$NAME" -x 200 -y 50
sleep 1
# short command — the shell expands $(cat ...) at submit time, so nothing long
# is ever pasted into the TUI.
tmux send-keys -t "$NAME" -l "cmf ${EXTRA:+$EXTRA }\"\$(cat $PROMPT_FILE)\""
sleep 0.6
tmux send-keys -t "$NAME" Enter          # submit — always a separate call

echo "launched MiniMax session '$NAME' (prompt: $PROMPT_FILE)"
echo "tail:  tmux capture-pane -t $NAME -p | grep -v '^[[:space:]]*\$' | tail"
echo "kill:  tmux kill-session -t $NAME"
