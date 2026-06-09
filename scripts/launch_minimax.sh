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
sleep 0.3

# 1) Write the prompt to a temp file and type ONLY `cmf "$(cat file)"`.
#    WHY: passing the prompt inline (`-p "$PROMPT"` via send-keys) breaks
#    whenever the prompt contains backticks, quotes, or `$(...)` — the pane's
#    shell evaluates them, dropping zsh into `dquote>`/`quote>` and corrupting
#    the launch. Reading from a file means the prompt bytes never pass through
#    a typed command line, so any characters are safe. The `$(cat file)` output
#    is NOT re-scanned by the shell, so backticks inside the file stay literal.
#    `cmf` is the claude-minimax-free launcher (scrubs parent-agent env, points
#    at api.minimaxi.com) and takes the prompt as its first positional arg.
PROMPT_FILE="$(mktemp /tmp/cmf-prompt.XXXXXX)"
printf '%s' "$PROMPT" > "$PROMPT_FILE"
tmux send-keys -t "$NAME" -l "cmf \"\$(cat $PROMPT_FILE)\""

# 2) Send Enter as a SEPARATE send-keys call. Bundling it with the previous
#    call (or sending keys while the agent is mid-turn) registers as an
#    interrupt and aborts the agent. This is the #1 most common mistake.
tmux send-keys -t "$NAME" Enter

echo "Launched: tmux session '$NAME'"
echo "  attach:   tmux attach -t $NAME"
echo "  tail:     tmux capture-pane -t $NAME -p | grep -v '^\s*\$' | tail"
echo "  kill:     tmux kill-session -t $NAME"
