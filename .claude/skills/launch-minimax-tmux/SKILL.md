---
name: launch-minimax-tmux
description: >-
  Launch a MiniMax-M3 (the `cmf` / claude-minimax-free launcher) Claude Code
  agent in a detached tmux session, reliably. Use this whenever you need to
  spin up a background MiniMax worker — e.g. "fire up a minimax agent", "run
  this on minimax in tmux", "launch a cmf session", "start a background
  minimax to build X", offloading a side task to MiniMax, or spawning a
  parallel agent that isn't the GPU runner. Also use it when a cmf/minimax
  tmux session is failing to authenticate ("Not logged in", endless
  API_TIMEOUT retries) — this skill encodes the fix. Prefer this over typing
  raw tmux/cmf commands, because getting the Enter-key handling and the
  parent-env scrub wrong silently breaks the session.
---

# Launch a MiniMax-M3 agent in tmux

`cmf` = the `claude-minimax-free` launcher → runs Claude Code against
**MiniMax-M3** (`api.minimaxi.com/anthropic`). This skill spawns one in a
detached tmux session so it survives disconnects and runs in parallel with
your other work.

## Fastest path: the helper script

```bash
scripts/launch_minimax.sh <session-name> "<prompt for MiniMax>"
```

It creates the session, sends the prompt, submits it correctly, and prints how
to tail it. Use a descriptive `<session-name>` (e.g. `dashboard`, `docs-pass`)
so `tmux ls` stays readable. That's usually all you need.

The script writes the prompt to a temp file and types only `cmf "$(cat file)"`,
because **long prompts pasted via `send-keys` break** — the terminal's
bracketed-paste mode eats the Enter or truncates the line and zsh drops into
`quote>`. Never `send-keys` a multi-line/long prompt directly; always go through
a file (the helper does this for you).

## What the script does (and why each step matters)

If you launch by hand instead of using the script, follow this exactly — each
rule exists because skipping it silently breaks the session:

1. `tmux new-session -d -s NAME -x 200 -y 50` — detached, wide enough that the
   TUI doesn't wrap weirdly.
2. Send the command text, **then send Enter as a SEPARATE `send-keys` call.**
   Bundling Enter into the same `send-keys` (or sending keys while the agent is
   mid-turn) registers as an **interrupt** and aborts the agent. This is the
   single most common mistake.
3. **Never send keys into a `cmf` that's still generating.** Wait until the pane
   shows an idle `❯` prompt. Tail with
   `tmux capture-pane -t NAME -p | grep -v '^\s*$' | tail`.
4. A healthy first reply takes a few seconds (M3 "thinks") and ends with
   `⏺ <answer>`.

## If auth fails ("Not logged in" / API_TIMEOUT loop)

This almost always means the launcher's env scrub is missing. When `cmf` is
spawned from *inside another Claude Code agent*, the parent's environment leaks
in (`CLAUDECODE=1`, `CLAUDE_CODE_SDK_HAS_OAUTH_REFRESH`, `AI_AGENT`, …), which
makes the child ignore the MiniMax token and fall back to OAuth → it fails.

Fix: the `claude-minimax-free` script must `unset` those vars before `exec`.
Confirm with:
```bash
grep -n "unset CLAUDECODE" ~/.local/bin/claude-minimax-free
```
If that line is absent, add an `unset` of the parent-agent vars right after
`set -e` (see the script comment for the full list). The API key itself is
rarely the problem — verify it independently:
```bash
KEY=$(grep ANTHROPIC_AUTH_TOKEN ~/.local/bin/claude-minimax-free | sed -E 's/.*="([^"]+)".*/\1/')
curl -s -o /dev/null -w '%{http_code}\n' https://api.minimaxi.com/anthropic/v1/messages \
  -H "x-api-key: $KEY" -H "anthropic-version: 2023-06-01" -H "Content-Type: application/json" \
  -d '{"model":"MiniMax-M3","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}'
```
`200` = key and endpoint are fine; the problem is the env scrub, not the key.

## Sending a follow-up to a running session

Only when the pane is idle at `❯`:
```bash
tmux send-keys -t NAME -l "your follow-up text"
tmux send-keys -t NAME Enter        # always separate
```

## Cleanup

```bash
tmux kill-session -t NAME
```
