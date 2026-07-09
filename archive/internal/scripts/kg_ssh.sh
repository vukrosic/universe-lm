#!/usr/bin/env bash
# kg_ssh.sh — Drive the KaggleLink SSH tunnel + file sync.
#
# Companion to kaggle_push.sh (which handles the batch push/status/pull loop
# via the kaggle CLI). This script is for the *interactive* path: a Zrok
# tunnel into a running Kaggle notebook, giving you a real bash shell,
# tmux/screen, rsync in both directions.
#
# Prereqs (one-time):
#   1. SSH key in place: ~/.ssh/kaggle_rsa, with the pubkey reachable on a
#      public raw URL (we use a gist: see ~/.ssh/config Host Kaggle).
#   2. `ssh Kaggle` works once a Kaggle notebook has the KaggleLink setup
#      cell running. See docs/kaggle_ssh_setup.md (or
#      https://github.com/bhdai/kagglelink) for the cell body.
#   3. zrok is installed and `zrok enable`d locally. The setup cell uses the
#      same Zrok account on the Kaggle side.
#
# Usage:
#   bash scripts/kg_ssh.sh                  # open the SSH session
#   bash scripts/kg_ssh.sh test             # is port 9191 reachable locally?
#   bash scripts/kg_ssh.sh up   <local> [remote]   # rsync into the notebook
#   bash scripts/kg_ssh.sh down <remote> [local]   # rsync out of the notebook
#   bash scripts/kg_ssh.sh start            # spawn the local zrok listener
#   bash scripts/kg_ssh.sh stop             # kill the local zrok listener
#   bash scripts/kg_ssh.sh status           # is the local listener running?
#   bash scripts/kg_ssh.sh keys             # show the public key + its raw URL
set -euo pipefail

SSH_KEY="$HOME/.ssh/kaggle_rsa"
PUBKEY_URL="https://gist.githubusercontent.com/vukrosic/c5b1a0d510b5127fce71a9f953573e0c/raw/kaggle_rsa.pub"
SSH_TARGET="Kaggle"
LOCAL_PORT=9191
KAGGLE_WORK="/kaggle/working"
# Set this to the share token printed by the KaggleLink setup cell. Override
# at the call site with KG_SHARE=<token> bash scripts/kg_ssh.sh start.
SHARE_TOKEN="${KG_SHARE:-m4eylqjtfdbz}"
ZROK_STATE_DIR="$HOME/.local/var/kg_ssh"
ZROK_LOG="$ZROK_STATE_DIR/zrok-access.log"
ZROK_PID="$ZROK_STATE_DIR/zrok-access.pid"

cmd="${1:-ssh}"
shift || true

case "$cmd" in
  ssh|"")
    if [ ! -f "$SSH_KEY" ]; then
      echo "Missing $SSH_KEY. Run: ssh-keygen -t rsa -b 4096 -f $SSH_KEY -N ''"
      exit 1
    fi
    # First, check the tunnel is actually up — if not, the user gets a
    # clearer error than a generic ssh timeout.
    if ! (echo > /dev/tcp/127.0.0.1/$LOCAL_PORT) >/dev/null 2>&1; then
      echo "Port $LOCAL_PORT is closed locally."
      echo "Is a Kaggle notebook running the KaggleLink setup cell?"
      echo "See: https://github.com/bhdai/kagglelink"
      exit 2
    fi
    exec ssh "$SSH_TARGET"
    ;;

  test)
    if (echo > /dev/tcp/127.0.0.1/$LOCAL_PORT) >/dev/null 2>&1; then
      echo "OK — 127.0.0.1:$LOCAL_PORT is open (Kaggle tunnel is up)."
    else
      echo "CLOSED — 127.0.0.1:$LOCAL_PORT not listening."
      echo "Start a Kaggle notebook running the KaggleLink setup cell first."
      exit 1
    fi
    ;;

  up)
    src="${1:?usage: kg_ssh.sh up <local> [remote]}"
    dst="${2:-$(basename "$src")}"
    rsync -avz --progress -e ssh "$src" "$SSH_TARGET:$KAGGLE_WORK/$dst"
    ;;

  down)
    src="${1:?usage: kg_ssh.sh down <remote> [local]}"
    dst="${2:-$(basename "$src")}"
    rsync -avz --progress -e ssh "$SSH_TARGET:$KAGGLE_WORK/$src" "$dst"
    ;;

  keys)
    echo "Private key: $SSH_KEY (mode 600)"
    echo "Public  key: $SSH_KEY.pub"
    echo "Raw URL:     $PUBKEY_URL"
    if [ -f "$SSH_KEY.pub" ]; then
      echo
      echo "Fingerprint:"
      ssh-keygen -lf "$SSH_KEY.pub"
    fi
    ;;

  start)
    # Spawn the local zrok access listener in the background. Required for
    # `ssh Kaggle` (and rsync) to reach the notebook. Stays running until
    # `kg_ssh.sh stop` or the system reboots.
    if [ -f "$ZROK_PID" ] && kill -0 "$(cat "$ZROK_PID")" 2>/dev/null; then
      echo "Already running (PID $(cat "$ZROK_PID")). Use 'stop' first to restart."
      exit 0
    fi
    command -v zrok >/dev/null || { echo "zrok not installed (brew install zrok)"; exit 1; }
    mkdir -p "$ZROK_STATE_DIR"
    echo "Starting zrok access for share $SHARE_TOKEN → 127.0.0.1:$LOCAL_PORT"
    nohup zrok access private "$SHARE_TOKEN" --headless > "$ZROK_LOG" 2>&1 &
    echo $! > "$ZROK_PID"
    sleep 3
    if (echo > /dev/tcp/127.0.0.1/$LOCAL_PORT) >/dev/null 2>&1; then
      echo "OK — listening on 127.0.0.1:$LOCAL_PORT (PID $(cat "$ZROK_PID"))"
    else
      echo "FAILED — see $ZROK_LOG"
      tail -20 "$ZROK_LOG"
      exit 1
    fi
    ;;

  stop)
    if [ ! -f "$ZROK_PID" ]; then
      echo "No PID file at $ZROK_PID — not running?"
      exit 0
    fi
    pid="$(cat "$ZROK_PID")"
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid"
      echo "Stopped zrok access (PID $pid)"
    else
      echo "PID $pid not alive."
    fi
    rm -f "$ZROK_PID"
    ;;

  status)
    if [ -f "$ZROK_PID" ] && kill -0 "$(cat "$ZROK_PID")" 2>/dev/null; then
      echo "RUNNING — PID $(cat "$ZROK_PID"), log: $ZROK_LOG"
    else
      echo "STOPPED — run 'kg_ssh.sh start' to begin."
    fi
    if (echo > /dev/tcp/127.0.0.1/$LOCAL_PORT) >/dev/null 2>&1; then
      echo "PORT  — 127.0.0.1:$LOCAL_PORT is accepting connections."
    else
      echo "PORT  — 127.0.0.1:$LOCAL_PORT is closed (Kaggle cell not running, or token expired)."
    fi
    ;;

  *)
    sed -n '2,30p' "$0"  # print the header as usage
    exit 1
    ;;
esac
