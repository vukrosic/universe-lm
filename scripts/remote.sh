#!/usr/bin/env bash
#
# remote.sh — one entry point for connecting to and controlling the remote
# Vast.ai training box. Config lives in `.remote` (gitignored); copy it from
# `.remote.example` once per rental. The PyTorch venv (REMOTE_VENV, default
# /venv/main) is activated automatically for every remote python command.
#
# Usage:  scripts/remote.sh <command> [args]
# Run `scripts/remote.sh help` for the full list.

set -euo pipefail

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONF="${REPO_ROOT}/.remote"

die() { echo "error: $*" >&2; exit 1; }

load_conf() {
  [[ -f "$CONF" ]] || die "no .remote config found. Run: scripts/remote.sh init"
  # shellcheck disable=SC1090
  set -a; source "$CONF"; set +a
  : "${REMOTE_HOST:?set REMOTE_HOST in .remote}"
  : "${REMOTE_PORT:?set REMOTE_PORT in .remote}"
  : "${REMOTE_DIR:?set REMOTE_DIR in .remote}"
  : "${REMOTE_VENV:=/venv/main}"
  : "${LOCAL_FWD:=8080}"
}

# ssh/scp with the configured port + sane timeouts
rsh() { ssh -p "$REMOTE_PORT" -o ConnectTimeout=15 -o StrictHostKeyChecking=no "$REMOTE_HOST" "$@"; }
# run a command on the remote, already cd'd into REMOTE_DIR with the venv active
rrun() { rsh "cd '$REMOTE_DIR' && source '$REMOTE_VENV/bin/activate' && $*"; }

EXCLUDES=(
  --exclude '.git/'           --exclude 'processed_data/'
  --exclude 'checkpoints/'    --exclude 'logs/'
  --exclude 'runs/'           --exclude 'hf_cache/'
  --exclude 'remote-results/' --exclude '.pytest_cache/'
  --exclude 'parameter-golf/' --exclude '__pycache__/'
  --exclude '.remote'
)

# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------
cmd_init() {
  if [[ -f "$CONF" ]]; then
    echo ".remote already exists — edit it directly:"; echo "  $CONF"; return 0
  fi
  cp "${REPO_ROOT}/.remote.example" "$CONF"
  echo "created $CONF — edit HOST/PORT for your current Vast box, then:"
  echo "  scripts/remote.sh install"
}

cmd_connect() {  # interactive shell + tmux + port forward
  load_conf
  local session="${1:-remote}"
  ssh -p "$REMOTE_PORT" -L "${LOCAL_FWD}:localhost:${LOCAL_FWD}" "$REMOTE_HOST" \
    "tmux new-session -A -s ${session}"
}

cmd_sync() {  # push local checkout up (code only, no data/checkpoints/logs)
  load_conf
  echo "syncing -> ${REMOTE_HOST}:${REMOTE_DIR}"
  rsh "mkdir -p '$REMOTE_DIR'"
  rsync -az --info=progress2 "${EXCLUDES[@]}" \
    -e "ssh -p ${REMOTE_PORT} -o StrictHostKeyChecking=no" \
    "${REPO_ROOT}/" "${REMOTE_HOST}:${REMOTE_DIR}/"
}

cmd_install() {  # mark safe.directory + pip install requirements in the venv
  load_conf
  rrun "git config --global --add safe.directory '$REMOTE_DIR' || true; \
        python -m pip install -q -r requirements.txt && python -c 'import torch;print(\"torch\",torch.__version__,\"cuda\",torch.cuda.is_available())'"
}

cmd_data() {  # kick off the dataset download in a tmux session
  load_conf
  rrun "mkdir -p logs; tmux new-session -d -s data \
        'cd $REMOTE_DIR && source $REMOTE_VENV/bin/activate && HF_HOME=/workspace/.hf_home python data/download_hf_data.py 2>&1 | tee logs/data_download.log'"
  echo "data download started in tmux 'data'. Follow:  scripts/remote.sh logs data data_download"
}

cmd_launch() {  # launch a training run: launch <config> [name] [-- extra args]
  load_conf
  local config="${1:?usage: launch <config> [name] [-- extra train_llm.py args]}"; shift || true
  local name="run"
  if [[ $# -gt 0 && "$1" != "--" ]]; then name="$1"; shift; fi
  [[ "${1:-}" == "--" ]] && shift || true
  local extra="$*"
  local out="runs/${name}/full" log="logs/${name}.log"
  cmd_sync
  rrun "mkdir -p logs runs; tmux new-session -d -s '${name}' \
        'cd $REMOTE_DIR && source $REMOTE_VENV/bin/activate && \
         python train_llm.py --config ${config} --dataset_path processed_data/pretrain_1B \
         --output_dir ${out} --seed 42 ${extra} 2>&1 | tee ${log}'"
  echo "launched '${config}' in tmux '${name}' -> ${log}"
  echo "watch:  scripts/remote.sh status ${name}"
}

cmd_status() {  # val-loss points + live step + ETA + running state
  load_conf
  local name="${1:-run}" log="logs/${1:-run}.log"
  rrun "
    echo '=== val loss ==='; grep -iE 'val loss' '${log}' 2>/dev/null | tail -6 || echo '(none yet)';
    echo '=== live ===';     tail -3 '${log}' 2>/dev/null | tr '\r' '\n' | grep -iE 'step=' | tail -1;
    echo '=== final ===';    grep -iE 'Final Val Loss' '${log}' 2>/dev/null | tail -1;
    echo '=== proc ===';     pgrep -f train_llm.py >/dev/null && echo RUNNING || echo STOPPED
  "
}

cmd_logs() {  # tail [-f] a run log: logs [name] [logbase]
  load_conf
  local follow=""; [[ "${1:-}" == "-f" ]] && { follow="-f"; shift; }
  local base="${2:-${1:-run}}"
  rrun "tail ${follow} -n 40 logs/${base}.log"
}

cmd_pull() {  # scp a run's metrics.json down: pull [name] [dest]
  load_conf
  local name="${1:-run}"
  local dest="${2:-/tmp/${name}_metrics.json}"
  scp -P "$REMOTE_PORT" -o StrictHostKeyChecking=no \
    "${REMOTE_HOST}:${REMOTE_DIR}/runs/${name}/full/metrics.json" "$dest"
  echo "pulled -> $dest"
  python3 -c "import json;d=json.load(open('$dest'));print('val_loss',d['final_metrics']['val_loss'],'steps',d.get('actual_steps'))" 2>/dev/null || true
}

cmd_promote() {  # pull metrics and overwrite the canonical baseline file
  load_conf
  local name="${1:-run}"
  local target="${REPO_ROOT}/baselines/10m_baseline.json"
  cmd_pull "$name" "$target"
  echo "wrote $target — review, then commit + update LEADERBOARD.md"
}

cmd_ssh() {  # escape hatch: run an arbitrary command on the box (venv active)
  load_conf; rrun "$*"
}

cmd_help() {
  cat <<'EOF'
remote.sh — connect to & control the Vast.ai training box

  init                         create .remote from .remote.example
  connect [session]            ssh in (tmux + localhost:8080 forward)
  sync                         rsync local code up (no data/checkpoints/logs)
  install                      pip install -r requirements.txt in the venv
  data                         start the dataset download (tmux 'data')
  launch <config> [name] [-- extra args]
                               sync + start training in tmux (tee to logs/<name>.log)
  status [name]                val-loss points, live step, final, RUNNING/STOPPED
  logs [-f] [name]             tail (or follow) logs/<name>.log
  pull [name] [dest]           scp runs/<name>/full/metrics.json down
  promote [name]               pull metrics -> overwrite baselines/10m_baseline.json
  ssh <cmd...>                 run any command on the box (venv active)

Config: .remote (copy from .remote.example). Default run name is "run".
Examples:
  scripts/remote.sh launch screen10m screen10m_base
  scripts/remote.sh status screen10m_base
  scripts/remote.sh promote screen10m_base
EOF
}

main() {
  local cmd="${1:-help}"; shift || true
  case "$cmd" in
    init)     cmd_init "$@";;
    connect)  cmd_connect "$@";;
    sync)     cmd_sync "$@";;
    install)  cmd_install "$@";;
    data)     cmd_data "$@";;
    launch)   cmd_launch "$@";;
    status)   cmd_status "$@";;
    logs)     cmd_logs "$@";;
    pull)     cmd_pull "$@";;
    promote)  cmd_promote "$@";;
    ssh)      cmd_ssh "$@";;
    help|-h|--help) cmd_help;;
    *) die "unknown command '$cmd' (try: scripts/remote.sh help)";;
  esac
}

main "$@"
