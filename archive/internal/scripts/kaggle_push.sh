#!/usr/bin/env bash
# Push the universe-lm sweep to Kaggle as a headless GPU script kernel,
# then poll / pull results. No SSH, no notebook UI.
#
# One-time setup:
#   pip install kaggle
#   # Kaggle.com -> Settings -> API -> Create New Token  (saves kaggle.json)
#   mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
#   # edit kaggle_job/kernel-metadata.json: set "id" to YOUR_KAGGLE_USERNAME/universe-lm-sweep
#
# Usage:
#   bash scripts/kaggle_push.sh          # push + start the run
#   bash scripts/kaggle_push.sh status   # poll status
#   bash scripts/kaggle_push.sh pull     # download outputs to ./remote-results/kaggle
#   bash scripts/kaggle_push.sh logs     # print the kernel log
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOB_DIR="$ROOT/kaggle_job"
META="$JOB_DIR/kernel-metadata.json"
OUT="$ROOT/remote-results/kaggle"

command -v kaggle >/dev/null || { echo "kaggle CLI missing -> pip install kaggle"; exit 1; }
ID="$(python3 -c "import json;print(json.load(open('$META'))['id'])")"
[ "$ID" = "YOUR_KAGGLE_USERNAME/universe-lm-sweep" ] && {
  echo "Edit $META: replace YOUR_KAGGLE_USERNAME with your Kaggle username."; exit 1; }

case "${1:-push}" in
  push)
    echo ">> pushing $ID (GPU, internet on) ..."
    kaggle kernels push -p "$JOB_DIR"
    echo ">> running. poll with: bash scripts/kaggle_push.sh status"
    ;;
  status)
    kaggle kernels status "$ID"
    ;;
  logs)
    kaggle kernels output "$ID" -p /tmp/kaggle_log >/dev/null
    cat /tmp/kaggle_log/*.log 2>/dev/null || echo "no log yet"
    ;;
  pull)
    mkdir -p "$OUT"
    kaggle kernels output "$ID" -p "$OUT"
    echo ">> results in $OUT"
    find "$OUT" -name 'metrics.json' | sort
    ;;
  *)
    echo "usage: kaggle_push.sh [push|status|logs|pull]"; exit 1 ;;
esac
