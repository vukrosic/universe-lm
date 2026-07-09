#!/usr/bin/env bash
# kaggle_unet_kcurve.sh — fill the sigmoid U-Net skip-count curve (k=1,3,5).
# Completes k=1..6 (have 2,4,6) to settle whether skip-count is monotonic.
# tiny1m @ 3M, seed 42, sigmoid gate init -1.5. Writes to shared unet_runs.
set -eu
REPO_DIR=${REPO_DIR:-/kaggle/working/universe-lm}
RUNS=${RUNS:-/kaggle/working/unet_runs}
DATA=${DATA:-processed_data/speedrun_40M}
FULL=3000000
cd "$REPO_DIR"; mkdir -p "$RUNS"
done_already(){ local f="$RUNS/$1/metrics.json"; [ -f "$f" ] || return 1; local t; t=$(python3 -c "import json;print(json.load(open('$f')).get('tokens_seen',0))" 2>/dev/null||echo 0); [ "${t%.*}" -ge "$FULL" ]; }
launch(){ local g=$1 n=$2; shift 2; if done_already "$n"; then echo "SKIP $n"; return; fi; CUDA_VISIBLE_DEVICES=$g PYTHONUNBUFFERED=1 nohup python3 -u train_llm.py --config tiny1m --seed 42 --use_unet_skips true --unet_gate_type sigmoid --unet_gate_init -1.5 "$@" --dataset_path "$DATA" --log_every 50 --output_dir "$RUNS/$n" > "$RUNS/$n.log" 2>&1 & echo "LAUNCH GPU$g $n (PID=$!)"; }
echo "=== sigmoid k-curve fill: k=1,3,5 ==="
launch 0 tiny_unet_sigmoid_m15_k1 --unet_skip_count 1
launch 1 tiny_unet_sigmoid_m15_k3 --unet_skip_count 3
wait; echo "--- pair 1 done ---"
launch 0 tiny_unet_sigmoid_m15_k5 --unet_skip_count 5
wait; echo "--- done ---"
