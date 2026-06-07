#!/usr/bin/env bash
# Greedy token-curve fill: 5M and 15M, sigmoid k=6 vs ctrl. Skips finished runs,
# launches each run as soon as a GPU is free. Safe to run alongside other jobs.
set -eu
REPO_DIR=${REPO_DIR:-/kaggle/working/universe-lm}
RUNS=${RUNS:-/kaggle/working/unet_runs}
DATA=${DATA:-processed_data/speedrun_40M}
cd "$REPO_DIR"
done_already(){ local n=$1 full=$2 f="$RUNS/$1/metrics.json"; [ -f "$f" ]||return 1; local t; t=$(python3 -c "import json;print(json.load(open('$f')).get('tokens_seen',0))" 2>/dev/null||echo 0); [ "${t%.*}" -ge "$full" ]; }
gpu_free(){ /opt/bin/nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n "$(( $1 + 1 ))p"; }
wait_gpu(){ while [ "$(gpu_free $1)" -ge 500 ]; do sleep 15; done; }
launch(){ local g=$1 n=$2 tok=$3; shift 3
  if done_already "$n" "$tok"; then echo "SKIP $n"; return; fi
  wait_gpu "$g"
  CUDA_VISIBLE_DEVICES=$g PYTHONUNBUFFERED=1 nohup python3 -u train_llm.py --config tiny1m --seed 42 "$@" --train_tokens "$tok" --dataset_path "$DATA" --log_every 100 --output_dir "$RUNS/$n" > "$RUNS/$n.log" 2>&1 &
  echo "LAUNCH GPU$g $n (PID=$!)"; sleep 8; }
SIG="--use_unet_skips true --unet_gate_type sigmoid --unet_gate_init -1.5 --unet_skip_count 6"
CTL="--use_unet_skips false"
launch 0 tiny_unet_sigmoid_m15_k6_5m  5000000  $SIG
launch 1 tiny_unet_ctrl_5m            5000000  $CTL
wait
launch 0 tiny_unet_sigmoid_m15_k6_15m 15000000 $SIG
launch 1 tiny_unet_ctrl_15m           15000000 $CTL
wait
echo "=== token curve done ==="
