#!/usr/bin/env bash
# 20M-token scale confirmation: sigmoid k=6 vs no-skip control.
set -eu
REPO_DIR=${REPO_DIR:-/kaggle/working/universe-lm}
RUNS=${RUNS:-/kaggle/working/unet_runs}
DATA=${DATA:-processed_data/speedrun_40M}
cd "$REPO_DIR"
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python3 -u train_llm.py --config tiny1m --seed 42 \
  --use_unet_skips true --unet_gate_type sigmoid --unet_gate_init -1.5 --unet_skip_count 6 \
  --train_tokens 20000000 --dataset_path "$DATA" --log_every 100 \
  --output_dir "$RUNS/tiny_unet_sigmoid_m15_k6_20m" > "$RUNS/tiny_unet_sigmoid_m15_k6_20m.log" 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python3 -u train_llm.py --config tiny1m --seed 42 \
  --use_unet_skips false \
  --train_tokens 20000000 --dataset_path "$DATA" --log_every 100 \
  --output_dir "$RUNS/tiny_unet_ctrl_20m" > "$RUNS/tiny_unet_ctrl_20m.log" 2>&1 &
P2=$!
wait $P1 $P2
echo "=== 20M done ==="
