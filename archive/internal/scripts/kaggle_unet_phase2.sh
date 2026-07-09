#!/usr/bin/env bash
#
# kaggle_unet_phase2.sh — sharpen the U-Net skip story with two clean questions.
#
# Phase 1 + the redo (kaggle_unet_phase1_redo.sh) left two things unproven:
#   1. Bound vs init: sigmoid(-1.5) and raw(0.18) START at the same effective
#      weight (~0.18). At k=6 sigmoid already wins. Does that hold at k=2, k=4?
#      We have sigmoid_m15 k2/k4 and raw0 k2/k4 from the redo; this adds the
#      MATCHED-INIT raw runs (raw, init 0.18) at k2/k4 so the only difference
#      vs sigmoid is the [0,1] bound, not the start point.
#   2. Scale: every prior run is tiny1m @ 3M tokens. Re-run the winner
#      (sigmoid_m15, k6) and the control at 10M tokens — same tiny arch, only
#      train_tokens changes — to see if the ~0.046 val_loss gap survives.
#
# Pairs experiments across the two T4s (CUDA_VISIBLE_DEVICES=0/1) and skips
# runs whose metrics.json already shows the full token count, so re-runs resume.
#
# Usage on the Kaggle SSH box (LD_LIBRARY_PATH must point at the driver libs):
#   export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
#   REPO_DIR=/kaggle/working/universe-lm bash scripts/kaggle_unet_phase2.sh
#
# Environment overrides (optional):
#   REPO_DIR  — default /kaggle/working/universe-lm
#   RUNS      — default /kaggle/working/unet_runs   (shared with phase 1 redo)
#   DATA      — default processed_data/speedrun_40M

set -eu

REPO_DIR=${REPO_DIR:-/kaggle/working/universe-lm}
RUNS=${RUNS:-/kaggle/working/unet_runs}
DATA=${DATA:-processed_data/speedrun_40M}

cd "$REPO_DIR"
mkdir -p "$RUNS"

# done_already <name> <full_tokens> — true if the run wrote a full metrics.json.
done_already() {
  local name=$1 full=$2
  local f="$RUNS/$name/metrics.json"
  [ -f "$f" ] || return 1
  local tok
  tok=$(python3 -c "import json; print(json.load(open('$f')).get('tokens_seen', 0))" 2>/dev/null || echo 0)
  [ "${tok%.*}" -ge "$full" ]
}

# launch <gpu> <name> <full_tokens> <extra-train-args...>
launch() {
  local gpu=$1 name=$2 full=$3; shift 3
  if done_already "$name" "$full"; then
    echo "SKIP   $name  (already finished, tokens >= $full)"
    return
  fi
  CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 nohup python3 -u train_llm.py --config tiny1m --seed 42 "$@" --dataset_path "$DATA" --log_every 25 --output_dir "$RUNS/$name" > "$RUNS/$name.log" 2>&1 &
  echo "LAUNCH GPU$gpu  $name  (PID=$!)"
}

echo "=== U-Net Phase 2 — bound-vs-init (matched) + scale check ==="
echo "Repo:  $REPO_DIR"
echo "Runs:  $RUNS"
echo "Data:  $DATA"
echo

# ----- Pair 1: matched-init raw (init 0.18) at k=2 and k=4, 3M tokens -----
launch 0 tiny_unet_raw018_k2 3000000 --use_unet_skips true --unet_gate_type raw --unet_gate_init 0.18 --unet_skip_count 2
launch 1 tiny_unet_raw018_k4 3000000 --use_unet_skips true --unet_gate_type raw --unet_gate_init 0.18 --unet_skip_count 4
wait
echo "=== pair 1 done (matched-init raw) ==="

# ----- Pair 2: scale check at 10M tokens (winner vs control), k=6 -----
launch 0 tiny_unet_sigmoid_m15_k6_10m 10000000 --use_unet_skips true --unet_gate_type sigmoid --unet_gate_init -1.5 --unet_skip_count 6 --train_tokens 10000000
launch 1 tiny_unet_ctrl_10m           10000000 --use_unet_skips false --train_tokens 10000000
wait
echo "=== pair 2 done (scale check) ==="

echo
echo "=== RESULTS (this phase) ==="
printf '%-34s %12s %12s\n' "run" "val_loss" "tokens"
printf '%-34s %12s %12s\n' "---" "--------" "------"
for name in tiny_unet_raw018_k2 tiny_unet_raw018_k4 tiny_unet_sigmoid_m15_k6_10m tiny_unet_ctrl_10m; do
  d="$RUNS/$name"
  if [ -f "$d/metrics.json" ]; then
    val=$(python3 -c "import json; m=json.load(open('$d/metrics.json'))['final_metrics']; print(f\"{m['val_loss']:.4f}\")" 2>/dev/null || echo '?')
    tok=$(python3 -c "import json; print(json.load(open('$d/metrics.json')).get('tokens_seen', '?'))" 2>/dev/null || echo '?')
    printf '%-34s %12s %12s\n' "$name" "$val" "$tok"
  fi
done
