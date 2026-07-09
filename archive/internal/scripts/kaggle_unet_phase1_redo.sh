#!/usr/bin/env bash
#
# kaggle_unet_phase1_redo.sh — Fill in the U-Net skip-count ablation.
#
# The original kaggle_unet_phase1.sh (commit 80040e9) ran the k=2 and k=4 raw
# runs WITHOUT passing --unet_skip_count, so they all used the default (k=6,
# full U). Three runs ended up bit-identical to the k=6 run. This script
# actually passes --unet_skip_count to fill in the missing k=2, k=4 data
# points, with both sigmoid_m15 and raw0 families.
#
# Pairs experiments across the two T4 GPUs (CUDA_VISIBLE_DEVICES=0/1) and
# skips runs whose metrics.json already shows a full token count, so re-runs
# resume cleanly after partial completion.
#
# Usage on Kaggle (single cell):
#   !bash /kaggle/working/universe-lm/scripts/kaggle_unet_phase1_redo.sh
#
# Environment overrides (optional):
#   REPO_DIR  — default /kaggle/working/universe-lm
#   RUNS      — default /kaggle/working/unet_runs
#   DATA      — default processed_data/speedrun_40M
#
# Phase 1 redo covers (4 new runs):
#   sigmoid_m15, k=2 and k=4 — does the sigmoid win survive at smaller k?
#   raw0,      k=2 and k=4 — does the k=2 dip seen in raw0_k2_real persist,
#                            and does raw0_k4 sit between k=2 and k=6?

set -eu

REPO_DIR=${REPO_DIR:-/kaggle/working/universe-lm}
RUNS=${RUNS:-/kaggle/working/unet_runs}
DATA=${DATA:-processed_data/speedrun_40M}
FULL_TOKENS=3000000   # tiny1m runs to ~3M tokens

cd "$REPO_DIR"
mkdir -p "$RUNS"

# done_already <name> — return 0 (true) if the run already wrote a full metrics.json.
done_already() {
  local name=$1
  local f="$RUNS/$name/metrics.json"
  [ -f "$f" ] || return 1
  local tok
  tok=$(python3 -c "import json; print(json.load(open('$f')).get('tokens_seen', 0))" 2>/dev/null || echo 0)
  [ "${tok%.*}" -ge "$FULL_TOKENS" ]
}

# launch <gpu_index> <run_name> <extra-train-args...>
launch() {
  local gpu=$1; local name=$2; shift 2
  if done_already "$name"; then
    echo "SKIP   $name  (already finished, tokens >= $FULL_TOKENS)"
    return
  fi
  # Single physical line for the python invocation — avoids backslash-continuation
  # parsing bugs that bit us when pasted into %%bash cells.
  CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 nohup python3 -u train_llm.py --config tiny1m --seed 42 --use_unet_skips true "$@" --dataset_path "$DATA" --log_every 25 --output_dir "$RUNS/$name" > "$RUNS/$name.log" 2>&1 &
  echo "LAUNCH GPU$gpu  $name  (PID=$!)"
}

echo "=== U-Net Phase 1 redo — fill skip-count ablation ==="
echo "Repo:  $REPO_DIR"
echo "Runs:  $RUNS"
echo "Data:  $DATA"
echo

# ----- Pair 1: sigmoid_m15 at k=2 and k=4 -----
launch 0 tiny_unet_sigmoid_m15_k2 --unet_gate_type sigmoid --unet_gate_init -1.5 --unet_skip_count 2
launch 1 tiny_unet_sigmoid_m15_k4 --unet_gate_type sigmoid --unet_gate_init -1.5 --unet_skip_count 4
wait
echo "=== pair 1 done ==="

# ----- Pair 2: raw0 at k=2 and k=4 (default init 0.0) -----
launch 0 tiny_unet_raw0_k2 --unet_gate_type raw --unet_skip_count 2
launch 1 tiny_unet_raw0_k4 --unet_gate_type raw --unet_skip_count 4
wait
echo "=== pair 2 done ==="

echo
echo "=== RESULTS ==="
printf '%-32s %12s %12s\n' "run" "val_loss" "tokens"
printf '%-32s %12s %12s\n' "---" "--------" "------"
for d in "$RUNS"/*/; do
  name=$(basename "$d")
  if [ -f "$d/metrics.json" ]; then
    val=$(python3 -c "import json; m=json.load(open('$d/metrics.json'))['final_metrics']; print(f\"{m['val_loss']:.4f}\")" 2>/dev/null || echo '?')
    tok=$(python3 -c "import json; print(json.load(open('$d/metrics.json')).get('tokens_seen', '?'))" 2>/dev/null || echo '?')
    printf '%-32s %12s %12s\n' "$name" "$val" "$tok"
  fi
done
