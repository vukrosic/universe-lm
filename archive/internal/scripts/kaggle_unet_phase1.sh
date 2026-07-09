#!/usr/bin/env bash
#
# kaggle_unet_phase1.sh — Run remaining U-Net Phase 1 ablations on Kaggle 2x T4.
#
# Pairs experiments across the two T4 GPUs (CUDA_VISIBLE_DEVICES=0/1) and
# skips runs whose metrics.json already shows a full token count, so re-runs
# resume cleanly after partial completion.
#
# Usage on Kaggle (single cell):
#   !bash /kaggle/working/universe-lm/scripts/kaggle_unet_phase1.sh
#
# Environment overrides (optional):
#   REPO_DIR  — default /kaggle/working/universe-lm
#   RUNS      — default /kaggle/working/unet_runs
#   DATA      — default processed_data/speedrun_40M
#
# Outputs:
#   $RUNS/<name>/metrics.json   — final metrics + history
#   $RUNS/<name>.log            — stdout/stderr of the run
#
# Phase 1 covers:
#   Batch A — gate parameterization
#     raw 0.18, sigmoid -1.5, sigmoid -3.0
#   Batch B — real skip-count sweep (now that --unet_skip_count is wired)
#     k=2
#   (ctrl and raw 0.0 / k=6 are expected to already exist from earlier runs.)

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

echo "=== U-Net Phase 1 remaining-ablation queue ==="
echo "Repo:  $REPO_DIR"
echo "Runs:  $RUNS"
echo "Data:  $DATA"
echo

# ----- Pair 1: gate parameterization headline -----
launch 0 tiny_unet_raw018       --unet_gate_type raw     --unet_gate_init 0.18
launch 1 tiny_unet_sigmoid_m15  --unet_gate_type sigmoid --unet_gate_init -1.5
wait
echo "=== pair 1 done ==="

# ----- Pair 2: lower sigmoid + real skip-count sweep -----
launch 0 tiny_unet_sigmoid_m30  --unet_gate_type sigmoid --unet_gate_init -3.0
launch 1 tiny_unet_raw0_k2_real --unet_skip_count 2
wait
echo "=== pair 2 done ==="

echo
echo "=== RESULTS ==="
printf '%-28s %12s %12s\n' "run" "val_loss" "tokens"
printf '%-28s %12s %12s\n' "---" "--------" "------"
for d in "$RUNS"/*/; do
  name=$(basename "$d")
  if [ -f "$d/metrics.json" ]; then
    val=$(python3 -c "import json; m=json.load(open('$d/metrics.json'))['final_metrics']; print(f\"{m['val_loss']:.4f}\")" 2>/dev/null || echo '?')
    tok=$(python3 -c "import json; print(json.load(open('$d/metrics.json')).get('tokens_seen', '?'))" 2>/dev/null || echo '?')
    printf '%-28s %12s %12s\n' "$name" "$val" "$tok"
  fi
done
