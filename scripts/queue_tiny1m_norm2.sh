#!/usr/bin/env bash
# ============================================================================
# tiny1m NORM study #2 — queued 2026-06-04  ·  GENERALIZED p-NORM SWEEP (#90)
# ----------------------------------------------------------------------------
# Derived from norm study #1's logic: a good norm divides by a smooth ALL-dim
# aggregate (scale-invariance), and lower-order/outlier-robust statistics
# (L1 > L2 > L-inf) seemed to help. So sweep the single knob `p` through one
# code path (PNorm), plus the L1-LayerNorm. Single seed (42), 546k base.
#   pnorm0.5  more outlier-robust than L1
#   pnorm1    = ManhattanNorm (the winner)
#   pnorm1.5  between L1 and L2
#   pnorm2    = RMSNorm (built-in sanity check — must match)
#   pnorm3    toward L-inf (should be worse)
#   centeredl1  L1 analogue of LayerNorm
#   rmsnorm   native baseline anchor
# ============================================================================
set -u

DATASET="${DATASET:-processed_data/pretrain_1B}"
PYTHON="${PYTHON:-python}"
DATE_TAG="0604"

BASE_FLAGS=(
    --use_value_embed true
    --use_q_gain true
    --use_sliding_window true
    --sliding_window_size 384
    --rope_base 250000
)

run_norm() {
    local nt="$1"
    echo
    echo "[tiny1m-norm2] === $(date) starting ${nt}_s42 ==="
    set +e
    "$PYTHON" train_llm.py \
        --config tiny1m \
        --dataset_path "$DATASET" \
        --output_dir "runs/tiny1m_${DATE_TAG}_norm2_${nt}_s42_full" \
        --seed 42 \
        "${BASE_FLAGS[@]}" \
        --norm_type "$nt"
    local rc=$?
    set -e
    if [ "$rc" -eq 0 ]; then
        echo "[tiny1m-norm2] === $(date) ${nt}_s42 DONE (rc=$rc) ==="
    else
        echo "[tiny1m-norm2] === $(date) ${nt}_s42 FAILED (rc=$rc) ==="
    fi
    "$PYTHON" scripts/collect_tiny1m_0604.py || true
}

for NT in rmsnorm pnorm2 pnorm1 pnorm1.5 pnorm0.5 pnorm3 centeredl1; do
    run_norm "$NT"
done

echo
echo "[tiny1m-norm2] === $(date) tiny norm study #2 complete ==="
