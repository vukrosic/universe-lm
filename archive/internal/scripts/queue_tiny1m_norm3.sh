#!/usr/bin/env bash
# ============================================================================
# tiny1m NORM study #3 — 2026-06-04  ·  OUTLIER-HANDLING MECHANISMS (#94-96)
# ----------------------------------------------------------------------------
# Diagnosis: massive-activation channels dominate the L2 denominator and live
# in the wide residual stream. Test three DIFFERENT ways to handle them (not a
# p-sweep), single seed 42, 546k base:
#   clipnorm3      winsorize: clip |x| to 3*mean|x| then RMS (remove outliers)
#   channelscale   learnable per-channel pre-scale then RMS (suppress channels)
#   median         divide by median|x| (maximal robustness)
# Anchors: rmsnorm (control) and pnorm1.5 (current winner) in the same batch.
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
    echo "[tiny1m-norm3] === $(date) starting ${nt}_s42 ==="
    set +e
    "$PYTHON" train_llm.py \
        --config tiny1m \
        --dataset_path "$DATASET" \
        --output_dir "runs/tiny1m_${DATE_TAG}_norm3_${nt}_s42_full" \
        --seed 42 \
        "${BASE_FLAGS[@]}" \
        --norm_type "$nt"
    local rc=$?
    set -e
    if [ "$rc" -eq 0 ]; then
        echo "[tiny1m-norm3] === $(date) ${nt}_s42 DONE (rc=$rc) ==="
    else
        echo "[tiny1m-norm3] === $(date) ${nt}_s42 FAILED (rc=$rc) ==="
    fi
    "$PYTHON" scripts/collect_tiny1m_0604.py || true
}

for NT in rmsnorm pnorm1.5 clipnorm3 channelscale median; do
    run_norm "$NT"
done

echo
echo "[tiny1m-norm3] === $(date) tiny norm study #3 complete ==="
