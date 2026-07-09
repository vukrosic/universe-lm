#!/usr/bin/env bash
# ============================================================================
# tiny1m sweep #11 — 2026-06-04  ·  LEARNING-RATE SWEEP (highest leverage)
# ----------------------------------------------------------------------------
# Architecture gave +-0.02-0.05. The optimizer LR is untouched and usually
# moves loss MORE in an undertrained regime. Sweep Muon LR on the pnorm1.5
# champion. Single seed 42. No rmsnorm control (champion = the anchor).
# Default muon_lr = 0.024. Chained after #10.
# ============================================================================
set -u

DATASET="${DATASET:-processed_data/pretrain_1B}"
PYTHON="${PYTHON:-python}"
DATE_TAG="0604"

echo "[tiny1m-arch11] waiting for arch#10 to complete..."
for _ in $(seq 1 900); do
    if grep -q "tiny arch#10 complete" /root/tiny1m_arch10.log 2>/dev/null; then
        echo "[tiny1m-arch11] arch#10 done, starting arch#11"
        break
    fi
    sleep 10
done

BASE_FLAGS=(
    --use_value_embed true
    --use_q_gain true
    --use_sliding_window true
    --sliding_window_size 384
    --rope_base 250000
    --norm_type pnorm1.5
)

run_lr() {
    local lr="$1"
    local name="muonlr${lr/./p}"
    echo
    echo "[tiny1m-arch11] === $(date) starting $name (muon_lr=$lr) ==="
    set +e
    "$PYTHON" train_llm.py \
        --config tiny1m \
        --dataset_path "$DATASET" \
        --output_dir "runs/tiny1m_${DATE_TAG}_arch11_${name}_s42_full" \
        --seed 42 \
        "${BASE_FLAGS[@]}" \
        --muon_lr "$lr"
    local rc=$?
    set -e
    if [ "$rc" -eq 0 ]; then
        echo "[tiny1m-arch11] === $(date) $name DONE (rc=$rc) ==="
    else
        echo "[tiny1m-arch11] === $(date) $name FAILED (rc=$rc) ==="
    fi
    "$PYTHON" scripts/collect_tiny1m_0604.py || true
}

for LR in 0.012 0.018 0.024 0.032 0.040 0.048; do
    run_lr "$LR"
done

echo
echo "[tiny1m-arch11] === $(date) tiny arch#11 complete ==="
