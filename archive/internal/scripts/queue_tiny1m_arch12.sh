#!/usr/bin/env bash
# ============================================================================
# tiny1m sweep #12 — 2026-06-04  ·  LR GRID ON THE BEST ARCHITECTURE
# ----------------------------------------------------------------------------
# arch11 sweeps LR on pnorm1.5 alone; this sweeps LR on the best architecture
# (pnorm1.5 + multiscale heads) to see if the optimal LR shifts with the
# stronger model, and probes AdamW LR too. Single seed 42. Chained after #11.
# ============================================================================
set -u

DATASET="${DATASET:-processed_data/pretrain_1B}"
PYTHON="${PYTHON:-python}"
DATE_TAG="0604"

echo "[tiny1m-arch12] waiting for norm#4 to complete..."
for _ in $(seq 1 1800); do
    if grep -q "tiny norm#4 complete" /root/tiny1m_norm4.log 2>/dev/null; then
        echo "[tiny1m-arch12] norm#4 done, starting arch#12"
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
    --use_multiscale_heads true
)

run_one() {
    local name="$1"; shift
    echo
    echo "[tiny1m-arch12] === $(date) starting $name ==="
    set +e
    "$PYTHON" train_llm.py \
        --config tiny1m \
        --dataset_path "$DATASET" \
        --output_dir "runs/tiny1m_${DATE_TAG}_arch12_${name}_s42_full" \
        --seed 42 \
        "${BASE_FLAGS[@]}" \
        "$@"
    local rc=$?
    set -e
    if [ "$rc" -eq 0 ]; then
        echo "[tiny1m-arch12] === $(date) $name DONE (rc=$rc) ==="
    else
        echo "[tiny1m-arch12] === $(date) $name FAILED (rc=$rc) ==="
    fi
    "$PYTHON" scripts/collect_tiny1m_0604.py || true
}

run_one "muonlr0p018"  --muon_lr 0.018
run_one "muonlr0p024"  --muon_lr 0.024
run_one "muonlr0p032"  --muon_lr 0.032
run_one "muonlr0p040"  --muon_lr 0.040
run_one "adamwlr0p004" --adamw_lr 0.004
run_one "adamwlr0p009" --adamw_lr 0.009

echo
echo "[tiny1m-arch12] === $(date) tiny arch#12 complete ==="
