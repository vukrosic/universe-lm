#!/usr/bin/env bash
# ============================================================================
# tiny1m NORM sweep #6 — 2026-06-04  ·  clean-baseline candidate search
# ----------------------------------------------------------------------------
# Goal: look for a norm that is better than RMSNorm or LayerNorm on the stripped
# baseline, using the most plausible candidates from the earlier sweep.
# We keep the same clean-baseline recipe and try a small p-range plus the
# strongest non-pnorm candidates (manhattan, channelscale, centeredl1).
# ============================================================================
set -u

DATASET="${DATASET:-processed_data/pretrain_1B}"
PYTHON="${PYTHON:-/venv/main/bin/python}"
DATE_TAG="0604"

BASE_FLAGS=(
    --use_value_embed false
    --use_q_gain false
    --use_sliding_window false
    --rope_base 10000
)

run_one() {
    local seed="$1"; shift
    local name="$1"; shift
    echo
    echo "[tiny1m-norm6] === $(date) starting ${name} seed=${seed} ==="
    set +e
    "$PYTHON" train_llm.py \
        --config tiny1m \
        --dataset_path "$DATASET" \
        --output_dir "runs/tiny1m_${DATE_TAG}_norm6_${name}_s${seed}_full" \
        --seed "$seed" \
        "${BASE_FLAGS[@]}" \
        "$@"
    local rc=$?
    set -e
    if [ "$rc" -eq 0 ]; then
        echo "[tiny1m-norm6] === $(date) ${name} seed=${seed} DONE (rc=$rc) ==="
    else
        echo "[tiny1m-norm6] === $(date) ${name} seed=${seed} FAILED (rc=$rc) ==="
    fi
    "$PYTHON" scripts/collect_tiny1m_0604.py || true
}

seed=45
run_one "$seed" "rmsnorm"              --norm_type rmsnorm
run_one "$seed" "layernorm"            --use_layernorm true
run_one "$seed" "manhattan"            --norm_type manhattan
run_one "$seed" "centeredl1"           --norm_type centeredl1
run_one "$seed" "channelscale"         --norm_type channelscale
run_one "$seed" "pnorm1.6"             --norm_type pnorm1.6
run_one "$seed" "pnorm1.7"             --norm_type pnorm1.7
run_one "$seed" "pnorm1.8"             --norm_type pnorm1.8

echo
echo "[tiny1m-norm6] === $(date) tiny norm#6 candidate search complete ==="
