#!/usr/bin/env bash
# ============================================================================
# tiny1m NORM sweep #5 — 2026-06-04  ·  clean-baseline paired-seed check
# ----------------------------------------------------------------------------
# norm4 found that clean full-attention baseline results differ from the richer
# stack: LayerNorm won, pnorm1.75 was the best plain p-norm, and QK/V placement
# helped. This reruns the key contenders on seeds 43 and 44 so we can separate
# real norm signal from single-seed noise.
# ============================================================================
set -u

DATASET="${DATASET:-processed_data/pretrain_1B}"
PYTHON="${PYTHON:-python}"
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
    echo "[tiny1m-norm5] === $(date) starting ${name} seed=${seed} ==="
    set +e
    "$PYTHON" train_llm.py \
        --config tiny1m \
        --dataset_path "$DATASET" \
        --output_dir "runs/tiny1m_${DATE_TAG}_norm5_${name}_s${seed}_full" \
        --seed "$seed" \
        "${BASE_FLAGS[@]}" \
        "$@"
    local rc=$?
    set -e
    if [ "$rc" -eq 0 ]; then
        echo "[tiny1m-norm5] === $(date) ${name} seed=${seed} DONE (rc=$rc) ==="
    else
        echo "[tiny1m-norm5] === $(date) ${name} seed=${seed} FAILED (rc=$rc) ==="
    fi
    "$PYTHON" scripts/collect_tiny1m_0604.py || true
}

for seed in 43 44; do
    run_one "$seed" "rmsnorm"              --norm_type rmsnorm
    run_one "$seed" "layernorm"            --use_layernorm true
    run_one "$seed" "pnorm1.75"            --norm_type pnorm1.75
    run_one "$seed" "pnorm1.5"             --norm_type pnorm1.5
    run_one "$seed" "body_qk_pnorm15"      --norm_type pnorm1.5 --qk_norm_type pnorm1.5
    run_one "$seed" "body_v_pnorm15"       --norm_type pnorm1.5 --v_norm_type pnorm1.5
done

echo
echo "[tiny1m-norm5] === $(date) tiny norm#5 complete ==="
