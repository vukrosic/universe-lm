#!/usr/bin/env bash
# ============================================================================
# tiny1m arch sweep #8 — 2026-06-04  ·  NON-NORM ARCHITECTURE TOPOLOGY
# ----------------------------------------------------------------------------
# Branch away from normalization into different STRUCTURE, single seed 42:
#   #97 multiscale heads  - per-head graded windows (96/192/384/768) vs uniform
#   #98 parallel block    - PaLM/GPT-J: attn+FFN share one norm, sum to residual
#   layerscale            - learnable per-channel residual gate (zero-init)
# In-batch control (rmsnorm, sequential) for a same-conditions anchor.
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

run_one() {
    local name="$1"; shift
    echo
    echo "[tiny1m-arch8] === $(date) starting $name ==="
    set +e
    "$PYTHON" train_llm.py \
        --config tiny1m \
        --dataset_path "$DATASET" \
        --output_dir "runs/tiny1m_${DATE_TAG}_arch8_${name}_s42_full" \
        --seed 42 \
        "${BASE_FLAGS[@]}" \
        "$@"
    local rc=$?
    set -e
    if [ "$rc" -eq 0 ]; then
        echo "[tiny1m-arch8] === $(date) $name DONE (rc=$rc) ==="
    else
        echo "[tiny1m-arch8] === $(date) $name FAILED (rc=$rc) ==="
    fi
    "$PYTHON" scripts/collect_tiny1m_0604.py || true
}

run_one "control"
run_one "multiscale" --use_multiscale_heads true
run_one "parallel"   --use_parallel_block true
run_one "layerscale" --use_layerscale true

echo
echo "[tiny1m-arch8] === $(date) tiny arch#8 complete ==="
