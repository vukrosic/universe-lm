#!/usr/bin/env bash
# ============================================================================
# tiny1m arch sweep #4 — queued 2026-06-04  ·  NEW ATTENTION MECHANISMS
# ----------------------------------------------------------------------------
# Three big-lab attention mechanisms coded from scratch and adapted to the
# tiny (0.94M, d_k=16) regime, all on the SwiGLU baseline:
#   #87 Differential Attention  (Microsoft DIFF Transformer)
#   #88 NSA compressed-global   (DeepSeek Native Sparse Attention)
#   #89 Hybrid heads            (DeepSeek-V4 hybrid attn, head granularity)
#
# Chained after arch#3 (waits on its completion marker — no GPU contention).
# Date-stamped tiny1m_0604_* to match the rest of today's batch.
# ============================================================================
set -u

DATASET="${DATASET:-processed_data/pretrain_1B}"
SEED="${SEED:-42}"
PYTHON="${PYTHON:-python}"
DATE_TAG="0604"

echo "[tiny1m-arch4] waiting for arch#3 to complete..."
for _ in $(seq 1 480); do
    if grep -q "tiny arch#3 queue complete" /root/tiny1m_arch3.log 2>/dev/null; then
        echo "[tiny1m-arch4] arch#3 done, starting arch#4"
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
    --ffn_variant swiglu
)

run_one() {
    local name="$1"
    shift
    echo
    echo "[tiny1m-arch4] === $(date) starting $name ==="
    set +e
    "$PYTHON" train_llm.py \
        --config tiny1m \
        --dataset_path "$DATASET" \
        --output_dir "runs/tiny1m_${DATE_TAG}_${name}_full" \
        --seed "$SEED" \
        "${BASE_FLAGS[@]}" \
        "$@"
    local rc=$?
    set -e
    if [ "$rc" -eq 0 ]; then
        echo "[tiny1m-arch4] === $(date) $name DONE (rc=$rc) ==="
    else
        echo "[tiny1m-arch4] === $(date) $name FAILED (rc=$rc) ==="
    fi
}

# #87 Differential Attention.
run_one "swiglu_diffattn" --use_diff_attn true

# #88 NSA compressed-global — block-size sweep (32 / 64 / 128).
run_one "swiglu_nsa32"  --use_nsa_global true --nsa_block 32
run_one "swiglu_nsa64"  --use_nsa_global true --nsa_block 64
run_one "swiglu_nsa128" --use_nsa_global true --nsa_block 128

# #89 Hybrid heads — default window (384) and a tighter local window (256).
run_one "swiglu_hybridheads"      --use_hybrid_heads true
run_one "swiglu_hybridheads_w256" --use_hybrid_heads true --sliding_window_size 256

echo
echo "[tiny1m-arch4] === $(date) tiny arch#4 queue complete ==="
