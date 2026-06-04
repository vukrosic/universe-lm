#!/usr/bin/env bash
# ============================================================================
# tiny1m arch sweep #2 — queued 2026-06-04
# ----------------------------------------------------------------------------
# Built on the arch#1 result: SwiGLU was the single-axis winner (6.2869),
# beating tied-QK (6.3041) and the V+q+SWA384+RoPE250k baseline (6.3350).
#
# This batch does two things:
#   1. Re-anchors the new baseline = old base flags + SwiGLU.
#   2. Tests the new #86 axis: interleaved global attention (DeepSeek-V4
#      hybrid-attention analog). The stack is all-SWA(384); making every
#      k-th layer full causal attention adds a cheap periodic "global"
#      layer (V4 HCA-style global context on top of local context).
#   3. Stacks the confirmed winners (SwiGLU + tied-QK + full MHA).
#
# Run dirs are date-stamped tiny1m_0604_* so the archive is self-dating.
# ============================================================================
set -u

DATASET="${DATASET:-processed_data/pretrain_1B}"
SEED="${SEED:-42}"
PYTHON="${PYTHON:-python}"
DATE_TAG="0604"

# New baseline: arch#1 base flags + the SwiGLU winner.
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
    echo "[tiny1m-arch2] === $(date) starting $name ==="
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
        echo "[tiny1m-arch2] === $(date) $name DONE (rc=$rc) ==="
    else
        echo "[tiny1m-arch2] === $(date) $name FAILED (rc=$rc) ==="
    fi
}

# Re-anchor the SwiGLU baseline (sanity: should land near 6.2869).
run_one "swiglu_base"

# #86 interleaved global attention sweep on the SwiGLU baseline.
# tiny1m has 12 layers, so k=3 -> 4 global layers, k=4 -> 3, k=6 -> 2.
run_one "swiglu_glob3" --global_attn_every_k 3
run_one "swiglu_glob4" --global_attn_every_k 4
run_one "swiglu_glob6" --global_attn_every_k 6

# Winner stacks (orthogonal axes: FFN vs QK-share vs KV-head count).
run_one "swiglu_tiedqk" --use_tied_qk true
run_one "swiglu_tiedqk_glob4" --use_tied_qk true --global_attn_every_k 4
run_one "swiglu_mha" --n_kv_heads 4

echo
echo "[tiny1m-arch2] === $(date) tiny arch#2 queue complete ==="
