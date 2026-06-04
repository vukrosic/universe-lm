#!/usr/bin/env bash
# ============================================================================
# tiny1m arch sweep #3 — queued 2026-06-04
# ----------------------------------------------------------------------------
# Chained after arch#2. Extends the SwiGLU-baseline search with cheap,
# no-code flag axes plus follow-ups on the arch#2 surprises:
#   - glob4 diverged (7.26) while glob3 tied baseline -> rerun glob4 +
#     add glob2 (heaviest global) to map the #86 axis properly.
#   - RoPE-base, window-size, norm, softcap, and stack follow-ups.
# Run dirs are date-stamped tiny1m_0604_* to match arch#2.
#
# Preamble blocks until the arch#2 queue reports complete, so the two
# never contend for the GPU.
# ============================================================================
set -u

DATASET="${DATASET:-processed_data/pretrain_1B}"
SEED="${SEED:-42}"
PYTHON="${PYTHON:-python}"
DATE_TAG="0604"

# --- wait for arch#2 to finish (max ~40 min guard) ---
echo "[tiny1m-arch3] waiting for arch#2 to complete..."
for _ in $(seq 1 240); do
    if grep -q "tiny arch#2 queue complete" /root/tiny1m_arch2.log 2>/dev/null; then
        echo "[tiny1m-arch3] arch#2 done, starting arch#3"
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
    echo "[tiny1m-arch3] === $(date) starting $name ==="
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
        echo "[tiny1m-arch3] === $(date) $name DONE (rc=$rc) ==="
    else
        echo "[tiny1m-arch3] === $(date) $name FAILED (rc=$rc) ==="
    fi
}

# #86 global-attention follow-ups: confirm the glob4 divergence + map heavier.
run_one "swiglu_glob4_rerun" --global_attn_every_k 4
run_one "swiglu_glob2"       --global_attn_every_k 2
run_one "swiglu_glob3_tiedqk" --global_attn_every_k 3 --use_tied_qk true

# RoPE-base sweep on the new SwiGLU baseline (arch#1 used 250k).
run_one "swiglu_rope500k" --rope_base 500000
run_one "swiglu_rope1m"   --rope_base 1000000

# Window-size sweep on the SwiGLU baseline (arch#1 used 384).
run_one "swiglu_swa256" --sliding_window_size 256
run_one "swiglu_swa512" --sliding_window_size 512

# Norm / logit / stack follow-ups.
run_one "swiglu_layernorm"  --use_layernorm true
run_one "swiglu_softcap15"  --logit_softcap 15.0
run_one "swiglu_tiedqk_mha" --use_tied_qk true --n_kv_heads 4

echo
echo "[tiny1m-arch3] === $(date) tiny arch#3 queue complete ==="
