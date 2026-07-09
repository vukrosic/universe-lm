#!/usr/bin/env bash
set -u

DATASET="${DATASET:-processed_data/pretrain_1B}"
SEED="${SEED:-42}"
PYTHON="${PYTHON:-python}"

BASE_FLAGS=(
    --use_value_embed true
    --use_q_gain true
    --use_sliding_window true
    --sliding_window_size 384
    --rope_base 250000
)

run_one() {
    local name="$1"
    shift
    echo
    echo "[tiny1m-arch] === $(date) starting $name ==="
    set +e
    "$PYTHON" train_llm.py \
        --config tiny1m \
        --dataset_path "$DATASET" \
        --output_dir "runs/${name}_full" \
        --seed "$SEED" \
        "${BASE_FLAGS[@]}" \
        "$@"
    local rc=$?
    set -e
    if [ "$rc" -eq 0 ]; then
        echo "[tiny1m-arch] === $(date) $name DONE (rc=$rc) ==="
    else
        echo "[tiny1m-arch] === $(date) $name FAILED (rc=$rc) ==="
    fi
}

run_one "tiny1m_arch_base"
run_one "tiny1m_arch_mha" --n_kv_heads 4
run_one "tiny1m_arch_gqa1" --n_kv_heads 1
run_one "tiny1m_arch_tiedqk" --use_tied_qk true
run_one "tiny1m_arch_mla" --use_mla true --mla_latent_dim 16
run_one "tiny1m_arch_layernorm" --use_layernorm true
run_one "tiny1m_arch_postnorm" --use_post_norm true
run_one "tiny1m_arch_gelu" --ffn_variant gelu
run_one "tiny1m_arch_swiglu" --ffn_variant swiglu
run_one "tiny1m_arch_linearattn" --use_linear_attn true
run_one "tiny1m_arch_qkpostnorm" --use_qk_norm_post_rope true

echo
echo "[tiny1m-arch] === $(date) tiny architecture queue complete ==="
