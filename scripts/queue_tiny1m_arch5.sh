#!/usr/bin/env bash
# ============================================================================
# tiny1m arch sweep #5 — queued 2026-06-04  ·  DEEPSEEK-HEAVY, on the winner
# ----------------------------------------------------------------------------
# New baseline = SwiGLU + full MHA (6.2678, the arch#2 winner). Everything
# here stacks a DeepSeek-lineage mechanism or a sweep on top of it:
#   #73 MLA latent sweep (DeepSeek-V2 Multi-head Latent Attention)
#   #88 NSA compressed-global block sweep (DeepSeek Native Sparse Attention)
#   #87 Differential Attention / #89 Hybrid heads on the winner
#   RoPE-base, window, norm, softcap, embed-lever sweeps — losers logged too.
#
# Chained after arch#4. Re-runs the ledger collector after EVERY run so no
# result is ever lost mid-batch.
# ============================================================================
set -u

DATASET="${DATASET:-processed_data/pretrain_1B}"
SEED="${SEED:-42}"
PYTHON="${PYTHON:-python}"
DATE_TAG="0604"

echo "[tiny1m-arch5] waiting for arch#4 to complete..."
for _ in $(seq 1 720); do
    if grep -q "tiny arch#4 queue complete" /root/tiny1m_arch4.log 2>/dev/null; then
        echo "[tiny1m-arch5] arch#4 done, starting arch#5"
        break
    fi
    sleep 10
done

# Winner baseline: SwiGLU + full MHA.
BASE_FLAGS=(
    --use_value_embed true
    --use_q_gain true
    --use_sliding_window true
    --sliding_window_size 384
    --rope_base 250000
    --ffn_variant swiglu
    --n_kv_heads 4
)

run_one() {
    local name="$1"
    shift
    echo
    echo "[tiny1m-arch5] === $(date) starting $name ==="
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
        echo "[tiny1m-arch5] === $(date) $name DONE (rc=$rc) ==="
    else
        echo "[tiny1m-arch5] === $(date) $name FAILED (rc=$rc) ==="
    fi
    "$PYTHON" scripts/collect_tiny1m_0604.py || true
}

# #73 DeepSeek-V2 MLA latent-dim sweep.
run_one "mha_mla8"   --use_mla true --mla_latent_dim 8
run_one "mha_mla16"  --use_mla true --mla_latent_dim 16
run_one "mha_mla32"  --use_mla true --mla_latent_dim 32

# #88 DeepSeek NSA compressed-global, on the winner, block sweep.
run_one "mha_nsa32"  --use_nsa_global true --nsa_block 32
run_one "mha_nsa64"  --use_nsa_global true --nsa_block 64
run_one "mha_nsa128" --use_nsa_global true --nsa_block 128

# #87 / #89 the other new mechanisms, on the winner.
run_one "mha_diffattn"    --use_diff_attn true
run_one "mha_hybridheads" --use_hybrid_heads true

# RoPE-base sweep on the winner (rope500k was bad on base; probe lower).
run_one "mha_rope100k" --rope_base 100000
run_one "mha_rope175k" --rope_base 175000
run_one "mha_rope350k" --rope_base 350000

# Window-size sweep on the winner.
run_one "mha_swa256" --sliding_window_size 256
run_one "mha_swa512" --sliding_window_size 512
run_one "mha_swa768" --sliding_window_size 768

# Norm / softcap / qk-position on the winner.
run_one "mha_layernorm"  --use_layernorm true
run_one "mha_qkpostnorm" --use_qk_norm_post_rope true
run_one "mha_softcap15"  --logit_softcap 15.0
run_one "mha_softcap30"  --logit_softcap 30.0

# Expected losers — logged anyway for completeness.
run_one "mha_postnorm" --use_post_norm true
run_one "mha_dilated2" --attention_dilation 2

# Embed levers on the winner.
run_one "mha_kgain"    --use_k_gain true
run_one "mha_ffnembed" --use_ffn_embed true

echo
echo "[tiny1m-arch5] === $(date) tiny arch#5 queue complete ==="
