#!/usr/bin/env bash
# ============================================================================
# tiny1m CLEAN study — queued 2026-06-04  ·  PARAM-MATCHED + MULTI-SEED
# ----------------------------------------------------------------------------
# The publication-grade core. Answers two clean questions with error bars:
#   Q1: Is SwiGLU's "win" real at ISO-PARAM, or is it the +36% FFN params?
#   Q2: Is MHA's "win" real at ISO-PARAM, or is it the +9.5% KV params?
#
# Five configs, all ~546k non-emb params except the two deliberately-inflated
# controls, each at 3 seeds (42/43/44) so every number has a 3-seed band:
#   ctrl        squared-ReLU, kv2, d_ff=256   (reference, 546,288)
#   swiglu_iso  SwiGLU,        kv2, d_ff=170   (iso-param, 544,752)
#   swiglu_big  SwiGLU,        kv2, d_ff=256   (+36% params — the illusion)
#   mha_iso     squared-ReLU, kv4, d_ff=220   (iso-param, 543,216)
#   mha_big     squared-ReLU, kv4, d_ff=256   (+9.5% params — the illusion)
#
# All share the established base recipe (V-embed + q-gain + SWA384 + RoPE250k).
# Ledger collector re-runs after every run.
# ============================================================================
set -u

DATASET="${DATASET:-processed_data/pretrain_1B}"
PYTHON="${PYTHON:-python}"
DATE_TAG="0604"
WAIT_ON="${WAIT_ON:-/root/tiny1m_arch4.log}"   # marker file to chain behind
WAIT_MARK="${WAIT_MARK:-tiny arch#4 queue complete}"

echo "[tiny1m-clean] waiting for: $WAIT_MARK ..."
for _ in $(seq 1 1080); do
    if grep -q "$WAIT_MARK" "$WAIT_ON" 2>/dev/null; then
        echo "[tiny1m-clean] dependency done, starting clean study"
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
)

# run_cfg <name> <seed> <extra flags...>
run_cfg() {
    local name="$1"; local seed="$2"; shift 2
    echo
    echo "[tiny1m-clean] === $(date) starting ${name}_s${seed} ==="
    set +e
    "$PYTHON" train_llm.py \
        --config tiny1m \
        --dataset_path "$DATASET" \
        --output_dir "runs/tiny1m_${DATE_TAG}_clean_${name}_s${seed}_full" \
        --seed "$seed" \
        "${BASE_FLAGS[@]}" \
        "$@"
    local rc=$?
    set -e
    if [ "$rc" -eq 0 ]; then
        echo "[tiny1m-clean] === $(date) ${name}_s${seed} DONE (rc=$rc) ==="
    else
        echo "[tiny1m-clean] === $(date) ${name}_s${seed} FAILED (rc=$rc) ==="
    fi
    "$PYTHON" scripts/collect_tiny1m_0604.py || true
}

for SEED in 42 43 44; do
    run_cfg "ctrl"       "$SEED" --ffn_variant squared_relu --n_kv_heads 2 --d_ff 256
    run_cfg "swiglu_iso" "$SEED" --ffn_variant swiglu       --n_kv_heads 2 --d_ff 170
    run_cfg "swiglu_big" "$SEED" --ffn_variant swiglu       --n_kv_heads 2 --d_ff 256
    run_cfg "mha_iso"    "$SEED" --ffn_variant squared_relu --n_kv_heads 4 --d_ff 220
    run_cfg "mha_big"    "$SEED" --ffn_variant squared_relu --n_kv_heads 4 --d_ff 256
done

echo
echo "[tiny1m-clean] === $(date) tiny clean study complete ==="
