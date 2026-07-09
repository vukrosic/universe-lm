#!/usr/bin/env bash
# ============================================================================
# tiny1m arch sweep #6 — queued 2026-06-04  ·  MULTI-SEED ROBUSTNESS
# ----------------------------------------------------------------------------
# The single-seed noise floor is ~0.02 val_loss. This batch re-runs the key
# configs at seeds 43 and 44 so each has a 3-seed band (with the seed-42
# originals), telling us which effects are REAL vs noise:
#   base, mha(winner), glob3, tiedqk, diffattn(#87), nsa64(#88), hybrid(#89)
#
# All on the SwiGLU base (mha adds n_kv_heads=4) to match the seed-42 runs.
# Chained after arch#5. Re-runs the ledger collector after every run.
# ============================================================================
set -u

DATASET="${DATASET:-processed_data/pretrain_1B}"
PYTHON="${PYTHON:-python}"
DATE_TAG="0604"

echo "[tiny1m-arch6] waiting for the clean study to complete..."
for _ in $(seq 1 1080); do
    if grep -q "tiny clean study complete" /root/tiny1m_clean.log 2>/dev/null; then
        echo "[tiny1m-arch6] clean study done, starting arch#6"
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

# run_seed <name> <seed> <extra flags...>
run_seed() {
    local name="$1"; local seed="$2"; shift 2
    echo
    echo "[tiny1m-arch6] === $(date) starting ${name}_s${seed} ==="
    set +e
    "$PYTHON" train_llm.py \
        --config tiny1m \
        --dataset_path "$DATASET" \
        --output_dir "runs/tiny1m_${DATE_TAG}_${name}_s${seed}_full" \
        --seed "$seed" \
        "${BASE_FLAGS[@]}" \
        "$@"
    local rc=$?
    set -e
    if [ "$rc" -eq 0 ]; then
        echo "[tiny1m-arch6] === $(date) ${name}_s${seed} DONE (rc=$rc) ==="
    else
        echo "[tiny1m-arch6] === $(date) ${name}_s${seed} FAILED (rc=$rc) ==="
    fi
    "$PYTHON" scripts/collect_tiny1m_0604.py || true
}

for SEED in 43 44; do
    run_seed "swiglu_base"        "$SEED"
    run_seed "swiglu_mha"         "$SEED" --n_kv_heads 4
    run_seed "swiglu_glob3"       "$SEED" --global_attn_every_k 3
    run_seed "swiglu_tiedqk"      "$SEED" --use_tied_qk true
    run_seed "swiglu_diffattn"    "$SEED" --use_diff_attn true
    run_seed "swiglu_nsa64"       "$SEED" --use_nsa_global true --nsa_block 64
    run_seed "swiglu_hybridheads" "$SEED" --use_hybrid_heads true
done

echo
echo "[tiny1m-arch6] === $(date) tiny arch#6 queue complete ==="
