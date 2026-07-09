#!/usr/bin/env bash
# ============================================================================
# tiny1m NORM study — queued 2026-06-04  ·  INVENTED NORMALIZATIONS (#90)
# ----------------------------------------------------------------------------
# Seven residual-stream norms, each at 3 seeds (42/43/44) for error bars, on
# the honest 546k squared-ReLU base (matches the existing arch_layernorm ref).
# Norms are ~param-free, so this is a CONFOUND-FREE comparison:
#   rmsnorm    baseline (L2 sphere projection)
#   layernorm  mean+var (the known -0.024 reference)
#   peak       L-inf  : x / max|x|              (no sqrt)         [invented]
#   manhattan  L1     : x / mean|x|             (no sqrt/square)  [invented]
#   squash     DyT    : g*tanh(a*x)  reduction-free               [invented]
#   center     mean-only: x - mean(x)                             [invented]
#   manifold   x / rms(x)**rho, rho learnable (inits to RMSNorm)  [invented]
#
# Chained after arch#6. Ledger collector re-runs after every run.
# ============================================================================
set -u

DATASET="${DATASET:-processed_data/pretrain_1B}"
PYTHON="${PYTHON:-python}"
DATE_TAG="0604"

if [ "${RUN_NOW:-0}" = "1" ]; then
    echo "[tiny1m-norm] RUN_NOW set — starting norm study immediately"
else
    echo "[tiny1m-norm] waiting for arch#6 to complete..."
    for _ in $(seq 1 1440); do
        if grep -q "tiny arch#6 queue complete" /root/tiny1m_arch6.log 2>/dev/null; then
            echo "[tiny1m-norm] arch#6 done, starting norm study"
            break
        fi
        sleep 10
    done
fi

# Honest 546k base: value-emb + q-gain + SWA384 + RoPE250k, squared-ReLU, kv2.
BASE_FLAGS=(
    --use_value_embed true
    --use_q_gain true
    --use_sliding_window true
    --sliding_window_size 384
    --rope_base 250000
)

# run_norm <norm_type> <seed>
run_norm() {
    local nt="$1"; local seed="$2"
    echo
    echo "[tiny1m-norm] === $(date) starting ${nt}_s${seed} ==="
    set +e
    "$PYTHON" train_llm.py \
        --config tiny1m \
        --dataset_path "$DATASET" \
        --output_dir "runs/tiny1m_${DATE_TAG}_norm_${nt}_s${seed}_full" \
        --seed "$seed" \
        "${BASE_FLAGS[@]}" \
        --norm_type "$nt"
    local rc=$?
    set -e
    if [ "$rc" -eq 0 ]; then
        echo "[tiny1m-norm] === $(date) ${nt}_s${seed} DONE (rc=$rc) ==="
    else
        echo "[tiny1m-norm] === $(date) ${nt}_s${seed} FAILED (rc=$rc) ==="
    fi
    "$PYTHON" scripts/collect_tiny1m_0604.py || true
}

for SEED in 42; do
    for NT in rmsnorm layernorm peak manhattan squash center manifold; do
        run_norm "$NT" "$SEED"
    done
done

echo
echo "[tiny1m-norm] === $(date) tiny norm study complete ==="
