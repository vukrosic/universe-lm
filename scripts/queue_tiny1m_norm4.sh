#!/usr/bin/env bash
# ============================================================================
# tiny1m NORM sweep #4 — 2026-06-04  ·  pnorm1.5 DEEP-DIVE on a CLEAN baseline
# ----------------------------------------------------------------------------
# Isolate the NORM. Plain full-attention transformer (NO swa / NO value-embed /
# NO q-gain / default rope=10k) so nothing confounds the norm signal. Vary only
# the residual-stream norm to (a) pin the optimal p, (b) find WHERE p=1.5 helps.
#
#   baseline:   --config tiny1m  (value_embed off, q_gain off, full attention,
#               rope_base=10000, n_kv_heads=2) — a normal small transformer.
#   p-grid:     rmsnorm(=p2) · pnorm1.0 · 1.25 · 1.375 · 1.5 · 1.625 · 1.75
#   centering:  layernorm (does subtract-mean help vs pure pnorm?)
#   placement:  qk pnorm1.5 / v pnorm1.5 (body stays pnorm1.5) — does the
#               outlier-robust norm help inside attention too, or only the body?
# Single seed 42. Chained after arch#12. rmsnorm is the in-batch anchor.
# ============================================================================
set -u

DATASET="${DATASET:-processed_data/pretrain_1B}"
PYTHON="${PYTHON:-python}"
DATE_TAG="0604"

echo "[tiny1m-norm4] waiting for arch#11 to complete..."
for _ in $(seq 1 1800); do
    if grep -q "tiny arch#11 complete" /root/tiny1m_arch11.log 2>/dev/null; then
        echo "[tiny1m-norm4] arch#11 done, starting norm#4"
        break
    fi
    sleep 10
done

# Clean baseline: everything explicitly OFF so config drift can't confound it.
BASE_FLAGS=(
    --use_value_embed false
    --use_q_gain false
    --use_sliding_window false
    --rope_base 10000
)

run_one() {
    local name="$1"; shift
    echo
    echo "[tiny1m-norm4] === $(date) starting $name ==="
    set +e
    "$PYTHON" train_llm.py \
        --config tiny1m \
        --dataset_path "$DATASET" \
        --output_dir "runs/tiny1m_${DATE_TAG}_norm4_${name}_s42_full" \
        --seed 42 \
        "${BASE_FLAGS[@]}" \
        "$@"
    local rc=$?
    set -e
    if [ "$rc" -eq 0 ]; then
        echo "[tiny1m-norm4] === $(date) $name DONE (rc=$rc) ==="
    else
        echo "[tiny1m-norm4] === $(date) $name FAILED (rc=$rc) ==="
    fi
    "$PYTHON" scripts/collect_tiny1m_0604.py || true
}

# --- fine p-grid: pin the optimum around 1.5 ---
run_one "rmsnorm"   --norm_type rmsnorm     # p=2 anchor
run_one "pnorm1.0"  --norm_type pnorm1.0
run_one "pnorm1.25" --norm_type pnorm1.25
run_one "pnorm1.375" --norm_type pnorm1.375
run_one "pnorm1.5"  --norm_type pnorm1.5    # current champion
run_one "pnorm1.625" --norm_type pnorm1.625
run_one "pnorm1.75" --norm_type pnorm1.75

# --- centering anchor: does subtract-mean add anything over pure pnorm? ---
run_one "layernorm" --use_layernorm true

# --- placement: does p=1.5 help INSIDE attention, or only the body? ---
run_one "body_qk_pnorm15" --norm_type pnorm1.5 --qk_norm_type pnorm1.5
run_one "body_v_pnorm15"  --norm_type pnorm1.5 --v_norm_type pnorm1.5

echo
echo "[tiny1m-norm4] === $(date) tiny norm#4 complete ==="
