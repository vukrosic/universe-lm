#!/usr/bin/env bash
# ============================================================================
# tiny1m arch sweep #7 — queued 2026-06-04  ·  OUTLIER-ROBUST ARCHITECTURES
# ----------------------------------------------------------------------------
# Derived from the norm finding (robust L1.5 > L2 because massive-activation
# channels dominate L2). Apply outlier-robustness at NEW architectural loci:
#   #91 robust QK-norm (pnorm1.5 on Q,K)   -> cleaner attention logits
#   #92 robust V-norm  (pnorm1.5 on V)     -> cleaner value aggregation
#   #93 satrelu FFN    (c*tanh(relu/c))    -> stop amplifying outliers at source
# plus a "stack everything robust" run to test whether the mechanism compounds.
# Single seed (42), 546k squared-ReLU base. Ledger collector after each run.
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
    echo "[tiny1m-arch7] === $(date) starting $name ==="
    set +e
    "$PYTHON" train_llm.py \
        --config tiny1m \
        --dataset_path "$DATASET" \
        --output_dir "runs/tiny1m_${DATE_TAG}_robust_${name}_s42_full" \
        --seed 42 \
        "${BASE_FLAGS[@]}" \
        "$@"
    local rc=$?
    set -e
    if [ "$rc" -eq 0 ]; then
        echo "[tiny1m-arch7] === $(date) $name DONE (rc=$rc) ==="
    else
        echo "[tiny1m-arch7] === $(date) $name FAILED (rc=$rc) ==="
    fi
    "$PYTHON" scripts/collect_tiny1m_0604.py || true
}

# anchor (rmsnorm everywhere, squared_relu) for same-batch comparison
run_one "control"
# #91 outlier-robust attention logits
run_one "qk15"       --qk_norm_type pnorm1.5
# #92 outlier-robust value aggregation
run_one "v15"        --v_norm_type pnorm1.5
# #93 anti-outlier FFN activation
run_one "satrelu"    --ffn_variant satrelu
# stack the three attention/FFN robustness levers
run_one "combo"      --qk_norm_type pnorm1.5 --v_norm_type pnorm1.5 --ffn_variant satrelu
# full robust transformer: also put the residual stream on pnorm1.5 (#90)
run_one "fullrobust" --norm_type pnorm1.5 --qk_norm_type pnorm1.5 --v_norm_type pnorm1.5 --ffn_variant satrelu

echo
echo "[tiny1m-arch7] === $(date) tiny arch#7 (robust) complete ==="
