#!/usr/bin/env bash
# ============================================================================
# tiny1m arch sweep #9 — 2026-06-04  ·  BEAT BASELINE (stack the winner)
# ----------------------------------------------------------------------------
# pnorm1.5 (residual norm) is the only proven win (-0.05). Try to BEAT it by
# stacking new structure on top. Single seed 42, 546k base. Chained after #8.
#   #99 attn_sink  - softmax-off-by-one (kill attention sinks at the source)
#   stacks: sink+pnorm1.5, multiscale+pnorm1.5  (compound the wins?)
# In-batch control + pnorm1.5 anchor for clean same-conditions comparison.
# ============================================================================
set -u

DATASET="${DATASET:-processed_data/pretrain_1B}"
PYTHON="${PYTHON:-python}"
DATE_TAG="0604"

echo "[tiny1m-arch9] waiting for arch#8 to complete..."
for _ in $(seq 1 720); do
    if grep -q "tiny arch#8 complete" /root/tiny1m_arch8.log 2>/dev/null; then
        echo "[tiny1m-arch9] arch#8 done, starting arch#9"
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

run_one() {
    local name="$1"; shift
    echo
    echo "[tiny1m-arch9] === $(date) starting $name ==="
    set +e
    "$PYTHON" train_llm.py \
        --config tiny1m \
        --dataset_path "$DATASET" \
        --output_dir "runs/tiny1m_${DATE_TAG}_arch9_${name}_s42_full" \
        --seed 42 \
        "${BASE_FLAGS[@]}" \
        "$@"
    local rc=$?
    set -e
    if [ "$rc" -eq 0 ]; then
        echo "[tiny1m-arch9] === $(date) $name DONE (rc=$rc) ==="
    else
        echo "[tiny1m-arch9] === $(date) $name FAILED (rc=$rc) ==="
    fi
    "$PYTHON" scripts/collect_tiny1m_0604.py || true
}

run_one "control"
run_one "pnorm15"             --norm_type pnorm1.5
run_one "sink"                --use_attn_sink true
run_one "sink_pnorm15"        --use_attn_sink true --norm_type pnorm1.5
run_one "multiscale_pnorm15"  --use_multiscale_heads true --norm_type pnorm1.5

echo
echo "[tiny1m-arch9] === $(date) tiny arch#9 complete ==="
