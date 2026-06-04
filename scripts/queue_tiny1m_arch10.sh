#!/usr/bin/env bash
# ============================================================================
# tiny1m arch sweep #10 — 2026-06-04  ·  BUILD ON WINNERS + NEW ARCH + ABLATE
# ----------------------------------------------------------------------------
# Best so far: pnorm1.5 residual (-0.05), multiscale heads (-0.019). Build on
# them with different architectures + ablate. Single seed 42. Chained after #9.
#   new arch:  U-net skips / token-smear gate / gated attention output
#   stack:     multiscale + U-net (two winners together)
#   ablate:    pnorm1.6 (winner exponent)
# All on pnorm1.5 base unless noted. In-batch pnorm1.5 control anchor.
# ============================================================================
set -u

DATASET="${DATASET:-processed_data/pretrain_1B}"
PYTHON="${PYTHON:-python}"
DATE_TAG="0604"

echo "[tiny1m-arch10] waiting for arch#9 to complete..."
for _ in $(seq 1 720); do
    if grep -q "tiny arch#9 complete" /root/tiny1m_arch9.log 2>/dev/null; then
        echo "[tiny1m-arch10] arch#9 done, starting arch#10"
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
    echo "[tiny1m-arch10] === $(date) starting $name ==="
    set +e
    "$PYTHON" train_llm.py \
        --config tiny1m \
        --dataset_path "$DATASET" \
        --output_dir "runs/tiny1m_${DATE_TAG}_arch10_${name}_s42_full" \
        --seed 42 \
        "${BASE_FLAGS[@]}" \
        "$@"
    local rc=$?
    set -e
    if [ "$rc" -eq 0 ]; then
        echo "[tiny1m-arch10] === $(date) $name DONE (rc=$rc) ==="
    else
        echo "[tiny1m-arch10] === $(date) $name FAILED (rc=$rc) ==="
    fi
    "$PYTHON" scripts/collect_tiny1m_0604.py || true
}

run_one "ctrl_pnorm15"      --norm_type pnorm1.5
run_one "unet"              --norm_type pnorm1.5 --use_unet_skips true --unet_skip_count 4
run_one "smear"             --norm_type pnorm1.5 --use_smear_gate true
run_one "attngate"          --norm_type pnorm1.5 --use_attn_output_gate true
run_one "multiscale_unet"   --norm_type pnorm1.5 --use_multiscale_heads true --use_unet_skips true --unet_skip_count 4
run_one "pnorm16"           --norm_type pnorm1.6

echo
echo "[tiny1m-arch10] === $(date) tiny arch#10 complete ==="
