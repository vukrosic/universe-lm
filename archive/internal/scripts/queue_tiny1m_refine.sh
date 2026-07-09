#!/usr/bin/env bash
set -u

DATASET="${DATASET:-processed_data/pretrain_1B}"
SEED="${SEED:-42}"
PYTHON="${PYTHON:-python}"

run_one() {
    local name="$1"
    shift
    echo
    echo "[tiny1m-refine] === $(date) starting $name ==="
    set +e
    "$PYTHON" train_llm.py \
        --config tiny1m \
        --dataset_path "$DATASET" \
        --output_dir "runs/${name}_full" \
        --seed "$SEED" \
        "$@"
    local rc=$?
    set -e
    if [ "$rc" -eq 0 ]; then
        echo "[tiny1m-refine] === $(date) $name DONE (rc=$rc) ==="
    else
        echo "[tiny1m-refine] === $(date) $name FAILED (rc=$rc) ==="
    fi
}

BASE_FLAGS=(
    --use_value_embed true
    --use_q_gain true
    --use_sliding_window true
    --sliding_window_size 512
)

# Search the positional scale around the first tiny winner.
run_one "tiny1m_vqgain_swa_rope125k" "${BASE_FLAGS[@]}" --rope_base 125000
run_one "tiny1m_vqgain_swa_rope375k" "${BASE_FLAGS[@]}" --rope_base 375000
run_one "tiny1m_vqgain_swa_rope750k" "${BASE_FLAGS[@]}" --rope_base 750000

# Hold RoPE at 250k and search window size.
run_one "tiny1m_vqgain_rope250k_swa256" "${BASE_FLAGS[@]}" --rope_base 250000 --sliding_window_size 256
run_one "tiny1m_vqgain_rope250k_swa384" "${BASE_FLAGS[@]}" --rope_base 250000 --sliding_window_size 384
run_one "tiny1m_vqgain_rope250k_swa768" "${BASE_FLAGS[@]}" --rope_base 250000 --sliding_window_size 768

# Test which ingredients are actually load-bearing after SWA + RoPE.
run_one "tiny1m_swa_rope250k" \
    --use_sliding_window true \
    --sliding_window_size 512 \
    --rope_base 250000
run_one "tiny1m_qgain_swa_rope250k" \
    --use_q_gain true \
    --use_sliding_window true \
    --sliding_window_size 512 \
    --rope_base 250000

echo
echo "[tiny1m-refine] === $(date) tiny refine queue complete ==="
for d in runs/tiny1m_*_full; do
    [ -f "$d/metrics.json" ] || continue
    "$PYTHON" - "$d/metrics.json" <<'PY'
import json, sys
from pathlib import Path

p = Path(sys.argv[1])
d = json.load(open(p))
fm = d.get("final_metrics", {}) or {}
print(f"  {p.parent.name}: {fm.get('val_loss')}")
PY
done
