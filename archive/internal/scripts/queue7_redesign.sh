#!/bin/bash
# Queue 7: single-seed redesign around the current screen20m winner.
set +e
cd /root/my-life/llm-research-kit-scaling
source /venv/main/bin/activate
mkdir -p logs runs

run_one() {
    local cfg="$1"
    local name="$2"
    local out="runs/${name}_full"
    local log="logs/${name}.log"
    echo ""
    echo "[q7] === $(date) starting $name ==="
    rm -rf "$out"
    python train_llm.py \
        --config_class "$cfg" \
        --dataset_path processed_data/pretrain_1B \
        --output_dir "$out" \
        --seed 42 \
        > "$log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "[q7] === $(date) $name DONE (rc=$rc) ==="
    else
        echo "[q7] === $(date) $name FAILED (rc=$rc) ==="
    fi
    find "$out" -name "*.pt" -delete 2>/dev/null
    return $rc
}

run_one "configs.llm_config.Screen10M20MVQGainSWAHighRoPEQKPostNormConfig" "s_vqgain_swa_highrope_qkpostnorm"
run_one "configs.llm_config.Screen10M20MVQGainHighRoPESWA384Config" "s_vqgain_highrope_swa384"
run_one "configs.llm_config.Screen10M20MVQGainHighRoPESWA768Config" "s_vqgain_highrope_swa768"
run_one "configs.llm_config.Screen10M20MVQGainSWAHighRoPE250KConfig" "s_vqgain_swa_rope250k"
run_one "configs.llm_config.Screen10M20MVQGainSWAHighRoPE1MConfig" "s_vqgain_swa_rope1m"
run_one "configs.llm_config.Screen10M20MVQGainSWAHighRoPELinearAttnConfig" "s_vqgain_swa_highrope_linearattn_fixed"

echo ""
echo "[q7] === $(date) q7 complete ==="
for r in \
    s_vqgain_swa_highrope_qkpostnorm_full \
    s_vqgain_highrope_swa384_full \
    s_vqgain_highrope_swa768_full \
    s_vqgain_swa_rope250k_full \
    s_vqgain_swa_rope1m_full \
    s_vqgain_swa_highrope_linearattn_fixed_full
do
    if [ -f "runs/$r/metrics.json" ]; then
        val=$(python3 -c "import json; print(json.load(open('runs/$r/metrics.json'))['final_metrics']['val_loss'])")
        echo "  $r: $val"
    else
        echo "  $r: NO METRICS"
    fi
done
