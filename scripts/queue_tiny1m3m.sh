#!/bin/bash
# Tiny 1M-param / 3M-token queue for fast single-seed iteration.
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
    echo "[tiny1m] === $(date) starting $name ==="
    rm -rf "$out"
    python train_llm.py \
        --config_class "$cfg" \
        --dataset_path processed_data/pretrain_1B \
        --output_dir "$out" \
        --seed 42 \
        > "$log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "[tiny1m] === $(date) $name DONE (rc=$rc) ==="
    else
        echo "[tiny1m] === $(date) $name FAILED (rc=$rc) ==="
    fi
    find "$out" -name "*.pt" -delete 2>/dev/null
    return $rc
}

run_one "configs.llm_config.Tiny1M3MConfig" "tiny1m_ctrl"
run_one "configs.llm_config.Tiny1M3MQGainConfig" "tiny1m_qgain"
run_one "configs.llm_config.Tiny1M3MVQGainConfig" "tiny1m_vqgain"
run_one "configs.llm_config.Tiny1M3MSWAConfig" "tiny1m_swa"
run_one "configs.llm_config.Tiny1M3MVQGainSWAHighRoPEConfig" "tiny1m_vqgain_swa_highrope"
run_one "configs.llm_config.Tiny1M3MVQGainHighRoPESWA384Config" "tiny1m_vqgain_highrope_swa384"
run_one "configs.llm_config.Tiny1M3MVQGainSWAHighRoPE250KConfig" "tiny1m_vqgain_swa_rope250k"

echo ""
echo "[tiny1m] === $(date) tiny queue complete ==="
for r in \
    tiny1m_ctrl_full \
    tiny1m_qgain_full \
    tiny1m_vqgain_full \
    tiny1m_swa_full \
    tiny1m_vqgain_swa_highrope_full \
    tiny1m_vqgain_highrope_swa384_full \
    tiny1m_vqgain_swa_rope250k_full
do
    if [ -f "runs/$r/metrics.json" ]; then
        val=$(python3 -c "import json; print(json.load(open('runs/$r/metrics.json'))['final_metrics']['val_loss'])")
        echo "  $r: $val"
    else
        echo "  $r: NO METRICS"
    fi
done
