#!/usr/bin/env bash
set -euo pipefail

cd /root/universe-lm
source /venv/main/bin/activate

mkdir -p logs runs/issue30/screen10m_warmup_decay runs/issue30/10m_warmup_decay

SCREEN_BASELINE=5.2041
SCREEN_METRICS="runs/issue30/screen10m_warmup_decay/metrics.json"
FULL_METRICS="runs/issue30/10m_warmup_decay/metrics.json"

echo "issue30 start $(date -Iseconds)" | tee logs/issue30_warmup_decay_status.log

python3 train_llm.py \
  --config screen10m \
  --schedule_type warmup_decay_to_zero \
  --warmup_ratio 0.02 \
  --seed 42 \
  --dataset_path processed_data/pretrain_1B \
  --output_dir runs/issue30/screen10m_warmup_decay \
  2>&1 | tee logs/issue30_screen10m_warmup_decay.log

SCREEN_VAL_LOSS="$(python3 - <<'PY'
import json
with open("runs/issue30/screen10m_warmup_decay/metrics.json") as f:
    m = json.load(f)
print(m.get("final_val_loss", m.get("val_loss", "")))
PY
)"

echo "screen10m final_val_loss=${SCREEN_VAL_LOSS} baseline=${SCREEN_BASELINE}" \
  | tee -a logs/issue30_warmup_decay_status.log

if python3 - <<'PY'
import json
baseline = 5.2041
with open("runs/issue30/screen10m_warmup_decay/metrics.json") as f:
    m = json.load(f)
val = float(m.get("final_val_loss", m.get("val_loss")))
raise SystemExit(0 if val < baseline else 1)
PY
then
  PROMOTE=0
else
  PROMOTE=1
fi

if [ "$PROMOTE" -eq 0 ]; then
  echo "decision=promote_to_10m" | tee -a logs/issue30_warmup_decay_status.log
  python3 train_llm.py \
    --config 10m \
    --schedule_type warmup_decay_to_zero \
    --warmup_ratio 0.02 \
    --seed 42 \
    --dataset_path processed_data/pretrain_1B \
    --output_dir runs/issue30/10m_warmup_decay \
    2>&1 | tee logs/issue30_10m_warmup_decay.log
else
  echo "decision=kill_after_screen10m" | tee -a logs/issue30_warmup_decay_status.log
fi

echo "issue30 done $(date -Iseconds)" | tee -a logs/issue30_warmup_decay_status.log
