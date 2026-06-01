#!/usr/bin/env bash
set -euo pipefail

cd /root/universe-lm
source /venv/main/bin/activate

mkdir -p logs runs/issue30_screen10m_sweep

run_one() {
  local name="$1"
  shift
  local out_dir="runs/issue30_screen10m_sweep/${name}"
  mkdir -p "${out_dir}"
  echo "===== START ${name} $(date -Iseconds) =====" | tee -a logs/issue30_screen10m_sweep_status.log
  python3 train_llm.py \
    --config screen10m \
    --seed 42 \
    --dataset_path processed_data/pretrain_1B \
    --output_dir "${out_dir}" \
    "$@" \
    2>&1 | tee "logs/issue30_screen10m_${name}.log"
  python3 - "${name}" "${out_dir}/metrics.json" <<'PY' | tee -a logs/issue30_screen10m_sweep_status.log
import json
import sys

name, path = sys.argv[1], sys.argv[2]
with open(path) as f:
    metrics = json.load(f)
final = metrics["final_metrics"]
print(f"RESULT {name} val_loss={final['val_loss']} val_acc={final['val_accuracy']} steps={metrics['actual_steps']}")
PY
  echo "===== END ${name} $(date -Iseconds) =====" | tee -a logs/issue30_screen10m_sweep_status.log
}

echo "issue30_screen10m_sweep start $(date -Iseconds)" | tee logs/issue30_screen10m_sweep_status.log

run_one baseline
run_one warmup_decay_w001 --schedule_type warmup_decay_to_zero --warmup_ratio 0.01
run_one warmup_decay_w002 --schedule_type warmup_decay_to_zero --warmup_ratio 0.02
run_one warmup_decay_w005 --schedule_type warmup_decay_to_zero --warmup_ratio 0.05
run_one cosine_w002 --schedule_type cosine --warmup_ratio 0.02

python3 - <<'PY' | tee logs/issue30_screen10m_sweep_summary.tsv
import json
from pathlib import Path

root = Path("runs/issue30_screen10m_sweep")
rows = []
for metrics_path in sorted(root.glob("*/metrics.json")):
    with metrics_path.open() as f:
        metrics = json.load(f)
    final = metrics["final_metrics"]
    rows.append((final["val_loss"], metrics_path.parent.name, final["val_accuracy"], metrics["actual_steps"]))

print("name\tval_loss\tval_accuracy\tactual_steps")
for val_loss, name, val_acc, steps in sorted(rows):
    print(f"{name}\t{val_loss:.6f}\t{val_acc:.6f}\t{steps}")
PY

echo "issue30_screen10m_sweep done $(date -Iseconds)" | tee -a logs/issue30_screen10m_sweep_status.log
