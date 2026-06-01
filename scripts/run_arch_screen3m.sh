#!/usr/bin/env bash
set -euo pipefail

cd /root/universe-lm

git checkout experiments >/dev/null 2>&1 || git checkout -b experiments >/dev/null 2>&1

mkdir -p \
  logs \
  runs/arch/screen3m/baseline \
  runs/arch/screen3m/swiglu \
  runs/arch/screen3m/residual \
  runs/arch/screen3m/embed_residual

run() {
  local name="$1"
  shift
  echo "===== START ${name} ====="
  python3 train_llm.py "$@" 2>&1 | tee "logs/arch_screen3m_${name}.log"
  echo "===== END ${name} ====="
}

run baseline \
  --config screen3m \
  --seed 42 \
  --dataset_path processed_data/pretrain_1B \
  --output_dir runs/arch/screen3m/baseline

run swiglu \
  --config screen3m \
  --seed 42 \
  --dataset_path processed_data/pretrain_1B \
  --output_dir runs/arch/screen3m/swiglu \
  --ffn_variant swiglu

run residual \
  --config screen3m \
  --seed 42 \
  --dataset_path processed_data/pretrain_1B \
  --output_dir runs/arch/screen3m/residual \
  --residual_scale_init 0.1

run embed_residual \
  --config screen3m \
  --seed 42 \
  --dataset_path processed_data/pretrain_1B \
  --output_dir runs/arch/screen3m/embed_residual \
  --embedding_residual_scale_init 0.1

date -Iseconds | tee logs/arch_screen3m_done.log
