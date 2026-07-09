#!/usr/bin/env bash
# Pinned 8-task baseline eval (lm-eval-harness v0.4.12 @ 6d64254).
# The SmolLM2-135M head-to-head suite: zero-shot, seed 42, bf16, our harness only.
# Usage: ./run_baseline_suite.sh [MODEL_ID_OR_PATH]   (default: HuggingFaceTB/SmolLM2-135M)
set -euo pipefail
MODEL="${1:-HuggingFaceTB/SmolLM2-135M}"
TASKS="hellaswag,arc_easy,arc_challenge,piqa,winogrande,openbookqa,lambada_openai,sciq"
OUT="results/$(echo "$MODEL" | tr '/' '__')-suite"
lm_eval --model hf \
  --model_args "pretrained=${MODEL},dtype=bfloat16" \
  --tasks "$TASKS" \
  --num_fewshot 0 \
  --batch_size auto \
  --seed 42 \
  --output_path "$OUT"
echo "done -> $OUT"
