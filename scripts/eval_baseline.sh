#!/usr/bin/env bash
# scripts/eval_baseline.sh
#
# Pinned Phase-0 evaluation of HuggingFaceTB/SmolLM2-135M against the
# protocol in plans/benchmark-protocol.md.
#
# Two modes:
#   --smoke    Install the pinned harness in a venv, do a 5-sample run on
#              hellaswag on CPU. Verifies install + model load. ~5 min.
#              SAFE TO RUN ON A LAPTOP.
#
#   (default)  Full 7-task 0-shot suite + MMLU 5-shot. Writes results JSON
#              to results/baseline-smollm2-135m/. REQUIRES A GPU (run on
#              the Vast box).
#
# Usage:
#   ./scripts/eval_baseline.sh --smoke
#   ./scripts/eval_baseline.sh                       # full, GPU
#   ./scripts/eval_baseline.sh --tasks hellaswag     # custom subset
#
# Pinned versions live in plans/benchmark-protocol.md — if they drift,
# update that file first, this script reads from it implicitly via
# HARVEST_COMMIT below.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# --- Pinned versions (must match plans/benchmark-protocol.md §1) ---
HARVEST_REPO="https://github.com/EleutherAI/lm-evaluation-harness"
HARVEST_COMMIT="6d642546f4688648fced259eb3302efd36ece5af"   # v0.4.12
SMOLLM2_ID="HuggingFaceTB/SmolLM2-135M"
SEED=42
VENV_DIR="$REPO_ROOT/.venv-lm-eval"
RESULTS_DIR="$REPO_ROOT/results/baseline-smollm2-135m"
LOG_DIR="$REPO_ROOT/logs/eval"

# --- Flags ---
MODE="full"
TASKS_0SHOT="hellaswag,arc_easy,arc_challenge,piqa,winogrande,openbookqa,commonsense_qa"
MMLU_TASKS="mmlu"
LIMIT=""
DEVICE="cuda:0"
BATCH="auto:4"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke)         MODE="smoke"; shift ;;
        --tasks)         TASKS_0SHOT="$2"; shift 2 ;;
        --limit)         LIMIT="$2"; shift 2 ;;
        --device)        DEVICE="$2"; shift 2 ;;
        --batch)         BATCH="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "Unknown flag: $1" >&2; exit 2 ;;
    esac
done

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

# --- Venv setup (idempotent) ---
if [[ ! -d "$VENV_DIR" ]]; then
    echo "[eval_baseline] Creating venv at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -V
pip install --quiet --upgrade pip

# --- Install pinned harness (exact commit, editable) ---
if [[ ! -f "$VENV_DIR/.lm-eval-pinned" ]] \
   || [[ "$(cat "$VENV_DIR/.lm-eval-pinned")" != "$HARVEST_COMMIT" ]]; then
    echo "[eval_baseline] Installing lm-eval-harness @ $HARVEST_COMMIT"
    # Clone at exact commit into a sibling dir
    WORK_DIR="$REPO_ROOT/.lm-eval-src"
    rm -rf "$WORK_DIR"
    git clone --quiet "$HARVEST_REPO" "$WORK_DIR"
    ( cd "$WORK_DIR" && git checkout --quiet "$HARVEST_COMMIT" )
    pip install --quiet -e "$WORK_DIR[hf]"
    # Add the deps the [hf] extra misses in some envs
    pip install --quiet "transformers>=4.45" "accelerate>=0.34" "datasets>=2.20"
    echo "$HARVEST_COMMIT" > "$VENV_DIR/.lm-eval-pinned"
else
    echo "[eval_baseline] Reusing existing venv with harness @ $HARVEST_COMMIT"
fi

# --- Resolve device/dtype for the mode ---
if [[ "$MODE" == "smoke" ]]; then
    DEVICE="cpu"
    BATCH="1"
    LIMIT_ARG="--limit 5"
    TASKS="hellaswag"
    SUITE_TAG="smoke"
    BPB_MAX_BYTES=51200        # 50 KB — small enough for CPU smoke
    echo "[eval_baseline] SMOKE: 5 examples on hellaswag, CPU, fp32"
    DTYPE="float32"
else
    LIMIT_ARG=""
    if [[ -n "$LIMIT" ]]; then
        LIMIT_ARG="--limit $LIMIT"
    fi
    SUITE_TAG="full"
    DTYPE="bfloat16"
    TASKS="$TASKS_0SHOT"
    BPB_MAX_BYTES=5242880      # 5 MB — protocol default
fi

# --- 1. Run the 0-shot suite ---
LOG_0SHOT="$LOG_DIR/eval-baseline-${SUITE_TAG}-0shot.log"
echo "[eval_baseline] 0-shot suite: $TASKS"
echo "[eval_baseline] log: $LOG_0SHOT"
{
    echo "# $(date -u +%FT%TZ) lm-eval-harness @ $HARVEST_COMMIT"
    echo "# model: $SMOLLM2_ID, device: $DEVICE, batch: $BATCH, dtype: $DTYPE"
    echo "# tasks: $TASKS, num_fewshot: 0, seed: $SEED"
} > "$LOG_0SHOT"

lm_eval --model hf \
    --model_args "pretrained=$SMOLLM2_ID,dtype=$DTYPE" \
    --tasks "$TASKS" \
    --num_fewshot 0 \
    --batch_size "$BATCH" \
    --device "$DEVICE" \
    --output_path "$RESULTS_DIR" \
    --log_samples \
    --seed "$SEED" \
    $LIMIT_ARG \
    2>&1 | tee -a "$LOG_0SHOT"

# --- 2. MMLU 5-shot (only on full run; smoke skips it) ---
if [[ "$MODE" != "smoke" ]]; then
    LOG_MMLU="$LOG_DIR/eval-baseline-${SUITE_TAG}-mmlu5shot.log"
    echo "[eval_baseline] MMLU 5-shot: $MMLU_TASKS"
    {
        echo "# $(date -u +%FT%TZ) lm-eval-harness @ $HARVEST_COMMIT"
        echo "# model: $SMOLLM2_ID, device: $DEVICE, batch: $BATCH, dtype: $DTYPE"
        echo "# tasks: $MMLU_TASKS, num_fewshot: 5, seed: $SEED"
    } > "$LOG_MMLU"

    lm_eval --model hf \
        --model_args "pretrained=$SMOLLM2_ID,dtype=$DTYPE" \
        --tasks "$MMLU_TASKS" \
        --num_fewshot 5 \
        --batch_size "$BATCH" \
        --device "$DEVICE" \
        --output_path "$RESULTS_DIR" \
        --log_samples \
        --seed "$SEED" \
        2>&1 | tee -a "$LOG_MMLU"
fi

# --- 3. Held-out BPB (continuous metric) ---
LOG_BPB="$LOG_DIR/bpb-baseline-${SUITE_TAG}.log"
echo "[eval_baseline] FineWeb-Edu held-out BPB"
{
    echo "# $(date -u +%FT%TZ) BPB on HuggingFaceFW/fineweb-edu sample-10BT, first 5MB"
    echo "# model: $SMOLLM2_ID"
} > "$LOG_BPB"

python scripts/bpb_fineweb_edu.py \
    --model "$SMOLLM2_ID" \
    --dataset HuggingFaceFW/fineweb-edu \
    --subset sample-10BT \
    --split train \
    --max_bytes "$BPB_MAX_BYTES" \
    --output "$RESULTS_DIR/bpb.json" \
    2>&1 | tee -a "$LOG_BPB"

# --- 4. Manifest (for BASELINE.md reproducibility checklist) ---
MANIFEST="$RESULTS_DIR/manifest.json"
{
    echo "{"
    echo "  \"timestamp_utc\": \"$(date -u +%FT%TZ)\","
    echo "  \"harness_repo\": \"$HARVEST_REPO\","
    echo "  \"harness_commit\": \"$HARVEST_COMMIT\","
    echo "  \"model_id\": \"$SMOLLM2_ID\","
    echo "  \"mode\": \"$SUITE_TAG\","
    echo "  \"device\": \"$DEVICE\","
    echo "  \"dtype\": \"$DTYPE\","
    echo "  \"seed\": $SEED,"
    echo "  \"tasks_0shot\": \"$TASKS_0SHOT\","
    echo "  \"tasks_mmlu_5shot\": \"$MMLU_TASKS\","
    echo "  \"limit\": \"$LIMIT\","
    echo "  \"git_head\": \"$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)\""
    echo "}"
} > "$MANIFEST"

echo
echo "[eval_baseline] Done. Results: $RESULTS_DIR"
echo "[eval_baseline] Manifest: $MANIFEST"
ls -la "$RESULTS_DIR" || true
