#!/bin/bash
# run_longcontext_screen_8m.sh — screen long-context levers at 8M rung
# All arms run at 8M (cheap), one-by-one sequentially (single GPU).
# Results logged to autoresearch/ladder/results.jsonl.
# Runs the full screening: RoPE (3 bases), QK-norm, Diff-Attn

set -u
cd /Users/vukrosic/my-life/llm-research-kit-scaling || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCHDYNAMO_DISABLE=1
PY=/usr/bin/python3
mkdir -p logs autoresearch/ladder

log(){ echo "$(date '+%F %T') | $*"; }

log "=== LONG-CONTEXT LEVER SCREENING AT 8M ==="
log "Baseline control already logged (4 points: 8M/13M baseline+deepnet)"
log "Now screening: RoPE (3 bases), QK-norm, Diff-Attn"
log ""

run(){
  local arm=$1
  local logf="logs/ladder_Ladder8M155MConfig_${arm}_s42.log"
  log "START $arm @ 8M → $logf"
  $PY autoresearch/bin/run_rung.py --arm "$arm" --rung Ladder8M155MConfig --seed 42 > "$logf" 2>&1
  rc=$?
  log "DONE  $arm rc=$rc ($(grep -c '"arch"' autoresearch/ladder/results.jsonl 2>/dev/null) total points)"
  return $rc
}

# Baseline is already done (point #1, #3 for 8M/13M). Start with long-context arms.
log "Running RoPE at 3 bases (100k, 250k, 500k)..."
run ropebase100k || log "ERROR: ropebase100k failed"
run ropebase250k || log "ERROR: ropebase250k failed"
run ropebase500k || log "ERROR: ropebase500k failed"

log ""
log "Running QK-norm..."
run qknorm || log "ERROR: qknorm failed"

log ""
log "Running Diff-Attn..."
run diffattn || log "ERROR: diffattn failed"

log ""
log "=== LONG-CONTEXT SCREENING COMPLETE ==="
log "All results logged to autoresearch/ladder/results.jsonl"
log ""
echo "=== RESULTS TABLE ==="
$PY autoresearch/bin/ladder_status.py --no-pull 2>&1 | tail -20
