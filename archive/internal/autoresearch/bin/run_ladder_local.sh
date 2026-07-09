#!/usr/bin/env bash
# run_ladder_local.sh — fill the LOCAL release-ladder points on the box, in
# sequence. Waits for any in-flight run_rung.py to finish first (so it composes
# with an already-launched baseline), then runs each remaining (arm x rung).
# Each run logs one non-embed-N point to autoresearch/ladder/results.jsonl.
#
# Local rungs only (8M/13M/23M); 52M + 135M need contributor GPUs. Arms are
# baseline + deepnet (the polyalibi arm is CUT under DECISIONS.jsonl D002).
#
#   nohup bash autoresearch/bin/run_ladder_local.sh > logs/ladder_driver.log 2>&1 &
set -u
cd /root/universe-lm || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCHDYNAMO_DISABLE=1
PY=/venv/main/bin/python
mkdir -p logs autoresearch/ladder

log() { echo "$(date '+%F %T') | $*"; }

# Wait for the currently-running rung (e.g. the first baseline) to finish so we
# never contend for VRAM.
log "waiting for any in-flight run_rung.py to finish..."
while pgrep -f "run_rung.py" >/dev/null 2>&1; do sleep 30; done
log "GPU clear — starting the remaining local ladder."

run () {  # arm rung
  local arm=$1 rung=$2
  local logf="logs/ladder_${rung}_${arm}_s42.log"
  log "START $arm $rung -> $logf"
  $PY autoresearch/bin/run_rung.py --arm "$arm" --rung "$rung" --seed 42 > "$logf" 2>&1
  log "DONE  $arm $rung rc=$? ($(grep -c '"arch"' autoresearch/ladder/results.jsonl 2>/dev/null) points logged so far)"
}

# rung 1: baseline already run before this driver; do its deepnet pair.
run deepnet  Ladder8M155MConfig
# rung 2 (both arms)
run baseline Ladder13M252MConfig
run deepnet  Ladder13M252MConfig
# rung 3 (both arms)
run baseline Ladder23M469MConfig
run deepnet  Ladder23M469MConfig

log "LOCAL LADDER COMPLETE."
log "results.jsonl:"
cat autoresearch/ladder/results.jsonl 2>/dev/null
