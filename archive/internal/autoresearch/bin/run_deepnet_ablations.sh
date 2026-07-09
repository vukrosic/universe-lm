#!/usr/bin/env bash
# run_deepnet_ablations.sh — the DeepNet ablation suite at the 8M rung.
#   E3: deepnet_ab (alpha + canonical beta init) — does the init-side beta stack on alpha?
#   E4: rezero (learned scalar alpha init 0), layerscale (learned per-channel gamma init 1e-4)
#       — specificity: is the (null) result DeepNet-specific or generic residual damping?
# See autoresearch/DEEPNET-RESEARCH.md. Predictions from the E5 init-probes: E3 ~ null
# (beta adds no flattening), E4 ~ all null (the whole family is redundant with Muon).
#
# Runs AFTER the main local ladder (run_ladder_local.sh) so we never contend for VRAM:
# waits for its "LOCAL LADDER COMPLETE" marker, then for any straggler run_rung.py.
#   nohup bash autoresearch/bin/run_deepnet_ablations.sh > logs/deepnet_ablations.log 2>&1 &
set -u
cd /root/universe-lm || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCHDYNAMO_DISABLE=1
PY=/venv/main/bin/python
mkdir -p logs autoresearch/ladder
log(){ echo "$(date '+%F %T') | $*"; }

log "deepnet-ablations queued — waiting for main ladder ('LOCAL LADDER COMPLETE')..."
while ! grep -q "LOCAL LADDER COMPLETE" logs/ladder_driver.log 2>/dev/null; do sleep 60; done
while pgrep -f run_rung.py >/dev/null 2>&1; do sleep 30; done
log "main ladder done — starting DeepNet ablations at 8M (baseline+deepnet already logged)."

run(){ local arm=$1; local logf="logs/ladder_Ladder8M155MConfig_${arm}_s42.log";
  log "START $arm 8M -> $logf";
  $PY autoresearch/bin/run_rung.py --arm "$arm" --rung Ladder8M155MConfig --seed 42 > "$logf" 2>&1;
  log "DONE  $arm rc=$? ($(grep -c '"arch"' autoresearch/ladder/results.jsonl 2>/dev/null) points logged)"; }

run deepnet_ab   # E3
run rezero       # E4
run layerscale   # E4

log "DEEPNET ABLATIONS COMPLETE."
cat autoresearch/ladder/results.jsonl 2>/dev/null
