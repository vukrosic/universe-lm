#!/usr/bin/env bash
#
# kaggle_arch_sweep.sh — "what to use" breadth sweep for the post.
#
# Control vs each architecture trick, ONE flag at a time, tiny1m @ 3M tokens,
# seed 42 only. Output: val_loss + delta-vs-control so each trick gets a clean
# "use this / skip this" verdict. No seeds/variance by design (this is a
# fast directional screen, not a significance test).
#
# Pairs runs across the two T4s (CUDA_VISIBLE_DEVICES=0/1) and skips runs whose
# metrics.json already shows the full token count, so re-runs resume.
#
# Usage on the Kaggle SSH box (driver libs must be on the path):
#   export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
#   REPO_DIR=/kaggle/working/universe-lm bash scripts/kaggle_arch_sweep.sh
#
# Env overrides: REPO_DIR, RUNS (default /kaggle/working/arch_runs), DATA.

set -eu

REPO_DIR=${REPO_DIR:-/kaggle/working/universe-lm}
RUNS=${RUNS:-/kaggle/working/arch_runs}
DATA=${DATA:-processed_data/speedrun_40M}
FULL_TOKENS=3000000

cd "$REPO_DIR"
mkdir -p "$RUNS"

# name|extra-train-flags  (control has empty flags)
SWEEP=(
  "ctrl|"
  "qk_norm_post_rope|--use_qk_norm_post_rope true"
  "value_embed|--use_value_embed true"
  "layerscale|--use_layerscale true"
  "attn_output_gate|--use_attn_output_gate true"
  "zero_init_resid|--zero_init_resid true"
  "parallel_block|--use_parallel_block true"
  "ffn_swiglu|--ffn_variant swiglu"
  "ffn_gelu|--ffn_variant gelu"
  "embed_residual|--use_embed_residual true"
  "unet_sigmoid_k6|--use_unet_skips true --unet_gate_type sigmoid --unet_gate_init -1.5 --unet_skip_count 6"
  "diff_attn|--use_diff_attn true"
  "smear_gate|--use_smear_gate true"
  "post_norm|--use_post_norm true"
  "attn_sink|--use_attn_sink true"
  "q_gain|--use_q_gain true"
  "qk_norm_pre|--qk_norm_type pnorm1.5"
)

done_already() {
  local name=$1
  local f="$RUNS/$name/metrics.json"
  [ -f "$f" ] || return 1
  local tok
  tok=$(python3 -c "import json; print(json.load(open('$f')).get('tokens_seen', 0))" 2>/dev/null || echo 0)
  [ "${tok%.*}" -ge "$FULL_TOKENS" ]
}

launch() {
  local gpu=$1 name=$2; shift 2
  if done_already "$name"; then echo "SKIP   $name"; return; fi
  CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 nohup python3 -u train_llm.py --config tiny1m --seed 42 "$@" --dataset_path "$DATA" --log_every 50 --output_dir "$RUNS/$name" > "$RUNS/$name.log" 2>&1 &
  echo "LAUNCH GPU$gpu  $name  (PID=$!)"
}

echo "=== Arch breadth sweep — tiny1m @ 3M, seed 42 ==="
echo "Runs: $RUNS"; echo

# Pair entries two-at-a-time across GPU 0 and 1.
i=0
while [ $i -lt ${#SWEEP[@]} ]; do
  a="${SWEEP[$i]}"; na="${a%%|*}"; fa="${a#*|}"
  launch 0 "$na" $fa
  j=$((i+1))
  if [ $j -lt ${#SWEEP[@]} ]; then
    b="${SWEEP[$j]}"; nb="${b%%|*}"; fb="${b#*|}"
    launch 1 "$nb" $fb
  fi
  wait
  echo "--- pair done ($na${nb:+, $nb}) ---"
  i=$((i+2))
done

echo
echo "=== RESULTS (delta vs ctrl, lower=better) ==="
python3 - "$RUNS" <<'PY'
import json, os, sys, glob
runs = sys.argv[1]
res = {}
for f in glob.glob(os.path.join(runs, "*", "metrics.json")):
    n = os.path.basename(os.path.dirname(f))
    try:
        res[n] = json.load(open(f))["final_metrics"]["val_loss"]
    except Exception:
        pass
ctrl = res.get("ctrl")
print(f'{"trick":24s} {"val_loss":>9s} {"delta":>9s}  verdict')
print(f'{"-"*24} {"-"*9} {"-"*9}  -------')
for n, v in sorted(res.items(), key=lambda kv: kv[1]):
    d = (v - ctrl) if ctrl is not None and n != "ctrl" else 0.0
    verdict = "baseline" if n == "ctrl" else ("USE  " if d < -0.005 else ("skip " if d > 0.005 else "meh  "))
    ds = "" if n == "ctrl" else f'{d:+.4f}'
    print(f'{n:24s} {v:9.4f} {ds:>9s}  {verdict}')
PY
