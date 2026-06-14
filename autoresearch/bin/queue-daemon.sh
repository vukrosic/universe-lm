#!/usr/bin/env bash
# queue-daemon.sh — drain the GPU queue with NO LLM in the hot loop.
#
# Replaces the AI "runner" pass (prompts/runner.md) with a deterministic,
# idempotent, cron-safe script. It does only the mechanical last mile — claim,
# run, poll, pull, judge, flip — because the verdict is pure arithmetic
# (bin/baseline.sh verdict) and every needs-run idea arrives in the fixed shape
# defined in ../RUN-CONTRACT.md (run.json + an _arq_<idea>.py that defines `C`).
#
# AI still owns everything upstream: mining, taste, review, and writing code.
#
# One tick does:
#   0. resolve the live box (remote-box.json)
#   1. finalize: pull + judge any OK / bounce any FAIL still left in ~/arq/STATUS
#   2. if the `arq` tmux queue is live -> report and stop (runs in flight)
#   3. else: reclaim dead `running` ideas, claim the needs-run set, sync+smoke
#      on the box, ask baseline.sh whether to MEASURE, build one guarded queue,
#      launch it in detached `arq` tmux. Next tick finalizes it.
#
# Idempotent: never relaunches a live queue, never re-finalizes a done idea.
# No auto-push — local working tree + remote run only.
#
# Usage:
#   autoresearch/bin/queue-daemon.sh [--once] [--loop SECONDS] [--dry-run]
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
IDEAS="$ROOT/autoresearch/ideas"
FLIP="$ROOT/autoresearch/bin/flip.sh"
BASELINE="$ROOT/autoresearch/bin/baseline.sh"
BOX_JSON="$ROOT/autoresearch/remote-box.json"
STATE="$ROOT/autoresearch/daemon-state.json"
CLOSED="$ROOT/autoresearch/closed.md"

REMOTE_TMUX="arq"
CTRL_CONFIG="configs.llm_config.Tiny1M3MConfig"
DATASET="processed_data/pretrain_1B"
STALE_MIN="${STALE_MIN:-20}"     # a `running` idea older than this with no live arq = dead
DEFAULT_TIMEOUT="${JOB_TIMEOUT:-12m}"
LOCK="/tmp/queue-daemon.lock"

DRY=0; MODE="once"; LOOP_SECS=600
while [ $# -gt 0 ]; do case "$1" in
  --dry-run) DRY=1;;
  --once) MODE="once";;
  --loop) MODE="loop"; [ -n "${2:-}" ] && { LOOP_SECS="$2"; shift; };;
  *) echo "unknown arg: $1" >&2; exit 2;;
esac; shift; done

log() { echo "[daemon $(date -u +%H:%M:%S)] $*"; }

# ── box (parsed once per tick from remote-box.json) ──────────────────────────
HOST=""; PORT=""; SSHUSER=""; REMOTE_REPO=""; REMOTE_VENV=""
load_box() {
  [ -f "$BOX_JSON" ] || { log "NO BOX: $BOX_JSON missing"; return 1; }
  local vals
  vals="$(python3 - "$BOX_JSON" <<'PY'
import json, sys
b = json.load(open(sys.argv[1]))
print(b.get("host", ""))
print(b.get("port", ""))
print(b.get("user", "root"))
print(b.get("remote_repo", "/root/universe-lm"))
print(b.get("remote_venv", "/venv/main"))
PY
)" || return 1
  { read -r HOST; read -r PORT; read -r SSHUSER; read -r REMOTE_REPO; read -r REMOTE_VENV; } <<<"$vals"
  [ -n "$HOST" ] && [ -n "$PORT" ] || { log "NO BOX: host/port empty in remote-box.json"; return 1; }
  CTL_PATH="/tmp/lab-arq-ctl-${SSHUSER}-${HOST}-${PORT}"   # shared multiplex socket
  return 0
}

# SSH connection multiplexing — one warm TCP connection shared by every ssh/scp
# this tick makes (and across ticks, within ControlPersist). A daemon tick fires
# a burst: box_reachable, finalize pulls, git pull, scp+smoke per idea, launch.
# Without multiplexing each of those is a fresh handshake and Vast.ai throttles
# auth ("too many ssh connections in quick succession" -> "Connection closed by
# remote host"), which wedges the tick and leaves ideas stuck `running`. Same
# pattern as voidspark app/api/gpu-usage/route.ts. Keyed by user-host-port so a
# new box gets a fresh socket; ControlPersist outlives one tick so back-to-back
# ssh calls reuse the master. Shared with orchestrate.sh's arq probe.
CTL_PATH=""   # set by load_box (needs host/port)
SSH() { ssh -o ControlMaster=auto -o "ControlPath=$CTL_PATH" -o ControlPersist=120 \
            -o StrictHostKeyChecking=accept-new -o BatchMode=yes -o ConnectTimeout=12 \
            -p "$PORT" "$SSHUSER@$HOST" "$@"; }
SCP_TO() { scp -o ControlMaster=auto -o "ControlPath=$CTL_PATH" -o ControlPersist=120 \
               -o StrictHostKeyChecking=accept-new -o BatchMode=yes -o ConnectTimeout=12 \
               -P "$PORT" "$1" "$SSHUSER@$HOST:$2"; }
SCP_FROM() { scp -o ControlMaster=auto -o "ControlPath=$CTL_PATH" -o ControlPersist=120 \
                 -o StrictHostKeyChecking=accept-new -o BatchMode=yes -o ConnectTimeout=12 \
                 -P "$PORT" "$SSHUSER@$HOST:$1" "$2"; }

arq_live() { SSH "tmux has-session -t $REMOTE_TMUX 2>/dev/null"; }
box_reachable() { SSH "true" >/dev/null 2>&1; }

iso_to_epoch() { date -j -u -f "%Y-%m-%dT%H:%M:%SZ" "$1" +%s 2>/dev/null || echo 0; }
field() { awk -F': *' -v k="$2" '$1==k{print $2; exit}' "$1"; }            # field <file> <key>
idea_status() { field "$IDEAS/$1/idea.md" status; }

# ── needs-run claim eligibility: must ship a valid run.json + existing arq_file ─
# Prints "<idea> <arq_file> <timeout>" per claimable idea; logs and skips the
# malformed ones (they never reach the box — an implementer bug, not a result).
claimable() {
  local d idea rj arq to
  for d in "$IDEAS"/*/; do
    [ -f "$d/idea.md" ] || continue          # skip non-idea dirs (_closed/ etc.)
    idea="$(basename "$d")"
    [ "$(idea_status "$idea")" = "needs-run" ] || continue
    rj="$d/run.json"
    if [ ! -f "$rj" ]; then log "SKIP $idea: no run.json (implementer must emit it)"; continue; fi
    read -r arq to < <(python3 - "$rj" <<'PY'
import json, sys
try:
    j = json.load(open(sys.argv[1]))
except Exception as e:
    print("ERR ERR"); sys.exit()
print(j.get("arq_file", "ERR"), j.get("job_timeout", ""))
PY
)
    if [ "$arq" = "ERR" ] || [ -z "$arq" ]; then log "SKIP $idea: bad run.json"; continue; fi
    if [ ! -f "$ROOT/$arq" ]; then log "SKIP $idea: arq_file $arq missing in repo"; continue; fi
    [ -z "$to" ] && to="$DEFAULT_TIMEOUT"
    echo "$idea $arq $to"
  done
}

# ── results.json helpers (the durable raw record baseline.sh reads) ──────────
# probe the box and (re)write instance{} into results.json
results_init() {
  local rdir="$1" probe gpu drv vram cc
  probe="$(SSH "nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1; echo '<<>>'; export PATH=$REMOTE_VENV/bin:\$PATH; python -c 'import torch;print(\".\".join(map(str,torch.cuda.get_device_capability())))' 2>/dev/null")" || return 1
  gpu="$(echo "$probe"  | sed -n '1p' | awk -F', *' '{print $1}')"
  drv="$(echo "$probe"  | sed -n '1p' | awk -F', *' '{print $2}')"
  vram="$(echo "$probe" | sed -n '1p' | awk -F', *' '{print $3}')"
  cc="$(echo "$probe"   | sed -n '/<<>>/{n;p}')"
  mkdir -p "$rdir"
  python3 - "$rdir/results.json" "$gpu" "$drv" "$vram" "$cc" "$HOST:$PORT" <<'PY'
import json, os, sys, datetime
path, gpu, drv, vram, cc, host = sys.argv[1:7]
r = json.load(open(path)) if os.path.exists(path) else {}
r.setdefault("date", datetime.date.today().isoformat())
r["tier"] = "tiny1m3m"; r["seed"] = 42
r["instance"] = {"id": host, "gpu": gpu, "vram_mib": vram, "driver": drv,
                 "compute_cap": cc, "host": host}
r.setdefault("dynamo", "TORCHDYNAMO_DISABLE=1 (sm_86 polar_express OOM work-around)")
r.setdefault("data", "processed_data/pretrain_1B")
r.setdefault("runs", [])
json.dump(r, open(path, "w"), indent=2); open(path, "a").write("\n")
PY
}

# upsert one finished run into results.json (skips if already recorded)
results_upsert() {
  python3 - "$1" "$2" "$3" "$4" "$5" "$6" <<'PY'
import json, sys
path, name, val, trn, acc, logname = sys.argv[1:7]
r = json.load(open(path))
runs = r.setdefault("runs", [])
e = next((x for x in runs if x.get("name") == name), None)
if e is None:
    e = {"name": name}; runs.append(e)
e["val_loss"]   = float(val) if val else None
e["train_loss"] = float(trn) if trn else None
e["val_accuracy"] = float(acc) if acc else None
e["log"] = logname
json.dump(r, open(path, "w"), indent=2); open(path, "a").write("\n")
PY
}

# ── local tick state (so finalize on a later tick finds the right dir/baseline) ─
state_write() {  # state_write <results_dir> <mode> <mean> <band> <box_key>
  python3 - "$STATE" "$1" "$2" "$3" "$4" "$5" "$HOST:$PORT" <<'PY'
import json, sys
path, rdir, mode, mean, band, key, host = sys.argv[1:8]
json.dump({"results_dir": rdir, "mode": mode, "mean": mean, "band": band,
           "box_key": key, "host": host}, open(path, "w"), indent=2)
PY
}
state_get() { [ -f "$STATE" ] && field_json "$STATE" "$1"; }
field_json() { python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get(sys.argv[2],""))' "$1" "$2" 2>/dev/null; }

# ── evidence.md (deterministic; the transfer-note prose is left to an analyzer) ─
write_evidence() {  # write_evidence <idea> <verdict> <val> <delta> <mean> <band> <rdir>
  local idea="$1" verdict="$2" val="$3" delta="$4" mean="$5" band="$6" rdir="$7"
  cat > "$IDEAS/$idea/evidence.md" <<EOF
# Evidence — $idea

## Verdict: $verdict
- tier: tiny1m3m, seed 42, box: $HOST:$PORT
- baseline: mean=$mean ±$band (box-keyed cache)
- treatment val: $val   Δ vs baseline: $delta
- bpb: n/a (pending harness)
- pass/fail bar: noise-band rule — WIN iff val < mean − band (see plan.md for the paper-level claim)
- raw: ${rdir#$ROOT/}/results.json (logs alongside)
- date: $(date -u +%F)
- judged-by: queue-daemon.sh (deterministic)

## Transfer note
(auto) deferred — see idea.md \`## Scale evidence\`. Written by the analyzer pass, not the daemon.
EOF
}

append_closed_null() {  # append_closed_null <idea> <delta>
  local idea="$1" delta="$2" line marker
  marker='<!-- reviewer/evidence step appends one line per close here -->'
  line="- $idea — null: Δ=$delta at tiny1m3m (inside variance) — $(date -u +%F)"
  [ -f "$CLOSED" ] || return 0
  grep -qF -- "$idea — null:" "$CLOSED" && return 0
  awk -v m="$marker" -v l="$line" '{print} index($0,m)&&!d{print l; d=1}' "$CLOSED" > "$CLOSED.tmp" \
    && mv "$CLOSED.tmp" "$CLOSED"
}

flip() {  # respects --dry-run
  if [ "$DRY" = 1 ]; then log "DRY flip $*"; return 0; fi
  "$FLIP" "$@"
}

# ── parse one pulled log's Final readout (deterministic — see RUN-CONTRACT.md) ─
# echoes "<val> <train> <acc>"; empty val => run did not finish cleanly
parse_log() {
  local f="$1" val trn acc
  val="$(grep -a 'Final Val Loss:'     "$f" 2>/dev/null | tail -1 | awk '{print $NF}')"
  trn="$(grep -a 'Final Train Loss:'   "$f" 2>/dev/null | tail -1 | awk '{print $NF}')"
  acc="$(grep -a 'Final Val Accuracy:' "$f" 2>/dev/null | tail -1 | awk '{print $NF}')"
  echo "$val $trn $acc"
}

# ── 1. finalize: drain ~/arq/STATUS into verdicts (idempotent) ───────────────
finalize() {
  local status rdir mean band
  status="$(SSH "cat ~/arq/STATUS 2>/dev/null")" || return 0
  [ -n "$status" ] || return 0
  rdir="$(state_get results_dir)"; [ -n "$rdir" ] || rdir="$ROOT/remote-results/$(date -u +%F)-vast-tiny1m3m"
  mean="$(state_get mean)"; band="$(state_get band)"
  mkdir -p "$rdir"

  local measured_ctrls=0 line tag name
  while IFS= read -r line; do
    tag="${line%% *}"; name="$(echo "$line" | awk '{print $2}')"
    [ -n "$name" ] || continue
    case "$tag" in
      OK)
        # ctrls feed the baseline; they are not ideas
        if [[ "$name" == ctrl* ]]; then
          local clog="$rdir/${name}_${PORT}.log"
          [ -f "$clog" ] || SCP_FROM "~/arq/logs/$name.log" "$clog" 2>/dev/null || true
          read -r cval ctrn cacc < <(parse_log "$clog")
          [ -n "$cval" ] && results_upsert "$rdir/results.json" "$name" "$cval" "$ctrn" "$cacc" "$(basename "$clog")"
          measured_ctrls=1; continue
        fi
        # a treatment idea: finalize only if still `running` (idempotent)
        [ "$(idea_status "$name")" = "running" ] || continue
        local tlog="$rdir/${name}_${PORT}.log"
        SCP_FROM "~/arq/logs/$name.log" "$tlog" 2>/dev/null || { log "pull failed $name"; continue; }
        read -r val trn acc < <(parse_log "$tlog")
        if [ -z "$val" ]; then
          flip "$name" needs-recode daemon "OK but no Final Val Loss in log — incomplete run"
          continue
        fi
        results_upsert "$rdir/results.json" "$name" "$val" "$trn" "$acc" "$(basename "$tlog")"
        finalize_one "$name" "$val" "$rdir"
        ;;
      FAIL|TIMEOUT)
        [[ "$name" == ctrl* ]] && continue
        [ "$(idea_status "$name")" = "running" ] || continue
        local flog="$rdir/${name}_${PORT}.log"
        SCP_FROM "~/arq/logs/$name.log" "$flog" 2>/dev/null || true
        local cause; cause="$(grep -aiE 'Error|Traceback|OOM|CUDA|assert|NaN' "$flog" 2>/dev/null | tail -1 | cut -c1-160)"
        flip "$name" needs-recode daemon "run FAILED ($tag): ${cause:-see log}"
        ;;
    esac
  done <<<"$status"

  # MEASURE queue just produced fresh ctrls -> (re)write the cache before any
  # later verdict trusts the band. CACHED queue -> advance the staleness counter.
  if [ -f "$rdir/results.json" ]; then
    if [ "$measured_ctrls" = 1 ]; then
      [ "$DRY" = 1 ] || "$BASELINE" measure "$rdir/results.json" || true
    fi
  fi
}

# judge one finished treatment via baseline.sh verdict (mean ± band)
finalize_one() {  # finalize_one <idea> <val> <rdir>
  local idea="$1" val="$2" rdir="$3" out verdict delta mean band
  # refresh mean/band from the cache for this box (works for CACHED + post-measure)
  read -r _tag mean band _key < <("$BASELINE" check "$rdir/results.json" 2>/dev/null || echo "X 0 0 0")
  out="$("$BASELINE" verdict "$rdir/results.json" "$val" 2>/dev/null || echo "NO-BASELINE 0")"
  verdict="${out%% *}"; delta="$(echo "$out" | awk '{print $2}')"
  case "$verdict" in
    WIN)
      write_evidence "$idea" WIN "$val" "$delta" "$mean" "$band" "$rdir"
      flip "$idea" done daemon "WIN: trt=$val vs $mean±$band (Δ$delta)"
      log "$idea — WIN Δ$delta" ;;
    NULL)
      write_evidence "$idea" NULL "$val" "$delta" "$mean" "$band" "$rdir"
      flip "$idea" done daemon "NULL: trt=$val inside $mean±$band (Δ$delta)"
      [ "$DRY" = 1 ] || append_closed_null "$idea" "$delta"
      log "$idea — NULL Δ$delta" ;;
    *)
      log "$idea — NO-BASELINE (left running; needs a MEASURE queue first)" ;;
  esac
}

# ── 3a. reclaim dead `running` ideas (queue died, no OK/FAIL) ────────────────
reclaim() {
  local now d idea up age
  now=$(date -u +%s)
  for d in "$IDEAS"/*/; do
    [ -f "$d/idea.md" ] || continue          # skip non-idea dirs (_closed/ etc.)
    idea="$(basename "$d")"
    [ "$(idea_status "$idea")" = "running" ] || continue
    up="$(field "$d/idea.md" updated)"
    age=$(( (now - $(iso_to_epoch "$up")) / 60 ))
    if [ "$age" -ge "$STALE_MIN" ]; then
      log "reclaim $idea (running, stale ${age}m, no live queue)"
      flip "$idea" needs-run daemon "reclaimed: queue died (stale ${age}m)"
    fi
  done
}

# ── 3b. sync + CPU build-smoke each claimed arq on the box ───────────────────
# echoes the subset of "<idea> <arq> <to>" lines that passed smoke
sync_and_smoke() {
  local batch="$1" pass="" idea arq to
  # git pull committed code; scp helper + each arq (idea-local, uncommitted-friendly)
  SSH "cd $REMOTE_REPO && git pull --ff-only 2>&1 | tail -1" >&2 || log "git pull warned (continuing)"
  SCP_TO "$ROOT/autoresearch/bin/_box_smoke.py" "$REMOTE_REPO/_box_smoke.py" 2>/dev/null || true
  while read -r idea arq to; do
    [ -n "$idea" ] || continue
    SCP_TO "$ROOT/$arq" "$REMOTE_REPO/$arq" 2>/dev/null || { log "scp $arq failed"; flip "$idea" needs-recode daemon "scp arq_file failed"; continue; }
    local sm
    sm="$(SSH "cd $REMOTE_REPO && export PATH=$REMOTE_VENV/bin:\$PATH TORCHDYNAMO_DISABLE=1 && python _box_smoke.py '$arq' 2>&1 | tail -3")"
    if echo "$sm" | grep -q "SMOKE_OK"; then
      pass+="$idea $arq $to"$'\n'
    else
      log "smoke FAIL $idea: $(echo "$sm" | tail -1)"
      flip "$idea" needs-recode daemon "build-smoke failed: $(echo "$sm" | tail -1 | cut -c1-160)"
    fi
  done <<<"$batch"
  printf '%s' "$pass"
}

# ── 3c. build the guarded queue script and launch it detached ───────────────
launch_queue() {  # launch_queue <smoked-batch> <mode CACHED|MEASURE>
  local batch="$1" mode="$2" rdir maxto="$DEFAULT_TIMEOUT" idea arq to
  rdir="$ROOT/remote-results/$(date -u +%F)-vast-tiny1m3m"
  # pick the largest job_timeout in the batch as the global cap
  while read -r idea arq to; do [ -n "$to" ] && maxto="$to"; done <<<"$batch"

  local qs="/tmp/run_queue.sh"
  {
    echo '#!/usr/bin/env bash'
    echo "export PATH=$REMOTE_VENV/bin:\$PATH"
    echo 'export PYTHONUNBUFFERED=1 TORCHDYNAMO_DISABLE=1'
    echo "cd $REMOTE_REPO || exit 1"
    echo 'mkdir -p ~/arq/logs'
    echo "JOB_TIMEOUT=\"$maxto\""
    cat <<'WRAP'
run () {
  local name="$1"; shift
  echo "START $name $(date -u +%FT%TZ)" >> ~/arq/STATUS
  if timeout "$JOB_TIMEOUT" "$@" > ~/arq/logs/"$name".log 2>&1; then
    echo "OK   $name $(date -u +%FT%TZ)" >> ~/arq/STATUS
  else
    rc=$?
    [ "$rc" = 124 ] && echo "TIMEOUT $name killed after $JOB_TIMEOUT" >> ~/arq/logs/"$name".log
    echo "FAIL $name rc=$rc $(date -u +%FT%TZ)" >> ~/arq/STATUS
  fi
}
WRAP
    if [ "$mode" = "MEASURE" ]; then
      local CTRL="python train_llm.py --config_class $CTRL_CONFIG --seed 42 --dataset_path $DATASET --warmup false"
      echo "run ctrl  $CTRL"
      echo "run ctrl2 $CTRL"
      echo "run ctrl3 $CTRL"
    fi
    while read -r idea arq to; do
      [ -n "$idea" ] || continue
      echo "run $idea python $arq"
    done <<<"$batch"
    echo 'echo "QUEUE_DONE $(date -u +%FT%TZ)" >> ~/arq/STATUS'
  } > "$qs"

  if [ "$DRY" = 1 ]; then log "DRY launch ($mode): $(echo "$batch" | grep -c .) jobs"; cat "$qs"; return 0; fi
  SSH "mkdir -p ~/arq && : > ~/arq/STATUS"
  SCP_TO "$qs" "~/arq/run_queue.sh"
  SSH "tmux new-session -d -s $REMOTE_TMUX 'bash ~/arq/run_queue.sh'"
  results_init "$rdir"
  # decide + cache the baseline mode/band for this queue, store for finalize
  local chk tag mean band key
  chk="$("$BASELINE" check "$rdir/results.json" 2>/dev/null || echo "MEASURE pending fresh)")"
  tag="${chk%% *}"
  if [ "$tag" = "CACHED" ]; then mean="$(echo "$chk" | awk '{print $2}')"; band="$(echo "$chk" | awk '{print $3}')"; key="$(echo "$chk" | awk '{print $4}')"
  else mean=""; band=""; key=""; fi
  state_write "$rdir" "$mode" "$mean" "$band" "$key"
  log "launched $mode queue: $(echo "$batch" | grep -c .) treatments in detached $REMOTE_TMUX tmux"
}

# ── one full tick ────────────────────────────────────────────────────────────
tick() {
  load_box || { log "skipping tick — no box"; return 0; }
  # Reachability guard FIRST: a transient ssh blip must be a no-op, never a
  # reason to claim ideas and then bounce them all to needs-recode on failed scp.
  if ! box_reachable; then log "box $HOST:$PORT unreachable — skipping tick (no claims)"; return 0; fi

  finalize    # always: drain any leftover OK/FAIL into verdicts (idempotent)

  if arq_live; then
    log "queue LIVE — runs in flight; polled + finalized what's done."
    return 0
  fi

  reclaim     # queue is gone: recover any idea stuck `running` with no result

  local batch; batch="$(claimable)"
  if [ -z "$(echo "$batch" | grep -c .)" ] || [ -z "$batch" ]; then
    log "GPU IDLE: no needs-run candidates with a valid run.json — upstream starving"
    return 0
  fi

  # claim the whole set first (lock them as running)
  local idea arq to claimed=""
  while read -r idea arq to; do
    [ -n "$idea" ] || continue
    flip "$idea" running daemon "claimed: queued by daemon"
    claimed+="$idea $arq $to"$'\n'
  done <<<"$batch"

  # sync + smoke (drops + bounces the ones that fail to build)
  local smoked; smoked="$(sync_and_smoke "$claimed")"
  if [ -z "$smoked" ] || [ -z "$(echo "$smoked" | grep -c .)" ]; then
    log "no idea survived build-smoke this tick"; return 0
  fi

  # baseline: does this box need fresh ctrls, or can we run treatment-only?
  local rdir="$ROOT/remote-results/$(date -u +%F)-vast-tiny1m3m"
  results_init "$rdir" 2>/dev/null || true
  local chk tag mode
  chk="$("$BASELINE" check "$rdir/results.json" 2>/dev/null)"; tag="${chk%% *}"
  if [ "$tag" = "CACHED" ]; then mode="CACHED"; else mode="MEASURE"; fi
  log "baseline: $chk -> $mode path"

  launch_queue "$smoked" "$mode"
}

# ── entry ────────────────────────────────────────────────────────────────────
# Single-tick lock. flock on Linux; macOS has no flock (the dashboard fires this
# from a Mac), so fall back to an atomic mkdir lock. Without this fallback every
# tick exited at "flock: command not found" on macOS — the daemon never ran and
# its reclaim/multiplexing were dead code. A stale lock (dead PID) is reclaimed
# so a crashed tick can't wedge the drainer.
if command -v flock >/dev/null 2>&1; then
  exec 9>"$LOCK"
  if ! flock -n 9; then log "another tick holds the lock — exiting"; exit 0; fi
else
  LOCKD="$LOCK.d"
  if ! mkdir "$LOCKD" 2>/dev/null; then
    opid="$(cat "$LOCKD/pid" 2>/dev/null)"
    if [ -n "$opid" ] && kill -0 "$opid" 2>/dev/null; then
      log "another tick holds the lock (pid $opid) — exiting"; exit 0
    fi
    rm -rf "$LOCKD"; mkdir "$LOCKD" 2>/dev/null || { log "lock race — exiting"; exit 0; }
  fi
  echo "$$" > "$LOCKD/pid"
  trap 'rm -rf "$LOCKD"' EXIT
fi

if [ "$MODE" = "loop" ]; then
  log "loop mode, every ${LOOP_SECS}s (ctrl-c to stop)"
  while true; do tick; sleep "$LOOP_SECS"; done
else
  tick
fi
