#!/usr/bin/env bash
# baseline.sh — manage the box-keyed baseline cache in ONE call.
# Never hand-edit autoresearch/baseline-cache.json — use this so the box_key,
# noise_band, and runs_since_measure stay consistent. See BASELINE-CACHE-DESIGN.md.
#
# Subcommands:
#   check  <results.json>
#       Print whether the box class in <results.json> has a fresh cached baseline.
#       Exit 0 + "CACHED <mean> <band> <box_key>"  -> reuse it, run treatment-only.
#       Exit 10 + "MEASURE <box_key> <reason>"     -> a trigger fired, run N ctrls.
#       Triggers: no entry for box_key | commit mismatch | runs_since_measure>=K.
#
#   measure <results.json> [commit]
#       (Re)measure the baseline for the box class in <results.json> from its
#       ctrl* runs (same-seed-42 re-runs). Writes/replaces the cache entry with
#       val_mean, val_std, noise_band=max(0.04,2*std), commit, runs_since_measure=0.
#
#   bump <results.json>
#       After a treatment-only run on a cached box, increment runs_since_measure
#       (drives the staleness guard K).
#
#   verdict <results.json> <treatment_val>
#       Judge a treatment against the cached baseline for its box class:
#       prints "WIN <delta>" | "NULL <delta>" | "NO-BASELINE". WIN iff
#       treatment < val_mean - noise_band (caller still applies the plan bar).
#
#   promote <results.json> <val> [band]
#       PIN the baseline for the box class to <val> (the champion's measured
#       loss) — no ctrl re-runs needed. Sets val_mean=<val>, pinned=true,
#       commit=unknown so `check` always returns CACHED for it (the champion is
#       the trusted baseline until the next record promotes a new one). This is
#       how a winning run becomes the new baseline without re-measuring.
#
# Env: K (staleness guard, default 25). Pinned entries ignore K + commit.
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
cache="$root/autoresearch/baseline-cache.json"
K="${K:-25}"

cmd="${1:?subcommand required: check|measure|bump|verdict}"; shift || true

python3 - "$cmd" "$cache" "$K" "$@" <<'PY'
import json, sys, os, glob, hashlib, statistics, subprocess, datetime

cmd, cache, K = sys.argv[1], sys.argv[2], int(sys.argv[3])
args = sys.argv[4:]

def box_key(inst):
    raw = f"{inst.get('gpu')}|{inst.get('compute_cap')}|{inst.get('driver')}"
    return hashlib.sha1(raw.encode()).hexdigest()[:12], raw

def load_cache():
    if os.path.exists(cache):
        return json.load(open(cache))
    return {"tier": "tiny1m3m", "seed": 42, "boxes": {}}

def save_cache(c):
    with open(cache, "w") as f:
        json.dump(c, f, indent=2)
        f.write("\n")

def head_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=os.path.dirname(cache)).decode().strip()
    except Exception:
        return "unknown"

def ctrl_vals(results):
    return [r["val_loss"] for r in results.get("runs", [])
            if r.get("name", "").startswith("ctrl") and r.get("val_loss") is not None]

def die(msg, code=1):
    print(msg); sys.exit(code)

# --- load the results.json the command operates on ---
res_path = args[0] if args else None
if not res_path or not os.path.exists(res_path):
    die(f"no results.json: {res_path}", 2)
results = json.load(open(res_path))
inst = results.get("instance", {})
bk, raw = box_key(inst)
c = load_cache()
entry = c["boxes"].get(bk)

if cmd == "check":
    cur = head_commit()
    if entry is None:
        die(f"MEASURE {bk} no-entry-for-box ({raw})", 10)
    # A pinned (champion) baseline is the trusted bar until the next record —
    # it never re-measures on commit churn or staleness (see `promote`).
    if entry.get("pinned"):
        die(f"CACHED {entry['val_mean']} {entry['noise_band']} {bk}", 0)
    if entry.get("commit") not in (cur, "unknown"):
        die(f"MEASURE {bk} commit-changed ({entry.get('commit')}->{cur})", 10)
    if entry.get("runs_since_measure", 0) >= K:
        die(f"MEASURE {bk} stale (runs_since_measure>={K})", 10)
    die(f"CACHED {entry['val_mean']} {entry['noise_band']} {bk}", 0)

elif cmd == "promote":
    if len(args) < 2:
        die("promote needs <results.json> <val> [band]", 6)
    val = round(float(args[1]), 4)
    band = round(float(args[2]), 4) if len(args) > 2 else (
        entry.get("noise_band", 0.04) if entry else 0.04)
    base = entry or {}
    base.update({
        "box_key": bk,
        "gpu": inst.get("gpu"), "compute_cap": inst.get("compute_cap"),
        "driver": inst.get("driver"), "commit": "unknown",
        "val_mean": val, "noise_band": band, "pinned": True,
        "measured_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "runs_since_measure": 0, "source_results": res_path,
    })
    c["boxes"][bk] = base
    save_cache(c)
    print(f"PROMOTED {bk} val_mean={val} band={band} pinned")

elif cmd == "measure":
    vals = ctrl_vals(results)
    if len(vals) < 1:
        die(f"no ctrl* runs in {res_path} to measure from", 3)
    mean = round(statistics.mean(vals), 4)
    std = round(statistics.pstdev(vals), 4) if len(vals) > 1 else 0.0
    band = round(max(0.04, 2 * std), 4)
    commit = args[1] if len(args) > 1 else head_commit()
    c["boxes"][bk] = {
        "box_key": bk,
        "gpu": inst.get("gpu"), "compute_cap": inst.get("compute_cap"),
        "driver": inst.get("driver"), "commit": commit,
        "n_measurements": len(vals), "val_runs": vals,
        "val_mean": mean, "val_std": std, "noise_band": band,
        "measured_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "runs_since_measure": 0, "source_results": res_path,
    }
    save_cache(c)
    print(f"MEASURED {bk} mean={mean} std={std} band={band} n={len(vals)}")

elif cmd == "bump":
    if entry is None:
        die(f"no cache entry for {bk} to bump", 4)
    entry["runs_since_measure"] = entry.get("runs_since_measure", 0) + 1
    save_cache(c)
    print(f"BUMP {bk} runs_since_measure={entry['runs_since_measure']}/{K}")

elif cmd == "verdict":
    if len(args) < 2:
        die("verdict needs <results.json> <treatment_val>", 5)
    if entry is None:
        die("NO-BASELINE", 0)
    trt = float(args[1]); mean = entry["val_mean"]; band = entry["noise_band"]
    delta = round(trt - mean, 4)
    if trt < mean - band:
        print(f"WIN {delta}")
    else:
        print(f"NULL {delta}")

else:
    die(f"unknown subcommand: {cmd}", 1)
PY
