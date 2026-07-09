#!/usr/bin/env python3
"""Verify a submitted run: schema -> config hash -> (optional) reproduce -> bar.

CI runs this on every PR. A run is `accepted` only when the submitted config
hashes to the recorded hash and, with REPRODUCE=1, the experiment re-runs to the
same number within tolerance. No network, no GPU, stdlib only.

    python verify/verify.py --run runs/T001/example-run
    REPRODUCE=1 python verify/verify.py --run runs/T001/example-run
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # token2science/
DEFAULT_TOLERANCE = 1e-9
REQUIRED = [
    "task_id", "goal_id", "worker", "metric", "value",
    "lower_is_better", "seed", "config_path", "config_hash", "command",
]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return "sha256:" + h.hexdigest()


def fail(msg):
    print(f"FAIL {msg}")
    print("VERDICT: rejected")
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True,
                    help="path to a run folder containing result.json")
    args = ap.parse_args()

    run_dir = os.path.abspath(args.run)
    res_path = os.path.join(run_dir, "result.json")
    if not os.path.isfile(res_path):
        fail(f"no result.json in {run_dir}")
    res = json.load(open(res_path))

    for k in REQUIRED:
        if k not in res:
            fail(f"result.json missing required key: {k}")

    goal_dir = os.path.join(ROOT, "goals", res["goal_id"])
    task_dir = os.path.join(goal_dir, "tasks", res["task_id"])
    if not os.path.isdir(task_dir):
        fail(f"task folder not found: {task_dir}")

    # 1. config hash: the thing that was run must match what was claimed.
    cfg_path = os.path.join(task_dir, res["config_path"])
    if not os.path.isfile(cfg_path):
        fail(f"config not found: {cfg_path}")
    actual = sha256_file(cfg_path)
    if actual != res["config_hash"]:
        fail(f"config hash mismatch: result={res['config_hash']} file={actual}")
    print("OK   config hash matches")

    # 2. the run must report the metric the goal actually asks for.
    goal = json.load(open(os.path.join(goal_dir, "goal.json")))
    if res["metric"] != goal["metric"]:
        fail(f"metric '{res['metric']}' != goal metric '{goal['metric']}'")
    if res["lower_is_better"] != goal["lower_is_better"]:
        fail("lower_is_better disagrees with the goal")
    print("OK   metric matches goal")

    # 3. reproduce: a machine that did not produce the run re-runs it.
    if os.environ.get("REPRODUCE") == "1":
        tol = float(res.get("tolerance", DEFAULT_TOLERANCE))
        out = subprocess.run(res["command"], cwd=task_dir, shell=True,
                             capture_output=True, text=True)
        if out.returncode != 0:
            fail(f"reproduce command failed:\n{out.stderr}")
        m = re.search(r"RESULT\b.*?value=([-+0-9.eE]+)", out.stdout)
        if not m:
            fail(f"no RESULT line in experiment output:\n{out.stdout}")
        repro = float(m.group(1))
        diff = abs(repro - res["value"])
        if diff > tol:
            fail(f"reproduce mismatch: submitted={res['value']} got={repro} "
                 f"diff={diff} > tol={tol}")
        print(f"OK   reproduced {repro} (diff {diff:.2e} <= tol {tol})")
    else:
        print("SKIP reproduce (set REPRODUCE=1 to re-run the experiment)")

    # 4. pass-bar (informational; confirmation needs K independent repros).
    v, bar, lib = res["value"], goal["bar"], goal["lower_is_better"]
    beats = v < bar if lib else v > bar
    print(f"BAR  value={v} bar={bar} lower_is_better={lib} -> "
          f"{'BEATS BAR' if beats else 'below bar'}")

    print("VERDICT: accepted")


if __name__ == "__main__":
    main()
