#!/usr/bin/env python3
"""Autonomous compute donor loop for token2science.

The runner scans open goals, finds runnable tasks, and spends donated compute
to fill K-replication gaps. It is distinct from worker.py: worker.py submits a
single task, while this loop keeps picking the task with the fewest distinct
supporters until it reaches the requested round limit or nothing needs this
worker anymore.

Stdlib only.
"""

import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import time


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOALS_DIR = os.path.join(ROOT, "goals")
RUNS_DIR = os.path.join(ROOT, "runs")
DEFAULT_TOLERANCE = 1e-9
RESULT_RE = re.compile(r"^RESULT\s+metric=(\S+)\s+value=([-+0-9.eE]+)\s*$")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return "sha256:" + h.hexdigest()


def load_json(path):
    with open(path) as f:
        return json.load(f)


def valid_goal(goal_path):
    if not os.path.isfile(goal_path):
        return False, None
    try:
        goal = load_json(goal_path)
    except Exception:
        return False, None
    return goal.get("status") == "open", goal


def readable_result(path, expected_task_id):
    try:
        res = load_json(path)
    except Exception:
        return None
    if not isinstance(res, dict):
        return None

    task_id = res.get("task_id")
    if task_id not in (None, expected_task_id):
        return None

    worker = res.get("worker")
    config_hash = res.get("config_hash")
    value = res.get("value")
    metric = res.get("metric")
    lower_is_better = res.get("lower_is_better")
    if (
        worker is None
        or config_hash is None
        or value is None
        or metric is None
        or not isinstance(lower_is_better, bool)
    ):
        return None

    try:
        value = float(value)
    except Exception:
        return None

    return {
        "path": path,
        "worker": str(worker),
        "config_hash": str(config_hash),
        "value": value,
        "metric": str(metric),
        "lower_is_better": lower_is_better,
    }


def load_task_results(task_id):
    results = []
    pattern = os.path.join(RUNS_DIR, task_id, "*", "result.json")
    for path in sorted(glob.glob(pattern)):
        res = readable_result(path, task_id)
        if res is not None:
            results.append(res)
    return results


def cluster_reproducing_runs(results, config_hash, tolerance):
    rows = [
        res
        for res in results
        if res["config_hash"] == config_hash
    ]
    if not rows:
        return []

    rows.sort(key=lambda r: (r["value"], r["worker"], r["path"]))
    groups = []
    current = None

    for res in rows:
        if current is None:
            current = {
                "min_value": res["value"],
                "runs": [res],
                "workers": {res["worker"]},
            }
            continue

        if res["value"] - current["min_value"] <= tolerance:
            current["runs"].append(res)
            current["workers"].add(res["worker"])
        else:
            groups.append(current)
            current = {
                "min_value": res["value"],
                "runs": [res],
                "workers": {res["worker"]},
            }

    if current is not None:
        groups.append(current)
    return groups


def task_support(task_dir, goal, worker_name, k, tolerance):
    task_id = os.path.basename(task_dir)
    goal_tolerance = float(goal.get("tolerance", tolerance))
    task_results = [
        res
        for res in load_task_results(task_id)
        if res["metric"] == goal["metric"]
        and res["lower_is_better"] == goal["lower_is_better"]
    ]
    if any(res["worker"] == worker_name for res in task_results):
        return None

    cfg_path = os.path.join(task_dir, "config.json")
    if not os.path.isfile(cfg_path):
        return None

    config_hash = sha256_file(cfg_path)
    groups = cluster_reproducing_runs(task_results, config_hash, goal_tolerance)
    support = max((len(group["workers"]) for group in groups), default=0)
    return {
        "task_dir": task_dir,
        "task_id": task_id,
        "config_hash": config_hash,
        "support": support,
        "needs_worker": support < k,
    }


def find_runnable_tasks(worker_name, k, tolerance):
    candidates = []
    if not os.path.isdir(GOALS_DIR):
        return candidates

    for goal_name in sorted(os.listdir(GOALS_DIR)):
        goal_dir = os.path.join(GOALS_DIR, goal_name)
        if not os.path.isdir(goal_dir):
            continue

        goal_path = os.path.join(goal_dir, "goal.json")
        goal_ok, goal = valid_goal(goal_path)
        if not goal_ok:
            continue

        tasks_dir = os.path.join(goal_dir, "tasks")
        if not os.path.isdir(tasks_dir):
            continue

        for task_name in sorted(os.listdir(tasks_dir)):
            task_dir = os.path.join(tasks_dir, task_name)
            if not os.path.isdir(task_dir):
                continue
            if not os.path.isfile(os.path.join(task_dir, "experiment.py")):
                continue
            if not os.path.isfile(os.path.join(task_dir, "config.json")):
                continue

            info = task_support(task_dir, goal, worker_name, k, tolerance)
            if info is None or not info["needs_worker"]:
                continue

            info["goal_id"] = goal.get("goal_id", goal_name)
            candidates.append(info)

    return candidates


def choose_task(candidates):
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (
            item["support"],
            item["goal_id"],
            item["task_id"],
        ),
    )


def parse_result_line(stdout):
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        return None
    match = RESULT_RE.match(lines[-1])
    if not match:
        return None
    return match.group(1), float(match.group(2))


def run_task(task_info, worker_name):
    task_dir = task_info["task_dir"]
    goal_dir = os.path.dirname(os.path.dirname(task_dir))
    goal = load_json(os.path.join(goal_dir, "goal.json"))
    cfg_path = os.path.join(task_dir, "config.json")
    cfg = load_json(cfg_path)

    command = ["python", "experiment.py", "--config", "config.json"]
    proc = subprocess.run(
        command,
        cwd=task_dir,
        capture_output=True,
        text=True,
    )
    log = proc.stdout + proc.stderr
    run_id = f"{worker_name}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    run_dir = os.path.join(RUNS_DIR, task_info["task_id"], run_id)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "run.log"), "w") as f:
        f.write(log)

    if proc.returncode != 0:
        return {
            "ok": False,
            "run_dir": run_dir,
            "run_id": run_id,
            "error": f"experiment failed with exit code {proc.returncode}",
            "log": log,
        }

    parsed = parse_result_line(proc.stdout)
    if parsed is None:
        return {
            "ok": False,
            "run_dir": run_dir,
            "run_id": run_id,
            "error": "missing RESULT line on the last output line",
            "log": log,
        }

    metric, value = parsed
    result = {
        "task_id": task_info["task_id"],
        "goal_id": goal.get("goal_id", os.path.basename(goal_dir)),
        "worker": worker_name,
        "metric": metric,
        "value": value,
        "lower_is_better": goal["lower_is_better"],
        "seed": cfg.get("seed"),
        "config_path": "config.json",
        "config_hash": sha256_file(cfg_path),
        "command": "python experiment.py --config config.json",
        "tolerance": float(goal.get("tolerance", DEFAULT_TOLERANCE)),
    }
    with open(os.path.join(run_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")

    return {
        "ok": True,
        "run_dir": run_dir,
        "run_id": run_id,
        "metric": metric,
        "value": value,
        "task_id": task_info["task_id"],
        "goal_id": goal.get("goal_id", os.path.basename(goal_dir)),
    }


def main():
    ap = argparse.ArgumentParser(prog="runner")
    ap.add_argument("--worker", required=True, help="worker name or handle")
    ap.add_argument("--rounds", type=int, default=1, help="maximum rounds")
    ap.add_argument("--k", type=int, default=2, help="confirmation threshold")
    args = ap.parse_args()

    attempted = set()
    for _ in range(max(args.rounds, 0)):
        candidates = find_runnable_tasks(args.worker, args.k, DEFAULT_TOLERANCE)
        candidates = [item for item in candidates if item["task_id"] not in attempted]
        task_info = choose_task(candidates)
        if task_info is None:
            break

        attempted.add(task_info["task_id"])
        outcome = run_task(task_info, args.worker)
        rel = os.path.relpath(outcome["run_dir"], ROOT)
        if outcome["ok"]:
            print(
                f"wrote {rel}/result.json  "
                f"(metric={outcome['metric']} value={outcome['value']})"
            )
        else:
            print(f"FAILED {task_info['task_id']}  {rel}/run.log  {outcome['error']}")
            break

    return 0


if __name__ == "__main__":
    sys.exit(main())
