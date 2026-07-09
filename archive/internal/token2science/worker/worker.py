#!/usr/bin/env python3
"""token2science worker CLI (Phase-1 skeleton).

Bring-your-own-compute: this runs on the contributor's machine and spends the
contributor's tokens. Two verbs:

  claim   pick an open task issue and self-assign it (needs `gh`)
  submit  run the task's experiment, write the run artifact, print PR steps

`submit` is fully local and needs no network: it runs the experiment, parses the
RESULT line, hashes the config, and writes runs/<task>/<run>/result.json. The
GitHub steps (branch, PR) are printed for you / your agent to execute.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # token2science/


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return "sha256:" + h.hexdigest()


def have_gh():
    return subprocess.run(["which", "gh"], capture_output=True).returncode == 0


def claim(args):
    if not have_gh():
        print("gh CLI not found. Install it, or claim a task by self-assigning "
              "an issue labeled `task:open` in the GitHub UI.")
        return 1
    out = subprocess.run(
        ["gh", "issue", "list", "--label", "task:open", "--limit", "1",
         "--json", "number,title"],
        capture_output=True, text=True)
    issues = json.loads(out.stdout or "[]")
    if not issues:
        print("no open tasks")
        return 0
    num = issues[0]["number"]
    subprocess.run(["gh", "issue", "edit", str(num), "--add-assignee", "@me"])
    print(f"claimed task issue #{num}: {issues[0]['title']}")
    return 0


def submit(args):
    task_dir = os.path.join(ROOT, "goals", args.goal, "tasks", args.task)
    if not os.path.isdir(task_dir):
        print(f"task folder not found: {task_dir}")
        return 1

    cfg_path = os.path.join(task_dir, args.config)
    cfg = json.load(open(cfg_path))
    command = f"python experiment.py --config {args.config}"

    # run the experiment (this is where the contributor's compute is spent)
    out = subprocess.run(command, cwd=task_dir, shell=True,
                         capture_output=True, text=True)
    log = out.stdout + out.stderr
    if out.returncode != 0:
        print(f"experiment failed:\n{log}")
        return 1
    m = re.search(r"RESULT\b.*?metric=(\S+)\s+value=([-+0-9.eE]+)", out.stdout)
    if not m:
        print(f"no RESULT line in output:\n{out.stdout}")
        return 1
    metric, value = m.group(1), float(m.group(2))

    goal = json.load(open(os.path.join(ROOT, "goals", args.goal, "goal.json")))
    run_id = args.run or f"{args.worker}-{int(time.time())}"
    run_dir = os.path.join(ROOT, "runs", args.task, run_id)
    os.makedirs(run_dir, exist_ok=True)
    tolerance = float(goal.get("tolerance", args.tolerance))

    result = {
        "task_id": args.task,
        "goal_id": args.goal,
        "worker": args.worker,
        "metric": metric,
        "value": value,
        "lower_is_better": goal["lower_is_better"],
        "seed": cfg.get("seed"),
        "config_path": args.config,
        "config_hash": sha256_file(cfg_path),
        "command": command,
        "tolerance": tolerance,
    }
    with open(os.path.join(run_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    with open(os.path.join(run_dir, "run.log"), "w") as f:
        f.write(log)

    rel = os.path.relpath(run_dir, ROOT)
    print(f"wrote {rel}/result.json  (metric={metric} value={value})")
    print("\nnext:")
    print(f"  git checkout -b run/{args.task}/{run_id}")
    print(f"  git add {rel}")
    print(f'  git commit -m "submit {args.task} run {run_id}"')
    print("  gh pr create --fill   # CI verifies (REPRODUCE=1) and posts verdict")
    return 0


def main():
    ap = argparse.ArgumentParser(prog="worker")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("claim").set_defaults(func=claim)

    s = sub.add_parser("submit")
    s.add_argument("--goal", required=True)
    s.add_argument("--task", required=True)
    s.add_argument("--worker", required=True, help="your GitHub handle")
    s.add_argument("--config", default="config.json")
    s.add_argument("--run", default=None, help="run id (default: worker-timestamp)")
    s.add_argument("--tolerance", type=float, default=1e-9)
    s.set_defaults(func=submit)

    args = ap.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
