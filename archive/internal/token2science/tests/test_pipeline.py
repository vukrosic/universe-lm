import hashlib
import json
import os
import re
import shutil
import subprocess

import pytest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOAL_ID = "G001-deterministic-demo"
TASK_ID = "T001"
POSITIVE_RUN = os.path.join(ROOT, "runs", TASK_ID, "pytest-tmp")
NEGATIVE_RUN = os.path.join(ROOT, "runs", "T101", "pytest-bad")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return "sha256:" + h.hexdigest()


def cleanup_run_dirs():
    for path in (POSITIVE_RUN, NEGATIVE_RUN):
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(autouse=True)
def _clean_runs():
    cleanup_run_dirs()
    try:
        yield
    finally:
        cleanup_run_dirs()


def run_cmd(args, env=None, cwd=ROOT):
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(
        args,
        cwd=cwd,
        env=merged_env,
        capture_output=True,
        text=True,
    )


def load_json(path):
    with open(path) as f:
        return json.load(f)


def build_valid_result():
    task_dir = os.path.join(ROOT, "goals", GOAL_ID, "tasks", TASK_ID)
    cfg_path = os.path.join(task_dir, "config.json")
    goal_path = os.path.join(ROOT, "goals", GOAL_ID, "goal.json")

    proc = run_cmd(
        ["python", "experiment.py", "--config", "config.json"],
        cwd=task_dir,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    match = re.search(r"RESULT\b.*?metric=(\S+)\s+value=([-+0-9.eE]+)", proc.stdout)
    assert match, proc.stdout

    cfg = load_json(cfg_path)
    goal = load_json(goal_path)
    return {
        "task_id": TASK_ID,
        "goal_id": GOAL_ID,
        "worker": "pytest-bot",
        "metric": match.group(1),
        "value": float(match.group(2)),
        "lower_is_better": goal["lower_is_better"],
        "seed": cfg.get("seed"),
        "config_path": "config.json",
        "config_hash": sha256_file(cfg_path),
        "command": "python experiment.py --config config.json",
        "tolerance": 1e-9,
    }


def test_submit_then_verify_accepts():
    submit = run_cmd([
        "python",
        "worker/worker.py",
        "submit",
        "--goal",
        GOAL_ID,
        "--task",
        TASK_ID,
        "--worker",
        "pytest-bot",
        "--run",
        "pytest-tmp",
    ])
    assert submit.returncode == 0, submit.stdout + submit.stderr

    verify = run_cmd(
        ["python", "verify/verify.py", "--run", "runs/T001/pytest-tmp"],
        env={"REPRODUCE": "1"},
    )
    assert verify.returncode == 0, verify.stdout + verify.stderr
    assert "VERDICT: accepted" in verify.stdout


def test_tampered_run_is_rejected():
    os.makedirs(NEGATIVE_RUN, exist_ok=True)
    result = build_valid_result()
    result["value"] = 1234567.0

    with open(os.path.join(NEGATIVE_RUN, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")

    verify = run_cmd(
        ["python", "verify/verify.py", "--run", "runs/T101/pytest-bad"],
        env={"REPRODUCE": "1"},
    )
    assert verify.returncode != 0, verify.stdout + verify.stderr
    assert "VERDICT: rejected" in verify.stdout
