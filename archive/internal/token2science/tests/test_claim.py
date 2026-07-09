import json
import os
import subprocess
import sys

import pytest


TOKEN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_claim(args, claims_dir):
    env = os.environ.copy()
    env["CLAIMS_DIR"] = claims_dir
    return subprocess.run(
        [sys.executable, "claim.py", *args],
        cwd=TOKEN_DIR,
        env=env,
        capture_output=True,
        text=True,
    )


def test_second_claim_on_held_task_exits_3(tmp_path):
    claims_dir = str(tmp_path / "claims")
    first = run_claim(["claim", "--task", "T1", "--worker", "alice"], claims_dir)
    assert first.returncode == 0
    assert first.stdout.strip() == "acquired"

    second = run_claim(["claim", "--task", "T1", "--worker", "bob"], claims_dir)
    assert second.returncode == 3
    holder = json.loads(second.stdout)
    assert holder["worker"] == "alice"


def test_release_by_non_holder_exits_4(tmp_path):
    claims_dir = str(tmp_path / "claims")
    first = run_claim(["claim", "--task", "T2", "--worker", "alice"], claims_dir)
    assert first.returncode == 0

    release = run_claim(["release", "--task", "T2", "--worker", "bob"], claims_dir)
    assert release.returncode == 4


def test_zero_lease_can_be_taken_over(tmp_path):
    claims_dir = str(tmp_path / "claims")
    first = run_claim(["claim", "--task", "T3", "--worker", "alice", "--lease", "0"], claims_dir)
    assert first.returncode == 0

    second = run_claim(["claim", "--task", "T3", "--worker", "bob"], claims_dir)
    assert second.returncode == 0
    assert second.stdout.strip() == "acquired"


def test_sweep_removes_expired(tmp_path):
    claims_dir = str(tmp_path / "claims")
    first = run_claim(["claim", "--task", "T4", "--worker", "alice", "--lease", "0"], claims_dir)
    assert first.returncode == 0

    sweep = run_claim(["sweep"], claims_dir)
    assert sweep.returncode == 0
    assert sweep.stdout.strip() == "1"
    assert not os.path.exists(os.path.join(claims_dir, "T4"))
