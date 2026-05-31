#!/usr/bin/env python3
"""Run a small sequential queue of shell commands (one research thread per queue).

Each job is a JSON object on its own line with fields:
  - name: required display name
  - cmd:  required shell command string
  - cwd:  optional working directory
  - env:  optional dict of environment overrides
  - kind: optional, "run" (default) or "pause" (wait for a human decision)
  - force: optional bool; if true, run even if the job's output already exists
  - done_when: optional path; if it exists the job is treated as already done
               (defaults to "<--output_dir>/metrics.json" parsed from cmd)

The runner executes jobs sequentially, blocking on each subprocess until it
exits. Because execution is blocking, no idle-polling is needed in the normal
case (one runner owns the GPU). Idle-waiting is therefore OFF by default and is
only useful when launching alongside an externally-started training process.

Resume: a job whose output already exists is skipped, so re-running a queue
after a disconnect continues where it left off instead of clobbering results.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a sequential research queue")
    parser.add_argument("--queue", required=True, help="Path to JSONL queue file")
    parser.add_argument(
        "--status-log",
        default="logs/job_queue_status.log",
        help="Path to append job status updates",
    )
    parser.add_argument(
        "--default-cwd",
        default=os.getcwd(),
        help="Fallback working directory if a job omits cwd",
    )
    parser.add_argument(
        "--log-dir",
        default="logs/job_queue",
        help="Directory for per-job logs",
    )
    parser.add_argument(
        "--wait-for-idle",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Wait until no external process matches --idle-pattern before each job "
        "(default: off; the runner already blocks on each job sequentially)",
    )
    parser.add_argument(
        "--idle-pattern",
        default="train_llm.py",
        help="Process pattern to watch when --wait-for-idle is set",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=30,
        help="Seconds to sleep between idle/decision polls",
    )
    parser.add_argument(
        "--stop-on-failure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop the queue as soon as one job fails (default: on)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run every job even if its output already exists (disables resume)",
    )
    return parser.parse_args()


def load_jobs(queue_path: Path) -> list[Dict[str, Any]]:
    jobs: list[Dict[str, Any]] = []
    with queue_path.open() as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                job = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"Bad job on line {line_no}: {e}")
            if "name" not in job or ("cmd" not in job and job.get("kind") != "pause"):
                raise SystemExit(f"Job on line {line_no} missing name/cmd")
            jobs.append(job)
    return jobs


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(message + "\n")


def is_idle(pattern: str) -> bool:
    """True if no process matching `pattern` runs OUTSIDE our own process group.

    The historic deadlock: the old check shelled out to
    `bash -lc "pgrep -af train_llm.py"`, and that wrapper's own command line
    contains the pattern, so pgrep always matched itself -> never idle.
    We fix it by ignoring every PID in our own process group (the runner, its
    shell children, and the pgrep call), leaving only genuinely external jobs.
    """
    try:
        out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    except FileNotFoundError:
        return True  # no pgrep available -> assume idle
    my_pgid = os.getpgrp()
    my_pid = os.getpid()
    for tok in out.stdout.split():
        try:
            pid = int(tok)
        except ValueError:
            continue
        if pid == my_pid:
            continue
        try:
            if os.getpgid(pid) == my_pgid:
                continue  # part of our own process group, not an external job
        except ProcessLookupError:
            continue
        return False
    return True


def _output_dir_from_cmd(cmd: str) -> Optional[str]:
    m = re.search(r"--output[-_]dir[=\s]+(\S+)", cmd)
    return m.group(1) if m else None


def job_already_done(job: Dict[str, Any], default_cwd: str, force: bool) -> bool:
    if force or job.get("force"):
        return False
    cwd = job.get("cwd", default_cwd)
    marker = job.get("done_when")
    if marker:
        p = Path(marker)
    else:
        out = _output_dir_from_cmd(job.get("cmd", ""))
        if not out:
            return False
        p = Path(out) / "metrics.json"
    if not p.is_absolute():
        p = Path(cwd) / p
    return p.exists()


def wait_for_human_decision(
    job: Dict[str, Any], status_log: Path, poll_seconds: int
) -> str:
    """Block until a decision file is written. Returns 'continue' or a stop reason."""
    name = job["name"]
    decision_file = Path(job.get("decision_file", f"logs/decisions/{name}"))
    decision_file.parent.mkdir(parents=True, exist_ok=True)
    continue_values = {str(v).lower() for v in job.get("continue_values", ["continue", "go", "yes"])}
    stop_values = {str(v).lower() for v in job.get("stop_values", ["stop", "abort", "skip"])}
    append_log(status_log, f"PAUSE {name} release_with='echo continue > {decision_file}'")
    while True:
        if decision_file.exists():
            decision = decision_file.read_text().strip().lower()
            if decision in continue_values:
                return "continue"
            if decision in stop_values:
                return decision
        time.sleep(poll_seconds)


def run_job(job: Dict[str, Any], default_cwd: str, log_dir: Path, status_log: Path) -> int:
    name = job["name"]
    cmd = job["cmd"]
    cwd = job.get("cwd", default_cwd)
    env = os.environ.copy()
    if job.get("env"):
        env.update({str(k): str(v) for k, v in job["env"].items()})
    log_path = log_dir / f"{name}.log"
    log_dir.mkdir(parents=True, exist_ok=True)
    append_log(status_log, f"START {name} {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    append_log(status_log, f"CMD {name} {cmd}")
    append_log(status_log, f"CWD {name} {cwd}")
    with log_path.open("w") as log_file:
        proc = subprocess.run(
            ["bash", "-lc", cmd],
            cwd=cwd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    rc = proc.returncode
    status = "END" if rc == 0 else "FAIL"
    append_log(status_log, f"{status} {name} {time.strftime('%Y-%m-%dT%H:%M:%S%z')} rc={rc}")
    return rc


def main() -> int:
    args = parse_args()
    queue_path = Path(args.queue)
    status_log = Path(args.status_log)
    log_dir = Path(args.log_dir)
    jobs = load_jobs(queue_path)

    append_log(status_log, f"QUEUE START {time.strftime('%Y-%m-%dT%H:%M:%S%z')} queue={queue_path}")
    for job in jobs:
        kind = job.get("kind", "run")
        if kind == "pause":
            decision = wait_for_human_decision(job, status_log, args.poll_seconds)
            if decision != "continue":
                append_log(status_log, f"QUEUE STOP {job['name']} decision={decision}")
                return 0
            continue

        if job_already_done(job, args.default_cwd, args.force):
            append_log(status_log, f"SKIP {job['name']} (output exists)")
            continue

        if args.wait_for_idle:
            while not is_idle(args.idle_pattern):
                append_log(
                    status_log,
                    f"WAIT {job['name']} busy_with={args.idle_pattern} poll={args.poll_seconds}s",
                )
                time.sleep(args.poll_seconds)

        rc = run_job(job, args.default_cwd, log_dir, status_log)
        if rc != 0 and args.stop_on_failure:
            append_log(status_log, f"QUEUE STOP {job['name']} failed rc={rc}")
            return rc

    append_log(status_log, f"QUEUE DONE {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
