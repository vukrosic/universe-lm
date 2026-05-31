#!/usr/bin/env python3
"""Run a small sequential queue of arbitrary shell commands.

Each job is a JSON object on its own line with fields:
  - name: required display name
  - cmd: required shell command string
  - cwd: optional working directory
  - env: optional dict of environment overrides

The runner waits for the machine to go idle before each job by default so it
can be launched while a previous training job is still winding down.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a sequential training queue")
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
        action="store_true",
        default=True,
        help="Wait until no train_llm.py process is running before each job",
    )
    parser.add_argument(
        "--idle-pattern",
        default="train_llm.py",
        help="Process pattern to watch before launching each job",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=30,
        help="Seconds to sleep between idle checks",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        default=True,
        help="Stop the queue as soon as one job fails",
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
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON on line {line_no} of {queue_path}: {exc}") from exc
            kind = job.get("kind", "run")
            if "name" not in job:
                raise SystemExit(f"Queue job on line {line_no} must include name")
            if kind == "run" and "cmd" not in job:
                raise SystemExit(f"Queue job on line {line_no} must include cmd unless kind=pause")
            if kind not in {"run", "pause"}:
                raise SystemExit(f"Queue job on line {line_no} has unknown kind {kind!r}")
            jobs.append(job)
    if not jobs:
        raise SystemExit(f"No jobs found in {queue_path}")
    return jobs


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(message.rstrip() + "\n")
        f.flush()


def is_idle(pattern: str) -> bool:
    proc = subprocess.run(
        ["bash", "-lc", f"pgrep -af {shlex_quote(pattern)} || true"],
        capture_output=True,
        text=True,
    )
    lines = [
        line
        for line in proc.stdout.splitlines()
        if pattern in line and "pgrep -af" not in line and "run_job_queue.py" not in line
    ]
    return len(lines) == 0


def shlex_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def current_git_metadata() -> Dict[str, Optional[str]]:
    def run(cmd: Iterable[str]) -> Optional[str]:
        try:
            return subprocess.run(
                list(cmd),
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except Exception:
            return None

    return {
        "git_commit": run(["git", "rev-parse", "HEAD"]),
        "git_branch": run(["git", "branch", "--show-current"]),
    }


def wait_for_human_decision(job: Dict[str, Any], status_log: Path, poll_seconds: int) -> str:
    name = job["name"]
    decision_file = Path(job.get("decision_file") or f"queues/{name}.decision")
    message = job.get("message") or "Awaiting human decision"
    continue_values = {str(v).lower() for v in job.get("continue_values", ["continue", "go", "yes"])}
    stop_values = {str(v).lower() for v in job.get("stop_values", ["stop", "abort", "skip"])}

    append_log(status_log, f"PAUSE {name} message={message}")
    append_log(status_log, f"PAUSE {name} decision_file={decision_file}")
    append_log(status_log, f"PAUSE {name} release_with='echo continue > {decision_file}'")
    append_log(status_log, f"PAUSE {name} stop_with='echo stop > {decision_file}'")

    while True:
        if decision_file.exists():
            decision = decision_file.read_text().strip().lower()
            if decision in continue_values:
                append_log(status_log, f"RESUME {name} decision={decision}")
                return "continue"
            if decision in stop_values:
                append_log(status_log, f"STOP {name} decision={decision}")
                return "stop"
            append_log(status_log, f"PAUSE {name} invalid_decision={decision!r}")
        time.sleep(poll_seconds)


def run_job(job: Dict[str, Any], default_cwd: str, log_dir: Path, status_log: Path) -> int:
    name = job["name"]
    cmd = job["cmd"]
    cwd = job.get("cwd") or default_cwd
    env = os.environ.copy()
    env.update({k: str(v) for k, v in job.get("env", {}).items()})
    env.update({f"QUEUE_{k.upper()}": v for k, v in current_git_metadata().items() if v})
    log_path = Path(job.get("log_file") or log_dir / f"{name}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    append_log(status_log, f"START {name} {start}")
    append_log(status_log, f"CMD {name} {cmd}")
    append_log(status_log, f"CWD {name} {cwd}")
    with log_path.open("a") as log_f:
        log_f.write(f"### START {name} {start}\n")
        log_f.write(f"### CMD {cmd}\n")
        log_f.write(f"### CWD {cwd}\n")
        log_f.flush()
        proc = subprocess.Popen(
            cmd,
            shell=True,
            cwd=cwd,
            env=env,
            executable="/bin/bash",
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )
        rc = proc.wait()
        end = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        log_f.write(f"\n### END {name} {end} rc={rc}\n")
        log_f.flush()
    append_log(status_log, f"END {name} {end} rc={rc}")
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
