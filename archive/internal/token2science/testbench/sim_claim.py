#!/usr/bin/env python3
"""Stress-test the token2science claim lease.

The simulator launches multiple worker processes. Each worker repeatedly picks a
random task, tries to claim it through token2science/claim.py, sleeps briefly if
successful, releases it, and records the hold interval in a per-process audit
file.

After all workers finish, the merged audit is checked for overlapping hold
intervals per task. A successful run proves that the lock never let two workers
hold the same task at once during the simulation.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import random
import shutil
import subprocess
import sys
import time


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CLAIM_CLI = os.path.join(ROOT, "claim.py")
CLAIMS_DIR = os.path.join(HERE, ".tmp-claims")
AUDIT_DIR = os.path.join(HERE, ".tmp-audit")
LEASE_SECONDS = 2.0
SLEEP_MIN = 0.01
SLEEP_MAX = 0.05


def clear_dir(path):
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)


def task_name(index):
    return f"SIM-{index}"


def claim_task(task, worker):
    env = os.environ.copy()
    env["CLAIMS_DIR"] = CLAIMS_DIR
    proc = subprocess.run(
        [
            sys.executable,
            CLAIM_CLI,
            "claim",
            "--task",
            task,
            "--worker",
            worker,
            "--lease",
            str(LEASE_SECONDS),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    return proc


def release_task(task, worker):
    env = os.environ.copy()
    env["CLAIMS_DIR"] = CLAIMS_DIR
    proc = subprocess.run(
        [
            sys.executable,
            CLAIM_CLI,
            "release",
            "--task",
            task,
            "--worker",
            worker,
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    return proc


def append_audit_line(path, record):
    with open(path, "a") as f:
        json.dump(record, f, sort_keys=True)
        f.write("\n")


def worker_main(worker, rounds, tasks, result_queue):
    rng = random.Random(time.time_ns() ^ os.getpid() ^ hash(worker))
    audit_path = os.path.join(AUDIT_DIR, f"audit-{worker}.jsonl")
    acquires = 0
    rejections = 0
    error = None

    try:
        for _ in range(rounds):
            task = task_name(rng.randint(1, tasks))
            proc = claim_task(task, worker)

            if proc.returncode == 0:
                if "acquired" not in proc.stdout.lower():
                    raise RuntimeError(
                        f"claim succeeded without acquired output: {proc.stdout!r}"
                    )

                acquired_at = time.time()
                time.sleep(rng.uniform(SLEEP_MIN, SLEEP_MAX))
                released_at = time.time()

                release_proc = release_task(task, worker)
                if release_proc.returncode != 0:
                    raise RuntimeError(
                        f"release failed for {task}/{worker}: "
                        f"{release_proc.returncode} {release_proc.stdout!r} {release_proc.stderr!r}"
                    )

                append_audit_line(
                    audit_path,
                    {
                        "task": task,
                        "worker": worker,
                        "acquired_at": acquired_at,
                        "released_at": released_at,
                    },
                )
                acquires += 1
            elif proc.returncode == 3:
                rejections += 1
            else:
                raise RuntimeError(
                    f"claim failed for {task}/{worker}: "
                    f"{proc.returncode} {proc.stdout!r} {proc.stderr!r}"
                )
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"

    result_queue.put(
        {
            "worker": worker,
            "acquires": acquires,
            "rejections": rejections,
            "error": error,
        }
    )


def load_audits():
    intervals = {}
    total_acquires = 0
    audit_files = [
        os.path.join(AUDIT_DIR, name)
        for name in sorted(os.listdir(AUDIT_DIR))
        if name.endswith(".jsonl")
    ]

    for path in audit_files:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                task = str(record["task"])
                worker = str(record["worker"])
                acquired_at = float(record["acquired_at"])
                released_at = float(record["released_at"])
                intervals.setdefault(task, []).append(
                    (acquired_at, released_at, worker, path)
                )
                total_acquires += 1

    return intervals, total_acquires


def count_overlaps(intervals):
    overlaps = 0
    for task, rows in sorted(intervals.items()):
        rows.sort(key=lambda row: (row[0], row[1], row[2]))
        active_end = None
        active_owner = None
        for start, end, worker, _path in rows:
            if active_end is not None and start < active_end:
                overlaps += 1
            if active_end is None or end > active_end:
                active_end = end
                active_owner = worker
        _ = active_owner
    return overlaps


def main():
    ap = argparse.ArgumentParser(prog="sim_claim.py")
    ap.add_argument("--agents", type=int, default=20)
    ap.add_argument("--tasks", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=5)
    args = ap.parse_args()

    clear_dir(CLAIMS_DIR)
    clear_dir(AUDIT_DIR)

    previous_claims_dir = os.environ.get("CLAIMS_DIR")
    os.environ["CLAIMS_DIR"] = CLAIMS_DIR

    result_queue = multiprocessing.Queue()
    processes = []

    try:
        for idx in range(1, args.agents + 1):
            worker = f"W{idx:02d}"
            proc = multiprocessing.Process(
                target=worker_main,
                args=(worker, args.rounds, args.tasks, result_queue),
                name=worker,
            )
            proc.start()
            processes.append(proc)

        results = []
        for _ in processes:
            results.append(result_queue.get())

        for proc in processes:
            proc.join()

        errors = []
        for result in results:
            if result["error"]:
                errors.append(f'{result["worker"]}: {result["error"]}')
            if result["worker"] and result["acquires"] < 0:
                errors.append(f'{result["worker"]}: invalid acquire count')

        for proc in processes:
            if proc.exitcode not in (0, None):
                errors.append(f"{proc.name}: exitcode {proc.exitcode}")

        intervals, total_acquires = load_audits()
        total_rejections = sum(result["rejections"] for result in results)
        overlaps_detected = count_overlaps(intervals)

        print(
            "agents={agents} tasks={tasks} total_acquires={acquires} "
            "total_rejections={rejections} overlaps_detected={overlaps}".format(
                agents=args.agents,
                tasks=args.tasks,
                acquires=total_acquires,
                rejections=total_rejections,
                overlaps=overlaps_detected,
            )
        )

        if errors or overlaps_detected:
            for error in errors:
                print(f"ERROR {error}", file=sys.stderr)
            return 1
        return 0
    finally:
        if previous_claims_dir is None:
            os.environ.pop("CLAIMS_DIR", None)
        else:
            os.environ["CLAIMS_DIR"] = previous_claims_dir


if __name__ == "__main__":
    sys.exit(main())
