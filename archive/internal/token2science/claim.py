#!/usr/bin/env python3
"""Local task lease manager for token2science.

The simulator exercises this CLI with:

  python claim.py claim --task T --worker W --lease N
  python claim.py release --task T --worker W

Claims live under CLAIMS_DIR and are guarded by a per-task file lock so two
workers cannot update the same task state at the same time.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time


ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CLAIMS_DIR = os.path.join(ROOT, "claims")
DEFAULT_LEASE_SECONDS = 60.0 * 60.0


def now():
    return time.time()


def claims_dir():
    return os.path.abspath(os.environ.get("CLAIMS_DIR", DEFAULT_CLAIMS_DIR))


def ensure_claims_dir():
    os.makedirs(claims_dir(), exist_ok=True)


def task_paths(task):
    base = claims_dir()
    return (
        os.path.join(base, task),
        os.path.join(base, f"{task}.lock"),
    )


def state_path(task):
    task_dir, _ = task_paths(task)
    return os.path.join(task_dir, "claim.json")


def read_state(task):
    path = state_path(task)
    if not os.path.exists(path):
        return None

    try:
        with open(path) as f:
            state = json.load(f)
    except Exception:
        return None

    if not isinstance(state, dict):
        return None

    return state


def state_active(state):
    if not state:
        return False

    try:
        expires_at = float(state["lease_expires_at"])
    except Exception:
        return False

    return now() < expires_at


def write_state(task, payload):
    path = state_path(task)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, path)


def remove_task_dir(task):
    task_dir, _ = task_paths(task)
    try:
        shutil.rmtree(task_dir)
    except FileNotFoundError:
        pass
    except NotADirectoryError:
        try:
            os.remove(task_dir)
        except OSError:
            pass
    except OSError:
        pass


def acquire_task_lock(task):
    _, lock_path = task_paths(task)
    ensure_claims_dir()
    while True:
        try:
            os.mkdir(lock_path)
            return lock_path
        except FileExistsError:
            time.sleep(0.01)


def release_task_lock(lock_path):
    try:
        os.rmdir(lock_path)
    except OSError:
        pass


def read_active_holder(task):
    state = read_state(task)
    if state_active(state):
        return state
    return None


def claim_task(task, worker, lease_seconds):
    lock_path = acquire_task_lock(task)
    try:
        task_dir, _ = task_paths(task)
        while True:
            try:
                os.mkdir(task_dir)
            except FileExistsError:
                state = read_state(task)
                if state is None or not state_active(state):
                    remove_task_dir(task)
                    continue

                if str(state.get("worker")) != worker:
                    print(json.dumps(state, sort_keys=True))
                    return 3

                acquired_at = now()
                payload = {
                    "task": task,
                    "worker": worker,
                    "acquired_at": acquired_at,
                    "lease_seconds": float(lease_seconds),
                    "lease_expires_at": acquired_at + float(lease_seconds),
                }
                write_state(task, payload)
                print("acquired")
                return 0
            else:
                acquired_at = now()
                payload = {
                    "task": task,
                    "worker": worker,
                    "acquired_at": acquired_at,
                    "lease_seconds": float(lease_seconds),
                    "lease_expires_at": acquired_at + float(lease_seconds),
                }
                write_state(task, payload)
                print("acquired")
                return 0
    finally:
        release_task_lock(lock_path)


def release_task(task, worker):
    lock_path = acquire_task_lock(task)
    try:
        state = read_state(task)
        if state is None or not state_active(state):
            remove_task_dir(task)
            print("released")
            return 0

        if str(state.get("worker")) != worker:
            print("held")
            return 4

        remove_task_dir(task)
        print("released")
        return 0
    finally:
        release_task_lock(lock_path)


def status_task(task, worker):
    lock_path = acquire_task_lock(task)
    try:
        state = read_state(task)
        if state_active(state):
            holder = str(state.get("worker", "unknown"))
            remaining = max(0.0, float(state["lease_expires_at"]) - now())
            if holder == worker:
                print(f"held by you ({remaining:.3f}s left)")
            else:
                print(f"held by {holder} ({remaining:.3f}s left)")
            return 3

        if state is not None:
            remove_task_dir(task)

        print("free")
        return 0
    finally:
        release_task_lock(lock_path)


def sweep_task():
    ensure_claims_dir()
    removed = 0
    for entry in sorted(os.scandir(claims_dir()), key=lambda e: e.name):
        if not entry.is_dir(follow_symlinks=False):
            continue
        if entry.name.endswith(".lock"):
            continue

        task = entry.name
        lock_path = acquire_task_lock(task)
        try:
            state = read_state(task)
            if state is None:
                remove_task_dir(task)
                continue

            if not state_active(state):
                remove_task_dir(task)
                removed += 1
        finally:
            release_task_lock(lock_path)

    print(removed)
    return 0


def main():
    ap = argparse.ArgumentParser(prog="claim.py")
    sub = ap.add_subparsers(dest="cmd", required=True)

    claim = sub.add_parser("claim")
    claim.add_argument("--task", required=True)
    claim.add_argument("--worker", required=True)
    claim.add_argument("--lease", type=float, default=DEFAULT_LEASE_SECONDS)

    status = sub.add_parser("status")
    status.add_argument("--task", required=True)
    status.add_argument("--worker", required=True)

    release = sub.add_parser("release")
    release.add_argument("--task", required=True)
    release.add_argument("--worker", required=True)

    sub.add_parser("sweep")

    args = ap.parse_args()

    if args.cmd == "claim":
        return claim_task(args.task, args.worker, args.lease)
    if args.cmd == "status":
        return status_task(args.task, args.worker)
    if args.cmd == "release":
        return release_task(args.task, args.worker)
    if args.cmd == "sweep":
        return sweep_task()
    return 1


if __name__ == "__main__":
    sys.exit(main())
