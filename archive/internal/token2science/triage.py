#!/usr/bin/env python3
"""Automated gate for proposed token2science goals.

Usage:
    python token2science/triage.py --goal token2science/goals/GOAL_DIR

Stdlib only. Prints a PASS/FAIL checklist line per check and a final summary.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


REQUIRED_KEYS = {
    "goal_id": str,
    "title": str,
    "metric": str,
    "bar": (int, float),
    "lower_is_better": bool,
    "status": str,
}


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def format_result(ok, label, detail=""):
    prefix = "PASS" if ok else "FAIL"
    if detail:
        print(f"{prefix} {label}: {detail}")
    else:
        print(f"{prefix} {label}")


def main():
    ap = argparse.ArgumentParser(prog="triage")
    ap.add_argument("--goal", required=True, help="path to a goal directory")
    args = ap.parse_args()

    goal_dir = os.path.abspath(args.goal)
    checks_ok = []

    goal_json_path = os.path.join(goal_dir, "goal.json")
    goal_md_path = os.path.join(goal_dir, "goal.md")
    tasks_dir = os.path.join(goal_dir, "tasks")

    # 1. goal.json exists.
    ok = os.path.isfile(goal_json_path)
    format_result(ok, "goal.json exists", goal_json_path if not ok else goal_json_path)
    checks_ok.append(ok)

    goal_data = None
    if ok:
        # 2. goal.json parses.
        try:
            with open(goal_json_path) as f:
                goal_data = json.load(f)
        except Exception as exc:
            format_result(False, "goal.json parses", str(exc))
            checks_ok.append(False)
        else:
            format_result(True, "goal.json parses")
            checks_ok.append(True)
    else:
        format_result(False, "goal.json parses", "missing goal.json")
        checks_ok.append(False)

    # 3. goal.json schema.
    schema_ok = False
    if isinstance(goal_data, dict):
        missing = [key for key in REQUIRED_KEYS if key not in goal_data]
        wrong_types = []
        if not missing:
            for key, expected in REQUIRED_KEYS.items():
                value = goal_data[key]
                if key == "bar":
                    if not is_number(value):
                        wrong_types.append(f"{key}={type(value).__name__}")
                elif not isinstance(value, expected):
                    wrong_types.append(f"{key}={type(value).__name__}")
                elif key == "lower_is_better" and isinstance(value, bool) is False:
                    wrong_types.append(f"{key}={type(value).__name__}")

        if not missing and not wrong_types:
            schema_ok = True
            format_result(True, "goal.json schema")
        else:
            detail_parts = []
            if missing:
                detail_parts.append("missing keys: " + ", ".join(missing))
            if wrong_types:
                detail_parts.append("wrong types: " + ", ".join(wrong_types))
            format_result(False, "goal.json schema", "; ".join(detail_parts))
    else:
        format_result(False, "goal.json schema", "goal.json is not an object")
    checks_ok.append(schema_ok)

    # 4. goal.md exists and mentions baseline.
    md_ok = False
    if os.path.isfile(goal_md_path):
        try:
            with open(goal_md_path) as f:
                goal_md = f.read()
        except Exception as exc:
            format_result(False, "goal.md exists and mentions baseline", str(exc))
        else:
            if "baseline" in goal_md.lower():
                md_ok = True
                format_result(True, "goal.md exists and mentions baseline")
            else:
                format_result(False, "goal.md exists and mentions baseline", "missing 'baseline'")
    else:
        format_result(False, "goal.md exists and mentions baseline", "missing goal.md")
    checks_ok.append(md_ok)

    # 5. tasks/ contains at least one task folder and each has task.md.
    tasks_ok = False
    if os.path.isdir(tasks_dir):
        task_dirs = [
            name
            for name in sorted(os.listdir(tasks_dir))
            if os.path.isdir(os.path.join(tasks_dir, name))
        ]
        if not task_dirs:
            format_result(False, "tasks contain task.md", "no task folders found")
        else:
            missing_task_md = [
                name
                for name in task_dirs
                if not os.path.isfile(os.path.join(tasks_dir, name, "task.md"))
            ]
            if missing_task_md:
                format_result(
                    False,
                    "tasks contain task.md",
                    "missing task.md in: " + ", ".join(missing_task_md),
                )
            else:
                tasks_ok = True
                format_result(
                    True,
                    "tasks contain task.md",
                    f"{len(task_dirs)} task folder(s)",
                )
    else:
        format_result(False, "tasks contain task.md", "missing tasks/ directory")
    checks_ok.append(tasks_ok)

    summary = all(checks_ok)
    print(f"SUMMARY {'PASS' if summary else 'FAIL'}")
    return 0 if summary else 1


if __name__ == "__main__":
    sys.exit(main())
