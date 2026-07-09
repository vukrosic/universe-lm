#!/usr/bin/env python3
"""Generate token2science/LEADERBOARD.md from goal and run artifacts.

The generator is stdlib-only and scans:
  token2science/goals/*/goal.json
  token2science/runs/*/*/result.json

It writes a compact markdown summary with contributor and goal tables.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
GOALS_DIR = ROOT / "goals"
RUNS_DIR = ROOT / "runs"
OUTPUT = ROOT / "LEADERBOARD.md"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def fmt_value(value):
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return format(value, ".15g")
    return str(value)


def better(value_a, value_b, lower_is_better):
    if value_b is None:
        return True
    if lower_is_better:
        return value_a < value_b
    return value_a > value_b


def collect_goals():
    goals = {}
    if not GOALS_DIR.exists():
        return goals
    for goal_path in sorted(GOALS_DIR.glob("*/goal.json")):
        goal = load_json(goal_path)
        goal_id = goal.get("goal_id") or goal_path.parent.name
        goals[goal_id] = goal
    return goals


def collect_runs():
    runs = []
    if not RUNS_DIR.exists():
        return runs
    for result_path in sorted(RUNS_DIR.glob("*/*/result.json")):
        run = load_json(result_path)
        run["__path"] = result_path
        run["__task_dir"] = result_path.parent.parent.name
        run["__run_dir"] = result_path.parent.name
        runs.append(run)
    return runs


def contributor_rows(runs):
    stats = {}
    for run in runs:
        worker = run.get("worker", "unknown")
        task_id = run.get("task_id", "unknown")
        value = run.get("value")
        lower_is_better = run.get("lower_is_better", True)
        cur = stats.setdefault(
            worker,
            {
                "runs": 0,
                "tasks": set(),
                "best": None,
            },
        )
        cur["runs"] += 1
        cur["tasks"].add(task_id)
        if isinstance(value, (int, float)) and better(value, cur["best"], lower_is_better):
            cur["best"] = value

    rows = []
    for worker, cur in stats.items():
        best = cur["best"]
        if best is None:
            best_text = "n/a"
        else:
            best_text = fmt_value(best)
        rows.append(
            {
                "worker": worker,
                "runs": cur["runs"],
                "tasks": len(cur["tasks"]),
                "best": best_text,
            }
        )

    rows.sort(key=lambda row: (-row["runs"], row["worker"]))
    return rows


def goal_rows(goals, runs):
    runs_by_goal = {}
    for run in runs:
        goal_id = run.get("goal_id", "unknown")
        runs_by_goal.setdefault(goal_id, []).append(run)

    rows = []
    for goal_id in sorted(goals):
        goal = goals[goal_id]
        lower_is_better = goal.get("lower_is_better", True)
        bar = goal.get("bar")
        best = None
        for run in runs_by_goal.get(goal_id, []):
            value = run.get("value")
            if isinstance(value, (int, float)) and better(value, best, lower_is_better):
                best = value
        beats = "yes" if best is not None and bar is not None and (
            best < bar if lower_is_better else best > bar
        ) else "no"
        rows.append(
            {
                "goal_id": goal_id,
                "title": goal.get("title", ""),
                "metric": goal.get("metric", ""),
                "bar": fmt_value(bar) if bar is not None else "n/a",
                "best": fmt_value(best) if best is not None else "n/a",
                "beats": beats,
                "status": goal.get("status", "unknown"),
            }
        )
    return rows


def render_table(headers, rows):
    if not rows:
        return ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return lines


def build_markdown(goals, runs):
    contributors = contributor_rows(runs)
    goal_table = goal_rows(goals, runs)

    lines = [
        "# token2science leaderboard",
        "",
        "## Contributors",
        "",
    ]
    lines.extend(
        render_table(
            ["worker", "runs submitted", "tasks touched", "best value they hold"],
            [
                {
                    "worker": row["worker"],
                    "runs submitted": row["runs"],
                    "tasks touched": row["tasks"],
                    "best value they hold": row["best"],
                }
                for row in contributors
            ],
        )
    )
    lines.extend(["", "## Goals", ""])
    lines.extend(
        render_table(
            ["goal_id", "title", "metric", "bar", "best value submitted", "beats bar", "status"],
            [
                {
                    "goal_id": row["goal_id"],
                    "title": row["title"],
                    "metric": row["metric"],
                    "bar": row["bar"],
                    "best value submitted": row["best"],
                    "beats bar": row["beats"],
                    "status": row["status"],
                }
                for row in goal_table
            ],
        )
    )
    lines.append("")
    return "\n".join(lines)


def main():
    goals = collect_goals()
    runs = collect_runs()
    OUTPUT.write_text(build_markdown(goals, runs), encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
