#!/usr/bin/env python3
"""Generate a public token2science task board."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
GOALS_DIR = ROOT / "goals"
RUNS_DIR = ROOT / "runs"
BOARD_PATH = ROOT / "BOARD.md"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def first_heading(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith("#"):
                return re.sub(r"^#+\s*", "", line).strip()
    return path.parent.name


def format_bar(value: object) -> str:
    if isinstance(value, bool) or value is None:
        return str(value)
    if isinstance(value, (int, float)):
        return f"{value:g}"
    return str(value)


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def iter_goal_dirs() -> list[Path]:
    if not GOALS_DIR.exists():
        return []
    return sorted([p for p in GOALS_DIR.iterdir() if p.is_dir() and (p / "goal.json").is_file()], key=lambda p: p.name)


def iter_task_dirs(goal_dir: Path) -> list[Path]:
    tasks_dir = goal_dir / "tasks"
    if not tasks_dir.exists():
        return []
    return sorted([p for p in tasks_dir.iterdir() if p.is_dir() and (p / "task.md").is_file()], key=lambda p: p.name)


def count_runs(task_id: str) -> int:
    task_runs_dir = RUNS_DIR / task_id
    if not task_runs_dir.exists():
        return 0
    return sum(1 for path in task_runs_dir.glob("*/result.json") if path.is_file())


def build_board() -> str:
    goal_rows: list[list[str]] = []
    task_rows: list[list[str]] = []

    for goal_dir in iter_goal_dirs():
        goal = load_json(goal_dir / "goal.json")
        if goal.get("status") == "open":
            goal_rows.append([
                str(goal.get("goal_id", goal_dir.name)),
                str(goal.get("title", "")),
                str(goal.get("metric", "")),
                format_bar(goal.get("bar", "")),
                str(goal.get("status", "")),
            ])

        for task_dir in iter_task_dirs(goal_dir):
            task_id = task_dir.name
            task_rows.append([
                task_id,
                str(goal.get("goal_id", goal_dir.name)),
                first_heading(task_dir / "task.md"),
                str(count_runs(task_id)),
                "open" if count_runs(task_id) == 0 else "has-runs",
            ])

    sections: list[str] = ["# Token2science Board", ""]

    sections.append("## Open Goals")
    sections.append("")
    if goal_rows:
        sections.append(md_table(["goal_id", "title", "metric", "bar", "status"], goal_rows))
    else:
        sections.append("No open goals.")

    sections.append("")
    sections.append("## Tasks")
    sections.append("")
    if task_rows:
        sections.append(md_table(["task_id", "goal_id", "title", "runs submitted", "state"], task_rows))
    else:
        sections.append("No tasks found.")

    sections.append("")
    return "\n".join(sections)


def main() -> int:
    BOARD_PATH.write_text(build_board(), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
