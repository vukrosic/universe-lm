#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


IDEAS_ROOT = Path("/Users/vukrosic/my-life/llm-research-kit-scaling/autoresearch/ideas")
SNAPSHOT_PATH = Path("/Users/vukrosic/my-life/open-superintelligence-lab-github-io/public/data/pipeline-snapshot.json")


@dataclass(frozen=True)
class Idea:
    id: str
    status: str
    round: int
    updated: str


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_frontmatter(path: Path) -> dict[str, str]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}

    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}

    data: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        match = re.match(r"^([A-Za-z0-9_-]+):\s*(.*)$", line)
        if not match:
            continue
        key = match.group(1).strip()
        value = match.group(2).strip()
        data[key] = value
    return data


def to_int(value: str | None, fallback: int = 0) -> int:
    if value is None:
        return fallback
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def load_ideas() -> list[Idea]:
    ideas: list[Idea] = []
    if not IDEAS_ROOT.exists():
        return ideas

    for path in sorted(IDEAS_ROOT.glob("*/idea.md")):
        fm = parse_frontmatter(path)
        idea_id = fm.get("id") or path.parent.name
        status = (fm.get("status") or "").strip().lower()
        if not idea_id or not status:
            continue
        ideas.append(
            Idea(
                id=idea_id,
                status=status,
                round=to_int(fm.get("round"), 0),
                updated=(fm.get("updated") or "").strip(),
            )
        )
    return ideas


def bucket_for_status(status: str) -> str | None:
    s = status.strip().lower()
    if s in {"needs-taste", "tasting", "needs-repitch", "repitching"}:
        return "taste"
    if s in {"needs-review", "reviewing", "needs-revision", "revising"}:
        return "definition"
    if s in {
        "needs-plan",
        "planning",
        "needs-recode",
        "recoding",
    }:
        return "code"
    if s == "needs-run":
        return "needs_run"
    if s == "running":
        return "running"
    if s == "done":
        return "done"
    if s == "rejected":
        return "rejected"
    return None


def sort_key(idea: Idea) -> tuple[int, str, str]:
    priority = {
        "running": 0,
        "needs-run": 1,
        "needs-recode": 2,
        "recoding": 3,
        "needs-plan": 4,
        "planning": 5,
        "needs-review": 6,
        "reviewing": 7,
        "needs-revision": 8,
        "revising": 9,
        "needs-taste": 10,
        "tasting": 11,
        "needs-repitch": 12,
        "repitching": 13,
        "done": 90,
        "rejected": 91,
    }
    updated_key = idea.updated or ""
    return (priority.get(idea.status, 50), updated_key, idea.id)


def build_snapshot() -> dict[str, Any]:
    ideas = load_ideas()
    counts = Counter(idea.status for idea in ideas)
    gate_summary = {
        "taste": 0,
        "definition": 0,
        "code": 0,
        "needs_run": 0,
        "running": 0,
        "done": 0,
        "rejected": 0,
    }

    for idea in ideas:
        bucket = bucket_for_status(idea.status)
        if bucket is not None:
            gate_summary[bucket] += 1

    in_flight = [
        {
            "id": idea.id,
            "status": idea.status,
            "round": idea.round,
            "updated": idea.updated,
        }
        for idea in sorted(
            (idea for idea in ideas if idea.status not in {"done", "rejected"}),
            key=sort_key,
        )
    ]

    return {
        "generated_at": now_utc(),
        "counts": dict(sorted(counts.items())),
        "gate_summary": gate_summary,
        "feeding_gpu": gate_summary["needs_run"] + gate_summary["running"],
        "gpu_idle_risk": (gate_summary["needs_run"] + gate_summary["running"]) < 3,
        "in_flight": in_flight,
    }


def main() -> int:
    snapshot = build_snapshot()
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_PATH.write_text(json.dumps(snapshot, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
