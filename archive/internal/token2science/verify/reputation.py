#!/usr/bin/env python3
"""Build token2science reputation from run and confirmation artifacts.

Scoring:
  - +1 for each submitted run (result.json worker field)
  - +3 for each confirmation.json where the worker appears in supporting_workers

The script scans token2science/runs/*/*/result.json and
token2science/runs/*/confirmation.json, then writes token2science/REPUTATION.md.
Stdlib only.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
OUTPUT = ROOT / "REPUTATION.md"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def collect_scores():
    submissions = Counter()
    confirmed_supported = Counter()
    workers = set()

    if RUNS_DIR.exists():
        for result_path in sorted(RUNS_DIR.glob("*/*/result.json")):
            try:
                result = load_json(result_path)
            except Exception:
                continue

            worker = str(result.get("worker", "unknown"))
            submissions[worker] += 1
            workers.add(worker)

        for confirmation_path in sorted(RUNS_DIR.glob("*/confirmation.json")):
            try:
                confirmation = load_json(confirmation_path)
            except Exception:
                continue

            for worker in confirmation.get("supporting_workers", []):
                worker = str(worker)
                confirmed_supported[worker] += 1
                workers.add(worker)

    rows = []
    for worker in workers:
        subs = int(submissions.get(worker, 0))
        supported = int(confirmed_supported.get(worker, 0))
        score = subs + 3 * supported
        rows.append(
            {
                "worker": worker,
                "submissions": subs,
                "confirmed_supported": supported,
                "score": score,
            }
        )

    rows.sort(key=lambda row: (-row["score"], row["worker"]))
    return rows


def render_table(rows):
    lines = [
        "| worker | submissions | confirmed_supported | score |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {worker} | {submissions} | {confirmed_supported} | {score} |".format(
                worker=row["worker"],
                submissions=row["submissions"],
                confirmed_supported=row["confirmed_supported"],
                score=row["score"],
            )
        )
    return "\n".join(lines) + "\n"


def main():
    rows = collect_scores()
    OUTPUT.write_text(render_table(rows), encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
