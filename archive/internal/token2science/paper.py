#!/usr/bin/env python3
"""Generate paper drafts and manage paper projects for token2science.

Legacy usage:
  python paper.py --goal GOAL_ID [--me HANDLE]

Project usage:
  python paper.py project-new --id ... --title ... --author ... --mechanism ...
  python paper.py project-list
  python paper.py project-add-exp --id ... --goal ... --label ...
  python paper.py project-status --id ...

The legacy path scans token2science/runs/*/*/result.json, filters by goal_id,
picks the best value according to the goal's lower_is_better flag, and writes a
short paper draft to token2science/papers/<goal_id>.md.
"""

import argparse
import glob
import json
import os
import re
from datetime import date
from statistics import mean
import sys


ROOT = os.path.dirname(os.path.abspath(__file__))
GOALS_DIR = os.path.join(ROOT, "goals")
RUNS_DIR = os.path.join(ROOT, "runs")
PAPERS_DIR = os.path.join(ROOT, "papers")


def load_json(path):
    with open(path) as f:
        return json.load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def paper_project_dir(paper_id):
    return os.path.join(PAPERS_DIR, paper_id)


def paper_project_json_path(paper_id):
    return os.path.join(paper_project_dir(paper_id), "paper.json")


def load_paper_project(paper_id):
    paper_path = paper_project_json_path(paper_id)
    if not os.path.isfile(paper_path):
        raise FileNotFoundError(f"paper project not found: {paper_path}")
    return load_json(paper_path)


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def markdown_escape(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def paper_authors_line(authors):
    return ", ".join(authors) if authors else "unassigned"


def paper_experiments_count(paper):
    experiments = paper.get("experiments", [])
    return len(experiments) if isinstance(experiments, list) else 0


def paper_list_rows():
    pattern = os.path.join(PAPERS_DIR, "*", "paper.json")
    rows = []
    for paper_path in sorted(glob.glob(pattern)):
        paper = load_json(paper_path)
        rows.append(
            {
                "paper_id": str(paper.get("paper_id", "")),
                "title": str(paper.get("title", "")),
                "authors": paper_authors_line(paper.get("authors", [])),
                "status": str(paper.get("status", "")),
                "num_experiments": paper_experiments_count(paper),
            }
        )
    return rows


def render_paper_list_table(rows):
    lines = ["| id | title | authors | status | num experiments |", "| --- | --- | --- | --- | --- |"]
    for row in rows:
        lines.append(
            "| `{id}` | {title} | {authors} | {status} | {num_experiments} |".format(
                id=markdown_escape(row["paper_id"]),
                title=markdown_escape(row["title"]),
                authors=markdown_escape(row["authors"]),
                status=markdown_escape(row["status"]),
                num_experiments=row["num_experiments"],
            )
        )
    return "\n".join(lines)


def render_paper_status_summary(paper):
    summary = {
        "paper_id": paper.get("paper_id"),
        "title": paper.get("title"),
        "authors": paper.get("authors", []),
        "status": paper.get("status"),
        "mechanism": paper.get("mechanism", {}),
        "experiments": paper.get("experiments", []),
        "manuscript": paper.get("manuscript"),
        "figures_dir": paper.get("figures_dir"),
        "created": paper.get("created"),
        "num_experiments": paper_experiments_count(paper),
    }
    return json.dumps(summary, indent=2, ensure_ascii=False)


def scaffold_paper_project(paper_id, title, authors, mechanism_name):
    paper_dir = paper_project_dir(paper_id)
    if os.path.exists(paper_dir):
        raise SystemExit(f"paper project already exists: {paper_dir}")

    mechanism_dir = os.path.join(paper_dir, "mechanism")
    figures_dir = os.path.join(paper_dir, "figures")
    ensure_dir(mechanism_dir)
    ensure_dir(figures_dir)

    paper = {
        "paper_id": paper_id,
        "title": title,
        "authors": authors,
        "status": "draft",
        "mechanism": {
            "name": mechanism_name,
            "spec": "mechanism/spec.md",
            "patch": "mechanism/mechanism.patch",
        },
        "experiments": [],
        "manuscript": "manuscript.md",
        "figures_dir": "figures",
        "created": date.today().isoformat(),
    }
    write_json(paper_project_json_path(paper_id), paper)

    manuscript_path = os.path.join(paper_dir, "manuscript.md")
    with open(manuscript_path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(f"**Author:** {paper_authors_line(authors)}\n\n")
        for header in ("Abstract", "Method", "Experiments", "Analysis", "Results", "Reproducibility"):
            f.write(f"## {header}\n\n")

    spec_path = os.path.join(mechanism_dir, "spec.md")
    with open(spec_path, "w", encoding="utf-8") as f:
        f.write(f"# {mechanism_name}\n\n")
        f.write("Describe the mechanism here.\n")

    patch_path = os.path.join(mechanism_dir, "mechanism.patch")
    with open(patch_path, "w", encoding="utf-8") as f:
        f.write(
            "# Placeholder patch\n\n"
            "# This patch will be filled from the GPU run.\n"
        )

    open(os.path.join(mechanism_dir, ".gitkeep"), "a", encoding="utf-8").close()
    open(os.path.join(figures_dir, ".gitkeep"), "a", encoding="utf-8").close()
    return paper_dir


def add_project_experiment(paper_id, goal_id, label):
    paper = load_paper_project(paper_id)
    experiments = paper.get("experiments")
    if not isinstance(experiments, list):
        experiments = []
    experiments.append({"goal_id": goal_id, "label": label, "result": ""})
    paper["experiments"] = experiments
    write_json(paper_project_json_path(paper_id), paper)
    return paper_project_json_path(paper_id)


def normalize_goal_question(goal_md):
    lines = goal_md.splitlines()
    for idx, line in enumerate(lines):
        m = re.search(r"\*\*Question:\*\*\s*(.*)$", line)
        if not m:
            continue
        parts = [m.group(1).strip()]
        for tail in lines[idx + 1 :]:
            stripped = tail.strip()
            if not stripped:
                break
            if stripped.startswith("**") and stripped.endswith("**"):
                break
            parts.append(stripped)
        return " ".join(part for part in parts if part)
    stripped = [line.strip() for line in goal_md.splitlines() if line.strip()]
    return stripped[0] if stripped else ""


def load_goal(goal_id):
    goal_dir = os.path.join(GOALS_DIR, goal_id)
    goal_json_path = os.path.join(goal_dir, "goal.json")
    goal_md_path = os.path.join(goal_dir, "goal.md")
    if not os.path.isfile(goal_json_path):
        raise FileNotFoundError(f"goal not found: {goal_json_path}")
    if not os.path.isfile(goal_md_path):
        raise FileNotFoundError(f"goal prose not found: {goal_md_path}")
    goal = load_json(goal_json_path)
    with open(goal_md_path) as f:
        goal_md = f.read()
    return goal, goal_md


def scan_runs(goal_id):
    pattern = os.path.join(RUNS_DIR, "*", "*", "result.json")
    matches = []
    for res_path in sorted(glob.glob(pattern)):
        try:
            res = load_json(res_path)
        except Exception:
            continue
        if str(res.get("goal_id")) != goal_id:
            continue
        task_id = str(res.get("task_id", ""))
        worker = str(res.get("worker", ""))
        value = res.get("value")
        config_hash = str(res.get("config_hash", ""))
        command = str(res.get("command", ""))
        try:
            value = float(value)
        except Exception:
            continue
        matches.append(
            {
                "path": res_path,
                "task_id": task_id,
                "worker": worker,
                "value": value,
                "config_hash": config_hash,
                "command": command,
                "raw": res,
            }
        )
    return matches


def group_runs_by_task(runs):
    grouped = {}
    for run in runs:
        grouped.setdefault(run["task_id"], []).append(run)
    return grouped


def beats_bar(value, bar, lower_is_better):
    return value < bar if lower_is_better else value > bar


def summarize_arms(runs, bar, lower_is_better):
    summaries = []
    for task_id, task_runs in sorted(group_runs_by_task(runs).items()):
        values = [run["value"] for run in task_runs]
        mean_value = mean(values)
        summaries.append(
            {
                "task_id": task_id,
                "n_runs": len(task_runs),
                "mean_value": mean_value,
                "beats_bar": beats_bar(mean_value, bar, lower_is_better),
            }
        )
    return summaries


def render_goal_chart(goal, arm_summaries):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        charts_dir = os.path.join(PAPERS_DIR, "charts")
        ensure_dir(charts_dir)
        chart_path = os.path.join(charts_dir, f"{goal['goal_id']}.png")

        title = goal.get("title", goal["goal_id"])
        metric = goal.get("metric", "value")
        bar = goal["bar"]

        x = list(range(len(arm_summaries)))
        labels = [arm["task_id"] for arm in arm_summaries]
        values = [arm["mean_value"] for arm in arm_summaries]
        colors = ["#2ca02c" if arm["beats_bar"] else "#888888" for arm in arm_summaries]

        fig_width = max(6.0, 1.3 * max(1, len(arm_summaries)))
        fig, ax = plt.subplots(figsize=(fig_width, 4.2))
        bars = ax.bar(x, values, color=colors, width=0.7)
        ax.axhline(bar, color="black", linestyle="--", linewidth=1, label=f"bar = {bar:.12g}")
        ax.set_title(title)
        ax.set_ylabel(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(loc="best", frameon=False)
        if hasattr(ax, "bar_label"):
            ax.bar_label(bars, labels=[f"{value:.12g}" for value in values], padding=3, fontsize=9)
        else:
            for bar_rect, value in zip(bars, values):
                ax.annotate(
                    f"{value:.12g}",
                    xy=(bar_rect.get_x() + bar_rect.get_width() / 2, bar_rect.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
        fig.tight_layout()
        fig.savefig(chart_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return chart_path
    except Exception:
        return None


def choose_best(runs, lower_is_better):
    if not runs:
        return None
    key = (lambda r: (r["value"], r["worker"], r["path"])) if lower_is_better else (
        lambda r: (-r["value"], r["worker"], r["path"])
    )
    return sorted(runs, key=key)[0]


def format_authors(workers, me=None):
    unique = sorted(set(workers))
    if me and me in unique:
        unique.remove(me)
        unique.insert(0, me)
    return unique


def format_confirmation(best_run):
    confirmation_path = os.path.join(RUNS_DIR, best_run["task_id"], "confirmation.json")
    if not os.path.isfile(confirmation_path):
        return None

    try:
        confirmation = load_json(confirmation_path)
    except Exception as exc:
        return f"unreadable ({exc})"

    confirmed_value = confirmation.get("confirmed_value")
    supporting_workers = confirmation.get("supporting_workers", [])
    confirmed_hash = confirmation.get("confirmed_config_hash")
    status = "confirmed" if confirmed_value is not None else "pending"

    parts = []
    parts.append(f"status={status}")
    if confirmed_value is not None:
        parts.append(f"confirmed_value={confirmed_value}")
    if confirmed_hash:
        parts.append(f"confirmed_config_hash={confirmed_hash}")
    if supporting_workers:
        parts.append("supporting_workers=" + ", ".join(str(w) for w in supporting_workers))
    elif "supporting_workers" in confirmation:
        parts.append("supporting_workers=[]")
    return "; ".join(parts)


def render_paper(goal, goal_md, runs, best_run, authors, confirmation_text, chart_path=None):
    title = goal.get("title", goal["goal_id"])
    question = normalize_goal_question(goal_md)
    bar = goal["bar"]
    lower_is_better = bool(goal["lower_is_better"])
    best_value = best_run["value"]
    arm_summaries = summarize_arms(runs, bar, lower_is_better)
    beats_best = beats_bar(best_value, bar, lower_is_better)

    authors_line = ", ".join(authors) if authors else "unassigned"
    experiment_command = best_run["command"]
    confirmation_line = confirmation_text if confirmation_text is not None else "not available"

    abstract = (
        f"This paper asks: {question} The best observed result is "
        f"{best_value:.12g} versus the bar of {bar:.12g}, so the run "
        f"{'beats' if beats_best else 'does not beat'} the target."
    )

    method = (
        f"{question}\n\n"
        f"Experiment command: `{experiment_command}`"
    )

    results_table = ["| task | n runs | mean value | beats bar |", "| --- | --- | --- | --- |"]
    for arm in arm_summaries:
        results_table.append(
            "| `{task}` | {n} | {mean:.12g} | {beats} |".format(
                task=arm["task_id"],
                n=arm["n_runs"],
                mean=arm["mean_value"],
                beats="yes" if arm["beats_bar"] else "no",
            )
        )

    results_parts = []
    if chart_path is not None:
        results_parts.append(f"![{title}](charts/{goal['goal_id']}.png)")
        results_parts.append("")
    results_parts.append("\n".join(results_table))
    results_parts.append("")
    results_parts.append(f"Best value: `{best_value:.12g}`")
    results_parts.append("")
    results_parts.append(f"Beats bar: `{'yes' if beats_best else 'no'}`")
    results_parts.append("")
    results_parts.append(f"Confirmation: {confirmation_line}")
    results = "\n".join(results_parts)

    appendix = (
        f"- Worker: `{best_run['worker']}`\n"
        f"- Config hash: `{best_run['config_hash']}`\n"
        f"- Exact command: `{best_run['command']}`"
    )

    lines = [
        f"# {title}",
        "",
        f"**Authors:** {authors_line}",
        "",
        "## Abstract",
        "",
        abstract,
        "",
        "## Method",
        "",
        method,
        "",
        "## Results",
        "",
        results,
        "",
        "## Reproducibility Appendix",
        "",
        appendix,
        "",
    ]
    return "\n".join(lines)


def maybe_render_chart(goal, runs):
    bar = goal["bar"]
    lower_is_better = bool(goal["lower_is_better"])
    arm_summaries = summarize_arms(runs, bar, lower_is_better)
    if not arm_summaries:
        return None
    return render_goal_chart(goal, arm_summaries)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="paper.py")
    parser.add_argument("--goal", default=None, help="goal id, e.g. G001-deterministic-demo")
    parser.add_argument("--me", default=None, help="your handle, placed first in Authors if present")
    subparsers = parser.add_subparsers(dest="command")

    new_parser = subparsers.add_parser("project-new", help="create a paper project scaffold")
    new_parser.add_argument("--id", required=True, help="paper id, e.g. entropy-gated-heads")
    new_parser.add_argument("--title", required=True, help="paper title")
    new_parser.add_argument(
        "--author",
        action="append",
        required=True,
        help="author name; repeat for multiple authors",
    )
    new_parser.add_argument("--mechanism", required=True, help="mechanism name")

    subparsers.add_parser("project-list", help="list paper projects")

    add_exp_parser = subparsers.add_parser("project-add-exp", help="append an experiment entry")
    add_exp_parser.add_argument("--id", required=True, help="paper id")
    add_exp_parser.add_argument("--goal", required=True, help="goal id")
    add_exp_parser.add_argument("--label", required=True, help="experiment label")

    status_parser = subparsers.add_parser("project-status", help="print a paper project summary")
    status_parser.add_argument("--id", required=True, help="paper id")

    args = parser.parse_args(argv)

    if args.command == "project-new":
        ensure_dir(PAPERS_DIR)
        paper_dir = scaffold_paper_project(args.id, args.title, args.author, args.mechanism)
        print(os.path.join(paper_dir, "paper.json"))
        return 0

    if args.command == "project-list":
        ensure_dir(PAPERS_DIR)
        rows = paper_list_rows()
        print(render_paper_list_table(rows))
        return 0

    if args.command == "project-add-exp":
        paper_json_path = add_project_experiment(args.id, args.goal, args.label)
        print(paper_json_path)
        return 0

    if args.command == "project-status":
        paper = load_paper_project(args.id)
        print(render_paper_status_summary(paper))
        return 0

    if not args.goal:
        parser.error("either a project subcommand or --goal is required")

    goal, goal_md = load_goal(args.goal)
    runs = scan_runs(goal["goal_id"])
    if not runs:
        raise SystemExit(f"no runs found for goal {goal['goal_id']}")

    best_run = choose_best(runs, bool(goal["lower_is_better"]))
    authors = format_authors([run["worker"] for run in runs], me=args.me)
    confirmation_text = format_confirmation(best_run)
    chart_path = maybe_render_chart(goal, runs)
    paper = render_paper(goal, goal_md, runs, best_run, authors, confirmation_text, chart_path=chart_path)

    ensure_dir(PAPERS_DIR)
    out_path = os.path.join(PAPERS_DIR, f"{goal['goal_id']}.md")
    with open(out_path, "w") as f:
        f.write(paper)

    print(out_path)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
