#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from registry.store import DEFAULT_DB_PATH, open_registry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local experiment registry")
    parser.add_argument(
        "--db",
        default=str(DEFAULT_DB_PATH),
        help="Path to the SQLite registry database",
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    subparsers.add_parser("init", help="Create or upgrade the registry schema")

    thread = subparsers.add_parser("thread", help="Manage research threads")
    thread_sub = thread.add_subparsers(dest="thread_command", required=True)
    thread_upsert = thread_sub.add_parser("upsert", help="Insert or update a thread")
    thread_upsert.add_argument("--name", required=True)
    thread_upsert.add_argument("--hypothesis")
    thread_upsert.add_argument("--owner")
    thread_upsert.add_argument("--status", default="active")
    thread_upsert.add_argument("--priority", type=int, default=0)
    thread_upsert.add_argument("--notes-path")
    thread_upsert.add_argument("--summary")

    queue = subparsers.add_parser("queue", help="Manage queue items")
    queue_sub = queue.add_subparsers(dest="queue_command", required=True)
    queue_upsert = queue_sub.add_parser("upsert", help="Insert or update a queue item")
    queue_upsert.add_argument("--thread-name", required=True)
    queue_upsert.add_argument("--name", required=True)
    queue_upsert.add_argument("--command", required=True)
    queue_upsert.add_argument("--status", default="planned")
    queue_upsert.add_argument("--priority", type=int, default=0)
    queue_upsert.add_argument("--depends-on")
    queue_upsert.add_argument("--gpu-class")
    queue_upsert.add_argument("--estimated-minutes", type=float)
    queue_upsert.add_argument("--created-by")
    queue_upsert.add_argument("--log-path")
    queue_upsert.add_argument("--output-dir")
    queue_upsert.add_argument("--decision")

    run = subparsers.add_parser("run", help="Manage run records")
    run_sub = run.add_subparsers(dest="run_command", required=True)
    run_start = run_sub.add_parser("start", help="Create a run row")
    run_start.add_argument("--thread-name", required=True)
    run_start.add_argument("--name", required=True)
    run_start.add_argument("--command")
    run_start.add_argument("--status", default="running")
    run_start.add_argument("--queue-item-id")
    run_start.add_argument("--commit")
    run_start.add_argument("--branch")
    run_start.add_argument("--config")
    run_start.add_argument("--seed", type=int)
    run_start.add_argument("--tokens-target", type=int)
    run_start.add_argument("--output-dir")
    run_start.add_argument("--metrics-path")
    run_start.add_argument("--checkpoint-path")
    run_start.add_argument("--git-commit")
    run_start.add_argument("--git-branch")
    run_start.add_argument("--git-dirty", action="store_true")
    run_start.add_argument("--run-id")

    run_finish = run_sub.add_parser("finish", help="Update a run row")
    run_finish.add_argument("--run-id", required=True)
    run_finish.add_argument("--status", required=True)
    run_finish.add_argument("--verdict")
    run_finish.add_argument("--tokens-seen", type=int)
    run_finish.add_argument("--actual-steps", type=int)
    run_finish.add_argument("--final-val-loss", type=float)
    run_finish.add_argument("--final-val-accuracy", type=float)
    run_finish.add_argument("--final-train-loss", type=float)
    run_finish.add_argument("--metrics-path")
    run_finish.add_argument("--checkpoint-path")
    run_finish.add_argument("--output-dir")
    run_finish.add_argument("--git-commit")
    run_finish.add_argument("--git-branch")
    run_finish.add_argument("--git-dirty", action="store_true")

    metrics = subparsers.add_parser("metrics", help="Import metrics.json data")
    metrics_sub = metrics.add_subparsers(dest="metrics_command", required=True)
    metrics_import = metrics_sub.add_parser("import", help="Load metrics.json into the registry")
    metrics_import.add_argument("--run-id", required=True)
    metrics_import.add_argument("--metrics-path", required=True, type=Path)
    metrics_import.add_argument("--checkpoint-path")
    metrics_import.add_argument("--output-dir")
    metrics_import.add_argument("--status", default="completed")
    metrics_import.add_argument("--verdict")

    comparison = subparsers.add_parser("comparison", help="Record baseline comparisons")
    comparison_sub = comparison.add_subparsers(dest="comparison_command", required=True)
    comparison_record = comparison_sub.add_parser("record", help="Insert a comparison row")
    comparison_record.add_argument("--run-id", required=True)
    comparison_record.add_argument("--baseline-name")
    comparison_record.add_argument("--baseline-run-id")
    comparison_record.add_argument("--matched-step", type=int)
    comparison_record.add_argument("--matched-tokens", type=int)
    comparison_record.add_argument("--baseline-val-loss", type=float)
    comparison_record.add_argument("--run-val-loss", type=float)
    comparison_record.add_argument("--delta-val-loss", type=float)
    comparison_record.add_argument("--verdict")

    decision = subparsers.add_parser("decision", help="Record a human decision")
    decision.add_argument("--thread-name")
    decision.add_argument("--run-id")
    decision.add_argument("--decision", required=True)
    decision.add_argument("--reason")
    decision.add_argument("--decided-by")

    idea = subparsers.add_parser("idea", help="Propose / approve / promote experiment ideas")
    idea_sub = idea.add_subparsers(dest="idea_command", required=True)

    idea_add = idea_sub.add_parser("add", help="Propose a new experiment idea (status=proposed)")
    idea_add.add_argument("--title", required=True)
    idea_add.add_argument("--summary", help="One-sentence summary shown on top")
    idea_add.add_argument("--explanation", help="Step-by-step plain-English walkthrough")
    idea_add.add_argument("--confidence", help="high|medium|low — confidence it lowers val loss")
    idea_add.add_argument("--expected-gain", help="Estimated val-loss change at 200M, e.g. '-0.02 to -0.05'")
    idea_add.add_argument("--pros", help="Pros, one per line (use \\n)")
    idea_add.add_argument("--cons", help="Cons, one per line (use \\n)")
    idea_add.add_argument("--reference-url", help="Source URL (paper/PR/blog) if any")
    idea_add.add_argument("--thread-name")
    idea_add.add_argument("--hypothesis")
    idea_add.add_argument("--lever", help="The one thing that changes")
    idea_add.add_argument("--rationale")
    idea_add.add_argument("--expected-effect")
    idea_add.add_argument("--scale-target", help="e.g. '5M screen -> 200M full if win'")
    idea_add.add_argument("--command", help="Ready-to-run command (needed to promote)")
    idea_add.add_argument("--gpu-class")
    idea_add.add_argument("--estimated-minutes", type=float)
    idea_add.add_argument("--priority", type=int, default=0)
    idea_add.add_argument("--proposed-by", default="ai")

    idea_list = idea_sub.add_parser("list", help="List ideas (optionally by status)")
    idea_list.add_argument("--status", help="proposed|approved|rejected|queued|done")

    idea_approve = idea_sub.add_parser("approve", help="Approve a proposed idea")
    idea_approve.add_argument("--id", required=True)
    idea_approve.add_argument("--reviewed-by", default="human")
    idea_approve.add_argument("--note")

    idea_reject = idea_sub.add_parser("reject", help="Reject a proposed idea")
    idea_reject.add_argument("--id", required=True)
    idea_reject.add_argument("--reviewed-by", default="human")
    idea_reject.add_argument("--note")

    idea_promote = idea_sub.add_parser("promote", help="Turn an approved idea into a queue item")
    idea_promote.add_argument("--id", required=True)
    idea_promote.add_argument("--created-by", default="registry")

    subparsers.add_parser("summary", help="Print table counts and status breakdowns")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    with open_registry(args.db) as registry:
        if args.action == "init":
            print(json.dumps({"db": str(Path(args.db)), "status": "initialized"}, indent=2))
            return 0
        if args.action == "summary":
            print(json.dumps(registry.summary(), indent=2, sort_keys=True))
            return 0
        if args.action == "thread":
            if args.thread_command == "upsert":
                registry.upsert_thread(
                    args.name,
                    hypothesis=args.hypothesis,
                    owner=args.owner,
                    status=args.status,
                    priority=args.priority,
                    notes_path=args.notes_path,
                    summary=args.summary,
                )
                print(json.dumps({"thread": args.name, "status": args.status}, indent=2))
                return 0
        if args.action == "queue":
            if args.queue_command == "upsert":
                queue_id = registry.upsert_queue_item(
                    args.thread_name,
                    args.name,
                    args.command,
                    status=args.status,
                    priority=args.priority,
                    depends_on=args.depends_on,
                    gpu_class=args.gpu_class,
                    estimated_minutes=args.estimated_minutes,
                    created_by=args.created_by,
                    log_path=args.log_path,
                    output_dir=args.output_dir,
                    decision=args.decision,
                )
                print(json.dumps({"queue_item_id": queue_id, "status": args.status}, indent=2))
                return 0
        if args.action == "run":
            if args.run_command == "start":
                run_id = registry.start_run(
                    thread_name=args.thread_name,
                    name=args.name,
                    status=args.status,
                    queue_item_id=args.queue_item_id,
                    command=args.command,
                    code_commit=args.commit,
                    branch=args.branch,
                    config=args.config,
                    seed=args.seed,
                    tokens_target=args.tokens_target,
                    output_dir=args.output_dir,
                    metrics_path=args.metrics_path,
                    checkpoint_path=args.checkpoint_path,
                    git_commit=args.git_commit,
                    git_branch=args.git_branch,
                    git_dirty=args.git_dirty,
                    run_id=args.run_id,
                )
                print(json.dumps({"run_id": run_id, "status": args.status}, indent=2))
                return 0
            if args.run_command == "finish":
                registry.finish_run(
                    args.run_id,
                    status=args.status,
                    verdict=args.verdict,
                    tokens_seen=args.tokens_seen,
                    actual_steps=args.actual_steps,
                    final_val_loss=args.final_val_loss,
                    final_val_accuracy=args.final_val_accuracy,
                    final_train_loss=args.final_train_loss,
                    metrics_path=args.metrics_path,
                    checkpoint_path=args.checkpoint_path,
                    output_dir=args.output_dir,
                    git_commit=args.git_commit,
                    git_branch=args.git_branch,
                    git_dirty=args.git_dirty,
                )
                print(json.dumps({"run_id": args.run_id, "status": args.status}, indent=2))
                return 0
        if args.action == "metrics":
            if args.metrics_command == "import":
                registry.import_metrics(
                    run_id=args.run_id,
                    metrics_path=args.metrics_path,
                    checkpoint_path=args.checkpoint_path,
                    output_dir=args.output_dir,
                    status=args.status,
                    verdict=args.verdict,
                )
                print(json.dumps({"run_id": args.run_id, "metrics_path": str(args.metrics_path)}, indent=2))
                return 0
        if args.action == "comparison":
            if args.comparison_command == "record":
                comparison_id = registry.record_comparison(
                    args.run_id,
                    baseline_name=args.baseline_name,
                    baseline_run_id=args.baseline_run_id,
                    matched_step=args.matched_step,
                    matched_tokens=args.matched_tokens,
                    baseline_val_loss=args.baseline_val_loss,
                    run_val_loss=args.run_val_loss,
                    delta_val_loss=args.delta_val_loss,
                    verdict=args.verdict,
                )
                print(json.dumps({"comparison_id": comparison_id}, indent=2))
                return 0
        if args.action == "idea":
            if args.idea_command == "add":
                idea_id = registry.add_idea(
                    args.title,
                    summary=args.summary,
                    explanation=args.explanation,
                    confidence=args.confidence,
                    expected_gain=args.expected_gain,
                    pros=args.pros,
                    cons=args.cons,
                    reference_url=args.reference_url,
                    thread_name=args.thread_name,
                    hypothesis=args.hypothesis,
                    lever=args.lever,
                    rationale=args.rationale,
                    expected_effect=args.expected_effect,
                    scale_target=args.scale_target,
                    command=args.command,
                    gpu_class=args.gpu_class,
                    estimated_minutes=args.estimated_minutes,
                    priority=args.priority,
                    proposed_by=args.proposed_by,
                )
                print(json.dumps({"idea_id": idea_id, "status": "proposed"}, indent=2))
                return 0
            if args.idea_command == "list":
                ideas = registry.list_ideas(status=args.status)
                for i in ideas:
                    print(json.dumps(
                        {k: i[k] for k in ("id", "status", "priority", "title", "thread_name", "scale_target")},
                        indent=2,
                    ))
                print(f"\n{len(ideas)} idea(s)")
                return 0
            if args.idea_command in ("approve", "reject"):
                status = registry.review_idea(
                    args.id,
                    decision=args.idea_command,
                    reviewed_by=args.reviewed_by,
                    review_note=args.note,
                )
                print(json.dumps({"idea_id": args.id, "status": status}, indent=2))
                return 0
            if args.idea_command == "promote":
                queue_id = registry.promote_idea_to_queue(args.id, created_by=args.created_by)
                print(json.dumps({"idea_id": args.id, "queue_item_id": queue_id, "status": "queued"}, indent=2))
                return 0
        if args.action == "decision":
            decision_id = registry.record_decision(
                thread_name=args.thread_name,
                run_id=args.run_id,
                decision=args.decision,
                reason=args.reason,
                decided_by=args.decided_by,
            )
            print(json.dumps({"decision_id": decision_id}, indent=2))
            return 0
    raise SystemExit("Unsupported command")


if __name__ == "__main__":
    raise SystemExit(main())
