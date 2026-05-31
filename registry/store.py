from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


SCHEMA_PATH = Path(__file__).with_name("schema.sql")
DEFAULT_DB_PATH = Path("registry/experiments.sqlite")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id() -> str:
    return uuid.uuid4().hex


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    return int(value)


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _bool_to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    return int(bool(value))


def _load_schema() -> str:
    return SCHEMA_PATH.read_text(encoding="utf-8")


@dataclass
class MetricsPayload:
    final_metrics: Dict[str, Any]
    history: Dict[str, list]
    raw: Dict[str, Any]


def load_metrics_payload(metrics_path: Path) -> MetricsPayload:
    raw = json.loads(metrics_path.read_text(encoding="utf-8"))
    final_metrics = (
        raw.get("final_metrics")
        or raw.get("final_eval")
        or raw.get("metrics")
        or {}
    )
    history = raw.get("history") or raw.get("metrics_history") or {}
    return MetricsPayload(final_metrics=final_metrics, history=history, raw=raw)


class ExperimentRegistry:
    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "ExperimentRegistry":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def initialize(self) -> None:
        self.conn.executescript(_load_schema())
        self.conn.commit()

    def upsert_thread(
        self,
        name: str,
        *,
        hypothesis: Optional[str] = None,
        owner: Optional[str] = None,
        status: str = "active",
        priority: int = 0,
        notes_path: Optional[str] = None,
        summary: Optional[str] = None,
    ) -> None:
        now = utc_now()
        self.conn.execute(
            """
            INSERT INTO threads (
                name, hypothesis, owner, status, priority, notes_path, summary, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                hypothesis=excluded.hypothesis,
                owner=excluded.owner,
                status=excluded.status,
                priority=excluded.priority,
                notes_path=excluded.notes_path,
                summary=excluded.summary,
                updated_at=excluded.updated_at
            """,
            (name, hypothesis, owner, status, priority, notes_path, summary, now, now),
        )
        self.conn.commit()

    def upsert_queue_item(
        self,
        thread_name: str,
        name: str,
        command: str,
        *,
        status: str = "planned",
        priority: int = 0,
        depends_on: Optional[str] = None,
        gpu_class: Optional[str] = None,
        estimated_minutes: Optional[float] = None,
        created_by: Optional[str] = None,
        log_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        decision: Optional[str] = None,
    ) -> str:
        now = utc_now()
        queue_id = self.get_queue_item_id(thread_name, name) or new_id()
        self.conn.execute(
            """
            INSERT INTO queue_items (
                id, thread_name, name, command, status, priority, depends_on,
                gpu_class, estimated_minutes, created_by, created_at, started_at,
                finished_at, log_path, output_dir, decision
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?)
            ON CONFLICT(thread_name, name) DO UPDATE SET
                command=excluded.command,
                status=excluded.status,
                priority=excluded.priority,
                depends_on=excluded.depends_on,
                gpu_class=excluded.gpu_class,
                estimated_minutes=excluded.estimated_minutes,
                created_by=excluded.created_by,
                log_path=excluded.log_path,
                output_dir=excluded.output_dir,
                decision=excluded.decision
            """,
            (
                queue_id,
                thread_name,
                name,
                command,
                status,
                priority,
                depends_on,
                gpu_class,
                estimated_minutes,
                created_by,
                now,
                log_path,
                output_dir,
                decision,
            ),
        )
        self.conn.commit()
        return queue_id

    def get_queue_item_id(self, thread_name: str, name: str) -> Optional[str]:
        row = self.conn.execute(
            "SELECT id FROM queue_items WHERE thread_name = ? AND name = ?",
            (thread_name, name),
        ).fetchone()
        return row["id"] if row else None

    def update_queue_item_status(
        self,
        thread_name: str,
        name: str,
        *,
        status: str,
        started_at: Optional[str] = None,
        finished_at: Optional[str] = None,
        log_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        decision: Optional[str] = None,
    ) -> None:
        queue_id = self.get_queue_item_id(thread_name, name)
        if not queue_id:
            raise KeyError(f"queue item not found: {thread_name}/{name}")
        self.conn.execute(
            """
            UPDATE queue_items
            SET status = ?,
                started_at = COALESCE(?, started_at),
                finished_at = COALESCE(?, finished_at),
                log_path = COALESCE(?, log_path),
                output_dir = COALESCE(?, output_dir),
                decision = COALESCE(?, decision)
            WHERE id = ?
            """,
            (status, started_at, finished_at, log_path, output_dir, decision, queue_id),
        )
        self.conn.commit()

    def start_run(
        self,
        *,
        thread_name: str,
        name: str,
        status: str = "running",
        queue_item_id: Optional[str] = None,
        command: Optional[str] = None,
        code_commit: Optional[str] = None,
        branch: Optional[str] = None,
        config: Optional[str] = None,
        seed: Optional[int] = None,
        tokens_target: Optional[int] = None,
        output_dir: Optional[str] = None,
        metrics_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        git_commit: Optional[str] = None,
        git_branch: Optional[str] = None,
        git_dirty: Optional[bool] = None,
        run_id: Optional[str] = None,
    ) -> str:
        run_id = run_id or new_id()
        now = utc_now()
        self.conn.execute(
            """
            INSERT INTO runs (
                id, queue_item_id, thread_name, name, command, code_commit, branch, config, seed,
                tokens_target, tokens_seen, actual_steps, status, verdict, output_dir,
                metrics_path, checkpoint_path, final_val_loss, final_val_accuracy,
                final_train_loss, git_commit, git_branch, git_dirty, created_at, finished_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, NULL, ?, ?, ?, NULL, NULL, NULL, ?, ?, ?, ?, NULL)
            """,
            (
                run_id,
                queue_item_id,
                thread_name,
                name,
                command,
                code_commit,
                branch,
                config,
                seed,
                tokens_target,
                status,
                output_dir,
                metrics_path,
                checkpoint_path,
                git_commit,
                git_branch,
                _bool_to_int(git_dirty),
                now,
            ),
        )
        self.conn.commit()
        return run_id

    def finish_run(
        self,
        run_id: str,
        *,
        status: str,
        verdict: Optional[str] = None,
        tokens_seen: Optional[int] = None,
        actual_steps: Optional[int] = None,
        final_val_loss: Optional[float] = None,
        final_val_accuracy: Optional[float] = None,
        final_train_loss: Optional[float] = None,
        metrics_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        git_commit: Optional[str] = None,
        git_branch: Optional[str] = None,
        git_dirty: Optional[bool] = None,
    ) -> None:
        now = utc_now()
        self.conn.execute(
            """
            UPDATE runs
            SET status = ?,
                verdict = COALESCE(?, verdict),
                tokens_seen = COALESCE(?, tokens_seen),
                actual_steps = COALESCE(?, actual_steps),
                final_val_loss = COALESCE(?, final_val_loss),
                final_val_accuracy = COALESCE(?, final_val_accuracy),
                final_train_loss = COALESCE(?, final_train_loss),
                metrics_path = COALESCE(?, metrics_path),
                checkpoint_path = COALESCE(?, checkpoint_path),
                output_dir = COALESCE(?, output_dir),
                git_commit = COALESCE(?, git_commit),
                git_branch = COALESCE(?, git_branch),
                git_dirty = COALESCE(?, git_dirty),
                finished_at = ?
            WHERE id = ?
            """,
            (
                status,
                verdict,
                tokens_seen,
                actual_steps,
                final_val_loss,
                final_val_accuracy,
                final_train_loss,
                metrics_path,
                checkpoint_path,
                output_dir,
                git_commit,
                git_branch,
                _bool_to_int(git_dirty),
                now,
                run_id,
            ),
        )
        self.conn.commit()

    def record_eval_point(
        self,
        run_id: str,
        *,
        step: int,
        tokens: Optional[int] = None,
        val_loss: Optional[float] = None,
        val_accuracy: Optional[float] = None,
        val_perplexity: Optional[float] = None,
        learning_rate: Optional[float] = None,
        elapsed_seconds: Optional[float] = None,
    ) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO eval_points (
                run_id, step, tokens, val_loss, val_accuracy, val_perplexity,
                learning_rate, elapsed_seconds, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                step,
                tokens,
                val_loss,
                val_accuracy,
                val_perplexity,
                learning_rate,
                elapsed_seconds,
                utc_now(),
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def record_comparison(
        self,
        run_id: str,
        *,
        baseline_name: Optional[str] = None,
        baseline_run_id: Optional[str] = None,
        matched_step: Optional[int] = None,
        matched_tokens: Optional[int] = None,
        baseline_val_loss: Optional[float] = None,
        run_val_loss: Optional[float] = None,
        delta_val_loss: Optional[float] = None,
        verdict: Optional[str] = None,
    ) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO comparisons (
                run_id, baseline_name, baseline_run_id, matched_step, matched_tokens,
                baseline_val_loss, run_val_loss, delta_val_loss, verdict, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                baseline_name,
                baseline_run_id,
                matched_step,
                matched_tokens,
                baseline_val_loss,
                run_val_loss,
                delta_val_loss,
                verdict,
                utc_now(),
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def record_decision(
        self,
        *,
        thread_name: Optional[str] = None,
        run_id: Optional[str] = None,
        decision: str,
        reason: Optional[str] = None,
        decided_by: Optional[str] = None,
    ) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO decisions (
                thread_name, run_id, decision, reason, decided_by, decided_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (thread_name, run_id, decision, reason, decided_by, utc_now()),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    # ── Ideas: proposal → human approval → scale ────────────────────────────
    def add_idea(
        self,
        title: str,
        *,
        summary: Optional[str] = None,
        explanation: Optional[str] = None,
        confidence: Optional[str] = None,
        expected_gain: Optional[str] = None,
        pros: Optional[str] = None,
        cons: Optional[str] = None,
        outcome: Optional[str] = None,
        reference_url: Optional[str] = None,
        thread_name: Optional[str] = None,
        hypothesis: Optional[str] = None,
        lever: Optional[str] = None,
        rationale: Optional[str] = None,
        expected_effect: Optional[str] = None,
        scale_target: Optional[str] = None,
        command: Optional[str] = None,
        gpu_class: Optional[str] = None,
        estimated_minutes: Optional[float] = None,
        priority: int = 0,
        status: str = "proposed",
        proposed_by: Optional[str] = None,
        idea_id: Optional[str] = None,
    ) -> str:
        idea_id = idea_id or new_id()
        now = utc_now()
        self.conn.execute(
            """
            INSERT INTO ideas (
                id, title, summary, explanation, confidence, expected_gain, pros, cons, outcome, reference_url,
                thread_name, hypothesis, lever, rationale, expected_effect, scale_target,
                command, gpu_class, estimated_minutes, priority, status, proposed_by,
                reviewed_by, review_note, reviewed_at, queue_item_id, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, ?, ?)
            """,
            (
                idea_id, title, summary, explanation, confidence, expected_gain, pros, cons, outcome, reference_url,
                thread_name, hypothesis, lever, rationale, expected_effect, scale_target,
                command, gpu_class, estimated_minutes, priority, status, proposed_by, now, now,
            ),
        )
        self.conn.commit()
        return idea_id

    def list_ideas(self, status: Optional[str] = None) -> list:
        if status:
            rows = self.conn.execute(
                "SELECT * FROM ideas WHERE status = ? ORDER BY priority, created_at",
                (status,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM ideas ORDER BY priority, created_at"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_idea(self, idea_id: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute("SELECT * FROM ideas WHERE id = ?", (idea_id,)).fetchone()
        return dict(row) if row else None

    def review_idea(
        self,
        idea_id: str,
        *,
        decision: str,
        reviewed_by: Optional[str] = None,
        review_note: Optional[str] = None,
    ) -> str:
        """Approve or reject a proposed idea. decision in {approve, reject}."""
        status = {"approve": "approved", "reject": "rejected"}.get(decision, decision)
        now = utc_now()
        self.conn.execute(
            """
            UPDATE ideas
            SET status = ?, reviewed_by = ?, review_note = ?, reviewed_at = ?, updated_at = ?
            WHERE id = ?
            """,
            (status, reviewed_by, review_note, now, now, idea_id),
        )
        self.conn.commit()
        return status

    def promote_idea_to_queue(
        self,
        idea_id: str,
        *,
        created_by: str = "registry",
        require_approved: bool = True,
    ) -> str:
        """Materialize an approved idea into a runnable queue_item.

        Creates the parent thread if missing, copies the command, and links the
        idea to the new queue item (idea status -> queued).
        """
        idea = self.get_idea(idea_id)
        if not idea:
            raise KeyError(f"idea not found: {idea_id}")
        if require_approved and idea["status"] != "approved":
            raise ValueError(
                f"idea {idea_id} is '{idea['status']}', must be 'approved' before promoting"
            )
        if not idea["command"]:
            raise ValueError(f"idea {idea_id} has no command to run")
        thread_name = idea["thread_name"] or "inbox"
        existing = self.conn.execute(
            "SELECT name FROM threads WHERE name = ?", (thread_name,)
        ).fetchone()
        if not existing:
            self.upsert_thread(
                thread_name,
                hypothesis=idea["hypothesis"],
                status="active",
                priority=idea["priority"],
            )
        queue_id = self.upsert_queue_item(
            thread_name,
            idea["title"],
            idea["command"],
            status="planned",
            priority=idea["priority"],
            gpu_class=idea["gpu_class"],
            estimated_minutes=idea["estimated_minutes"],
            created_by=created_by,
        )
        now = utc_now()
        self.conn.execute(
            "UPDATE ideas SET status = 'queued', queue_item_id = ?, updated_at = ? WHERE id = ?",
            (queue_id, now, idea_id),
        )
        self.conn.commit()
        return queue_id

    def import_metrics(
        self,
        *,
        run_id: str,
        metrics_path: Path,
        checkpoint_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        status: str = "completed",
        verdict: Optional[str] = None,
    ) -> None:
        payload = load_metrics_payload(metrics_path)
        final = payload.final_metrics
        history = payload.history
        steps = list(history.get("steps", []))
        val_losses = list(history.get("val_losses", []))
        val_accuracies = list(history.get("val_accuracies", []))
        val_perplexities = list(history.get("val_perplexities", []))
        learning_rates = list(history.get("learning_rates", []))
        elapsed_times = list(history.get("elapsed_times", []))
        tokens_series = list(history.get("tokens_seen", [])) if history.get("tokens_seen") else []

        for index, step in enumerate(steps):
            self.record_eval_point(
                run_id,
                step=_as_int(step) or 0,
                tokens=_as_int(tokens_series[index]) if index < len(tokens_series) else None,
                val_loss=_as_float(val_losses[index]) if index < len(val_losses) else None,
                val_accuracy=_as_float(val_accuracies[index]) if index < len(val_accuracies) else None,
                val_perplexity=_as_float(val_perplexities[index]) if index < len(val_perplexities) else None,
                learning_rate=_as_float(learning_rates[index]) if index < len(learning_rates) else None,
                elapsed_seconds=_as_float(elapsed_times[index]) if index < len(elapsed_times) else None,
            )

        self.finish_run(
            run_id,
            status=status,
            verdict=verdict,
            tokens_seen=_as_int(final.get("tokens_seen") or payload.raw.get("tokens_seen")),
            actual_steps=_as_int(final.get("actual_steps") or payload.raw.get("actual_steps")),
            final_val_loss=_as_float(final.get("val_loss")),
            final_val_accuracy=_as_float(final.get("val_accuracy")),
            final_train_loss=_as_float(final.get("train_loss")),
            metrics_path=str(metrics_path),
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            git_commit=payload.raw.get("git_commit") or payload.raw.get("commit"),
            git_branch=payload.raw.get("git_branch") or payload.raw.get("branch"),
            git_dirty=_bool_to_int(payload.raw.get("git_dirty")),
        )

    def summary(self) -> Dict[str, Any]:
        def count(sql: str) -> int:
            row = self.conn.execute(sql).fetchone()
            return int(row[0]) if row else 0

        return {
            "threads": count("SELECT COUNT(*) FROM threads"),
            "ideas": count("SELECT COUNT(*) FROM ideas"),
            "ideas_by_status": {
                row["status"]: row["count"]
                for row in self.conn.execute(
                    "SELECT status, COUNT(*) AS count FROM ideas GROUP BY status ORDER BY count DESC, status"
                )
            },
            "queue_items": count("SELECT COUNT(*) FROM queue_items"),
            "runs": count("SELECT COUNT(*) FROM runs"),
            "eval_points": count("SELECT COUNT(*) FROM eval_points"),
            "comparisons": count("SELECT COUNT(*) FROM comparisons"),
            "decisions": count("SELECT COUNT(*) FROM decisions"),
            "runs_by_status": {
                row["status"]: row["count"]
                for row in self.conn.execute(
                    "SELECT status, COUNT(*) AS count FROM runs GROUP BY status ORDER BY count DESC, status"
                )
            },
            "queue_by_status": {
                row["status"]: row["count"]
                for row in self.conn.execute(
                    "SELECT status, COUNT(*) AS count FROM queue_items GROUP BY status ORDER BY count DESC, status"
                )
            },
        }


def open_registry(db_path: Path | str = DEFAULT_DB_PATH) -> ExperimentRegistry:
    registry = ExperimentRegistry(db_path)
    registry.initialize()
    return registry
