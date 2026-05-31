import json
import tempfile
import unittest
from pathlib import Path

from registry.store import ExperimentRegistry


class ExperimentRegistryTest(unittest.TestCase):
    def test_init_and_basic_ingest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "experiments.sqlite"
            metrics_path = Path(tmpdir) / "metrics.json"
            metrics_payload = {
                "final_metrics": {
                    "val_loss": 4.5,
                    "val_accuracy": 0.25,
                    "train_loss": 4.2,
                    "tokens_seen": 4096,
                    "actual_steps": 2,
                },
                "history": {
                    "steps": [1, 2],
                    "val_losses": [4.8, 4.5],
                    "val_accuracies": [0.2, 0.25],
                    "val_perplexities": [10.0, 9.5],
                    "elapsed_times": [1.0, 2.0],
                    "learning_rates": [0.1, 0.05],
                },
                "git_commit": "abc123",
                "git_branch": "main",
                "git_dirty": False,
            }
            metrics_path.write_text(json.dumps(metrics_payload), encoding="utf-8")

            with ExperimentRegistry(db_path) as registry:
                registry.initialize()
                registry.upsert_thread("layerscale", hypothesis="test", status="active")
                queue_id = registry.upsert_queue_item(
                    "layerscale",
                    "screen5m",
                    "python train_llm.py --config screen5m",
                    status="planned",
                )
                run_id = registry.start_run(
                    thread_name="layerscale",
                    name="screen5m",
                    command="python train_llm.py --config screen5m",
                    queue_item_id=queue_id,
                )
                registry.import_metrics(run_id=run_id, metrics_path=metrics_path)
                registry.record_comparison(
                    run_id,
                    baseline_name="screen5m_baseline",
                    matched_step=2,
                    matched_tokens=4096,
                    baseline_val_loss=4.6,
                    run_val_loss=4.5,
                    delta_val_loss=-0.1,
                    verdict="scale",
                )
                registry.record_decision(
                    thread_name="layerscale",
                    run_id=run_id,
                    decision="scale",
                    reason="beats baseline",
                    decided_by="test",
                )

                summary = registry.summary()
                self.assertEqual(summary["threads"], 1)
                self.assertEqual(summary["queue_items"], 1)
                self.assertEqual(summary["runs"], 1)
                self.assertEqual(summary["eval_points"], 2)
                self.assertEqual(summary["comparisons"], 1)
                self.assertEqual(summary["decisions"], 1)
                self.assertEqual(summary["runs_by_status"]["completed"], 1)
                self.assertEqual(summary["queue_by_status"]["planned"], 1)

                row = registry.conn.execute(
                    "SELECT final_val_loss, final_val_accuracy, final_train_loss, git_commit, git_branch, git_dirty "
                    "FROM runs WHERE id = ?",
                    (run_id,),
                ).fetchone()
                self.assertAlmostEqual(row["final_val_loss"], 4.5)
                self.assertAlmostEqual(row["final_val_accuracy"], 0.25)
                self.assertAlmostEqual(row["final_train_loss"], 4.2)
                self.assertEqual(row["git_commit"], "abc123")
                self.assertEqual(row["git_branch"], "main")
                self.assertEqual(row["git_dirty"], 0)


if __name__ == "__main__":
    unittest.main()

