#!/usr/bin/env python3
"""K-replication confirmation for token2science runs.

This scans runs/<task>/*/result.json, groups runs by config hash and near-
equal value, counts distinct workers per group, prints a report, and writes
token2science/runs/<task>/confirmation.json.

Stdlib only. Always exits 0.
"""

import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # token2science/
DEFAULT_TOLERANCE = 1e-9
REQUIRED_KEYS = (
    "task_id",
    "goal_id",
    "worker",
    "metric",
    "value",
    "lower_is_better",
    "seed",
    "config_path",
    "config_hash",
    "command",
)


def short_hash(config_hash):
    if config_hash.startswith("sha256:") and len(config_hash) > 19:
        return config_hash[:19]
    return config_hash[:12]


def load_results(task_id):
    task_runs_dir = os.path.join(ROOT, "runs", task_id)
    pattern = os.path.join(task_runs_dir, "*", "result.json")
    results = []

    for res_path in sorted(glob.glob(pattern)):
        try:
            with open(res_path) as f:
                res = json.load(f)
        except Exception as exc:
            print(f"WARN skipping unreadable result: {res_path} ({exc})")
            continue

        missing = [key for key in REQUIRED_KEYS if key not in res]
        if missing:
            print(f"WARN skipping invalid result: {res_path} (missing {missing})")
            continue

        try:
            value = float(res["value"])
            tolerance = float(res.get("tolerance", DEFAULT_TOLERANCE))
        except Exception as exc:
            print(f"WARN skipping invalid result: {res_path} ({exc})")
            continue

        results.append(
            {
                "path": res_path,
                "task_id": res["task_id"],
                "goal_id": res["goal_id"],
                "worker": str(res["worker"]),
                "metric": str(res["metric"]),
                "value": value,
                "lower_is_better": bool(res["lower_is_better"]),
                "seed": res["seed"],
                "config_path": res["config_path"],
                "config_hash": str(res["config_hash"]),
                "command": res["command"],
                "tolerance": tolerance,
            }
        )

    return task_runs_dir, results


def cluster_results(results):
    by_hash = {}
    for res in results:
        by_hash.setdefault(res["config_hash"], []).append(res)

    groups = []
    for config_hash in sorted(by_hash):
        rows = sorted(
            by_hash[config_hash],
            key=lambda r: (r["value"], r["worker"], r["path"]),
        )
        current = None

        for res in rows:
            if current is None:
                current = {
                    "config_hash": config_hash,
                    "runs": [res],
                    "min_value": res["value"],
                    "tolerance": res["tolerance"],
                }
                continue

            threshold = min(current["tolerance"], res["tolerance"])
            if res["value"] - current["min_value"] <= threshold:
                current["runs"].append(res)
                current["tolerance"] = threshold
            else:
                groups.append(current)
                current = {
                    "config_hash": config_hash,
                    "runs": [res],
                    "min_value": res["value"],
                    "tolerance": res["tolerance"],
                }

        if current is not None:
            groups.append(current)

    summaries = []
    for group in groups:
        values = [run["value"] for run in group["runs"]]
        workers = sorted({run["worker"] for run in group["runs"]})
        agreed_value = sum(values) / len(values)
        distinct_workers = len(workers)
        summaries.append(
            {
                "config_hash": group["config_hash"],
                "short_config_hash": short_hash(group["config_hash"]),
                "agreed_value": agreed_value,
                "run_count": len(group["runs"]),
                "distinct_worker_count": distinct_workers,
                "distinct_workers": workers,
                "status": None,  # filled in by caller
                "tolerance": group["tolerance"],
            }
        )

    return summaries


def choose_confirmation(groups, lower_is_better):
    confirmed = [g for g in groups if g["distinct_worker_count"] > 0]
    if not confirmed:
        return None

    def sort_key(group):
        value_key = group["agreed_value"] if lower_is_better else -group["agreed_value"]
        return (-group["distinct_worker_count"], value_key, group["config_hash"])

    confirmed.sort(key=sort_key)
    return confirmed[0]


def write_confirmation(task_runs_dir, payload):
    os.makedirs(task_runs_dir, exist_ok=True)
    out_path = os.path.join(task_runs_dir, "confirmation.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return out_path


def main():
    ap = argparse.ArgumentParser(prog="confirm")
    ap.add_argument("--task", required=True, help="task id, e.g. T001")
    ap.add_argument("--k", type=int, default=2, help="minimum distinct workers")
    args = ap.parse_args()

    task_runs_dir, results = load_results(args.task)

    if not results:
        print(f"no valid runs found for task {args.task}")
        payload = {
            "task_id": args.task,
            "k": args.k,
            "confirmed_value": None,
            "confirmed_config_hash": None,
            "supporting_workers": [],
            "groups": [],
        }
        out_path = write_confirmation(task_runs_dir, payload)
        print(f"wrote {os.path.relpath(out_path, ROOT)}")
        return 0

    lower_is_better = results[0]["lower_is_better"]
    groups = cluster_results(results)

    for group in groups:
        group["status"] = (
            "CONFIRMED"
            if group["distinct_worker_count"] >= args.k
            else f"pending {group['distinct_worker_count']}/{args.k}"
        )

    for idx, group in enumerate(groups, start=1):
        print(
            "group {idx}: value={value:.12g} config={cfg} workers={workers} {status}".format(
                idx=idx,
                value=group["agreed_value"],
                cfg=group["short_config_hash"],
                workers=group["distinct_worker_count"],
                status=group["status"],
            )
        )

    confirmed = [g for g in groups if g["distinct_worker_count"] >= args.k]
    winner = choose_confirmation(confirmed, lower_is_better)
    if winner is None:
        confirmed_value = None
        confirmed_config_hash = None
        supporting_workers = []
    else:
        confirmed_value = winner["agreed_value"]
        confirmed_config_hash = winner["config_hash"]
        supporting_workers = winner["distinct_workers"]

    payload = {
        "task_id": args.task,
        "k": args.k,
        "confirmed_value": confirmed_value,
        "confirmed_config_hash": confirmed_config_hash,
        "supporting_workers": supporting_workers,
        "groups": groups,
    }
    out_path = write_confirmation(task_runs_dir, payload)
    print(f"wrote {os.path.relpath(out_path, ROOT)}")
    return 0


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:  # pragma: no cover - keep the CLI non-fatal
        print(f"ERROR {exc}")
    sys.exit(0)
