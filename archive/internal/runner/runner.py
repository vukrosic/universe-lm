#!/usr/bin/env python3
from __future__ import annotations

import argparse
import getpass
import json
import os
import platform
import shlex
import subprocess
import sys
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Spec:
    path: Path
    data: dict[str, Any]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def current_head(root: Path) -> str:
    git_dir = root / ".git"
    head_path = git_dir / "HEAD"
    if not head_path.exists():
        raise RuntimeError(f"missing git HEAD at {head_path}")
    head = head_path.read_text().strip()
    if not head.startswith("ref: "):
        return head
    ref = head[5:].strip()
    ref_path = git_dir / ref
    if ref_path.exists():
        return ref_path.read_text().strip()
    packed_refs = git_dir / "packed-refs"
    if packed_refs.exists():
        for line in packed_refs.read_text().splitlines():
            if not line or line.startswith("#") or line.startswith("^"):
                continue
            sha, packed_ref = line.split(" ", 1)
            if packed_ref.strip() == ref:
                return sha.strip()
    raise RuntimeError(f"unable to resolve ref {ref}")


def load_spec(path: Path) -> Spec:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: spec must be a mapping")
    return Spec(path=path, data=data)


def validate_spec(spec: Spec, root: Path) -> None:
    data = spec.data
    required = ["id", "title", "repo", "commit", "command", "config", "requires", "report"]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"{spec.path}: missing keys: {', '.join(missing)}")
    if data["repo"] != "https://github.com/vukrosic/universe-lm":
        raise ValueError(f"{spec.path}: unexpected repo {data['repo']}")
    if data["commit"] != current_head(root):
        raise ValueError(f"{spec.path}: commit {data['commit']} does not match current HEAD {current_head(root)}")
    if not isinstance(data["command"], str) or not data["command"].strip():
        raise ValueError(f"{spec.path}: command must be a non-empty string")
    if not isinstance(data["config"], dict):
        raise ValueError(f"{spec.path}: config must be a mapping")
    if not isinstance(data["requires"], dict):
        raise ValueError(f"{spec.path}: requires must be a mapping")
    if not isinstance(data["report"], dict):
        raise ValueError(f"{spec.path}: report must be a mapping")
    metrics = data["report"].get("metrics")
    if not isinstance(metrics, list) or not metrics:
        raise ValueError(f"{spec.path}: report.metrics must be a non-empty list")


def spec_status(path: Path, data: dict[str, Any]) -> str:
    for suffix in (".queued.yaml", ".claimed.yaml", ".done.yaml"):
        if path.name.endswith(suffix):
            return suffix[len(".") : -len(".yaml")]
    status = data.get("status")
    if isinstance(status, str) and status:
        return status
    return "unknown"


def normalize_command(command: str) -> list[str]:
    tokens = shlex.split(command)
    if tokens and tokens[0] in {"python", "python3"}:
        tokens[0] = sys.executable
    return tokens


def snapshot_artifacts(root: Path) -> dict[Path, tuple[int, int]]:
    snapshot: dict[Path, tuple[int, int]] = {}
    for base_name in ("results", "runs", "local_results"):
        base = root / base_name
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix in {".json", ".csv"}:
                try:
                    st = path.stat()
                except FileNotFoundError:
                    continue
                snapshot[path] = (st.st_mtime_ns, st.st_size)
    return snapshot


def changed_artifacts(root: Path, before: dict[Path, tuple[int, int]]) -> list[Path]:
    changed: list[Path] = []
    for path, prev in before.items():
        try:
            st = path.stat()
        except FileNotFoundError:
            continue
        now = (st.st_mtime_ns, st.st_size)
        if now != prev:
            changed.append(path)
    for base_name in ("results", "runs", "local_results"):
        base = root / base_name
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix in {".json", ".csv"} and path not in before:
                changed.append(path)
    changed.sort(key=lambda p: p.stat().st_mtime_ns if p.exists() else 0, reverse=True)
    return changed


def parse_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    if isinstance(data, dict):
        return data
    return None


def pick_metrics(data: dict[str, Any]) -> dict[str, Any]:
    if isinstance(data.get("final_metrics"), dict):
        metrics = dict(data["final_metrics"])
    elif isinstance(data.get("metrics"), dict):
        metrics = dict(data["metrics"])
    else:
        metrics = {}
        for key in ("val_loss", "train_loss", "val_accuracy", "val_perplexity", "tokens_per_sec"):
            if key in data:
                metrics[key] = data[key]
    if "tokens_per_sec" not in metrics:
        tokens_seen = data.get("tokens_seen")
        active_seconds = data.get("active_training_time_seconds")
        if isinstance(tokens_seen, (int, float)) and isinstance(active_seconds, (int, float)) and active_seconds > 0:
            metrics["tokens_per_sec"] = tokens_seen / active_seconds
    return metrics


def maybe_curve_rows(data: dict[str, Any]) -> list[dict[str, Any]] | None:
    history = data.get("history")
    if not isinstance(history, dict):
        return None
    steps = history.get("steps")
    train_losses = history.get("train_losses")
    val_losses = history.get("val_losses")
    if not (isinstance(steps, list) and isinstance(train_losses, list) and isinstance(val_losses, list)):
        return None
    if not (len(steps) == len(train_losses) == len(val_losses)):
        return None
    rows: list[dict[str, Any]] = []
    for step, train_loss, val_loss in zip(steps, train_losses, val_losses):
        rows.append({"step": step, "train_loss": train_loss, "val_loss": val_loss})
    return rows


def newest_result_candidate(paths: list[Path]) -> Path | None:
    def key(path: Path) -> tuple[int, int]:
        name_priority = 0 if path.name == "result.json" else 1 if path.name == "metrics.json" else 2
        try:
            mtime = path.stat().st_mtime_ns
        except FileNotFoundError:
            mtime = 0
        return (name_priority, -mtime)

    if not paths:
        return None
    return sorted(paths, key=key)[0]


def build_environment(spec: Spec) -> dict[str, Any]:
    python_version = platform.python_version()
    torch_version = "unknown"
    gpu_name = "unknown"
    try:
        out = subprocess.run(
            [sys.executable, "-c", "import torch; print(torch.__version__)"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        torch_version = out.stdout.strip() or "unknown"
    except Exception:
        pass
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        lines = [line.strip() for line in out.stdout.splitlines() if line.strip()]
        if lines:
            gpu_name = lines[0]
    except Exception:
        pass
    seed = spec.data.get("config", {}).get("seed")
    return {"gpu": gpu_name, "torch": torch_version, "python": python_version, "seed": seed}


def write_text(path: Path, text: str) -> None:
    path.write_text(text if text.endswith("\n") else text + "\n")


def write_curve(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write("step,train_loss,val_loss\n")
        for row in rows:
            f.write(f"{row['step']},{row['train_loss']},{row['val_loss']}\n")


def run_spec(spec: Spec, root: Path, dry_run: bool) -> dict[str, Any]:
    validate_spec(spec, root)
    status = spec_status(spec.path, spec.data)
    command = normalize_command(spec.data["command"])
    if dry_run:
        print(f"[dry-run] {spec.data['id']} ({status}) OK -> {' '.join(command)}")
        return {"spec_id": spec.data["id"], "status": "ok", "dry_run": True}

    started = utc_now()
    before = snapshot_artifacts(root)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(root) if not pythonpath else f"{root}{os.pathsep}{pythonpath}"
    tail = deque(maxlen=100)
    log_lines: list[str] = []
    proc = subprocess.Popen(
        command,
        cwd=root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        cleaned = line.rstrip("\n")
        tail.append(cleaned)
        log_lines.append(cleaned)
        print(cleaned)
    returncode = proc.wait()
    finished = utc_now()

    changed = changed_artifacts(root, before)
    candidate = newest_result_candidate([p for p in changed if p.suffix == ".json"])
    parsed: dict[str, Any] | None = parse_json(candidate) if candidate else None
    if parsed is None:
        for line in reversed(log_lines):
            try:
                maybe = json.loads(line)
            except Exception:
                continue
            if isinstance(maybe, dict):
                parsed = maybe
                break
    metrics = pick_metrics(parsed) if parsed else {}
    curve_rows = maybe_curve_rows(parsed) if parsed else None
    if curve_rows is None:
        csv_candidate = next((p for p in changed if p.name == "loss_curve.csv"), None)
        if csv_candidate:
            curve_rows = None
    run_user = getpass.getuser()
    run_stamp = finished.strftime("%Y%m%dT%H%M%SZ")
    result_dir = root / "results" / spec.data["id"] / f"{run_user}-{run_stamp}"
    result_dir.mkdir(parents=True, exist_ok=False)

    if curve_rows:
        write_curve(result_dir / "loss_curve.csv", curve_rows)
    if tail:
        write_text(result_dir / "log_tail.txt", "\n".join(tail))

    result = {
        "spec_id": spec.data["id"],
        "worker": run_user,
        "started": iso_z(started),
        "finished": iso_z(finished),
        "exit_status": "success" if returncode == 0 else "failed",
        "metrics": metrics,
        "environment": build_environment(spec),
    }
    (result_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"[result] wrote {result_dir}")
    return {"spec_id": spec.data["id"], "status": result["exit_status"], "result_dir": str(result_dir), "returncode": returncode}


def collect_specs(paths: list[str] | None, root: Path) -> list[Spec]:
    if paths:
        spec_paths = [Path(p) for p in paths]
    else:
        spec_paths = sorted((root / "queue").glob("*.yaml"))
    if not spec_paths:
        raise RuntimeError("no queue specs found")
    return [load_spec(path.resolve()) for path in spec_paths]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("specs", nargs="*")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = repo_root()
    specs = collect_specs(args.specs, root)
    failures = 0
    for spec in specs:
        try:
            outcome = run_spec(spec, root, args.dry_run)
            if outcome.get("status") not in {"ok", "success"}:
                failures += 1
        except Exception as exc:
            failures += 1
            print(f"[error] {spec.path}: {exc}", file=sys.stderr)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
