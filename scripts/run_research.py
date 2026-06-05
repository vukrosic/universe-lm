#!/usr/bin/env python3
"""One-command research-folder runner (Colab-friendly, failure-tolerant).

Runs every ablation config in the chosen research folders across N seeds,
each as a separate `train_llm.py` invocation, and writes one `metrics.json`
per run under `runs/<folder>/<Config>/seed<N>/`. Optionally git-commits
after each folder so a long Colab session checkpoints incrementally.

Colab-robustness contract:
  * Each run is wrapped in try/except — a crash in one config DOES NOT
    kill the loop. Failures are appended to `runs/failures.jsonl` with
    the config, seed, returncode/exception, and a UTC timestamp.
  * `--resume` (default ON) skips any run whose `metrics.json` already
    exists on disk, so a Colab disconnect / timeout can be resumed
    cheaply by re-running the same command.
  * `--timeout SECS` (default none) kills any single run that exceeds
    the cap; its failure is logged and the loop continues.
  * `--max-failures N` (default unlimited) bails out the whole loop
    once N consecutive failures hit — surfaces a real "all broken"
    condition instead of silently grinding.

Usage (Colab):
    python scripts/run_research.py --folders all                  # resume by default
    python scripts/run_research.py --folders all --no-resume      # force re-run
    python scripts/run_research.py --folders query output_head
    python scripts/run_research.py --folders muon --no-commit
    python scripts/run_research.py --folders all --seeds 0 1 2    # 3 seeds
    python scripts/run_research.py --folders all --timeout 1800   # 30 min/run cap
    python scripts/run_research.py --folders all --dry-run        # print plan

Each folder is paired with its scale control automatically
(Tiny1M3MConfig for tiny-scale folders, Screen10M20MConfig for query)
so every lever has a same-seed baseline to diff against.

Run lists are auto-discovered from `configs/*_ablations.py` (every
dataclass ending in `Config` = one run). The query folder's 29 configs
live in `configs/llm_config.py`; they are read from the folder's
`experiments.md` so the runner stays in sync with the tutorial.
"""
import argparse
import importlib
import inspect
import json
import re
import subprocess
import sys
import time
import traceback
from dataclasses import is_dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))  # allow `import configs.*` when run as a script

# folder -> module holding its ablation config classes (auto-discovered).
ABLATION_MODULES = {
    "output_head": "configs.output_head_ablations",
    "muon": "configs.muon_ablations",
    "rmsnorm": "configs.rmsnorm_ablations",
    "residual_stream": "configs.residual_stream_ablations",
    "attention_output": "configs.attention_output_ablations",
    "attention_logits": "configs.attention_logits_ablations",
    "data_packing": "configs.data_packing_ablations",
    "optimizer_routing": "configs.optimizer_routing_ablations",
}

# query lives in configs.llm_config; its canonical 29 are listed in experiments.md.
QUERY_EXPERIMENTS_MD = REPO / "docs/research/query/tutorial/experiments.md"
QUERY_MODULE = "configs.llm_config"

# scale control per folder, run alongside the levers for a same-seed baseline.
CONTROL_FOR = {
    "query": "configs.llm_config.Screen10M20MConfig",
    "_tiny": "configs.llm_config.Tiny1M3MConfig",
}

FAILURE_LOG = REPO / "runs" / "failures.jsonl"


def discover_ablation_configs(module_path):
    """Every dataclass named *Config *defined in* this module = one run."""
    mod = importlib.import_module(module_path)
    out = []
    for name, obj in inspect.getmembers(mod, inspect.isclass):
        if name.endswith("Config") and is_dataclass(obj) and obj.__module__ == module_path:
            out.append(f"{module_path}.{name}")
    return sorted(out)


def discover_query_configs():
    """Read the 29 query configs from experiments.md (drop the control)."""
    if not QUERY_EXPERIMENTS_MD.exists():
        return []
    names = set(re.findall(r"[A-Za-z0-9_]+Config", QUERY_EXPERIMENTS_MD.read_text()))
    names.discard("Screen10M20MConfig")  # that's the control, added separately
    return [f"{QUERY_MODULE}.{n}" for n in sorted(names)]


def build_plan(folders):
    """folder -> ordered list of fully-qualified config classes (control first)."""
    plan = {}
    for f in folders:
        if f == "query":
            configs = discover_query_configs()
            control = CONTROL_FOR["query"]
        elif f in ABLATION_MODULES:
            try:
                configs = discover_ablation_configs(ABLATION_MODULES[f])
            except ModuleNotFoundError as e:
                print(f"  ! skipping '{f}': module not found ({e})")
                continue
            control = CONTROL_FOR["_tiny"]
        else:
            print(f"  ! skipping '{f}': no runnable configs (plan-only folder)")
            continue
        if not configs:
            print(f"  ! skipping '{f}': discovered 0 configs")
            continue
        plan[f] = [control] + configs
    return plan


def short(fqcn):
    return fqcn.rsplit(".", 1)[1]


def has_metrics(out_dir: Path) -> bool:
    """A run is 'done' if its metrics.json exists and is non-empty."""
    p = out_dir / "metrics.json"
    return p.exists() and p.stat().st_size > 0


def log_failure(folder, fqcn, seed, rc, err_msg):
    """Append one JSON line to runs/failures.jsonl (best-effort)."""
    try:
        FAILURE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with FAILURE_LOG.open("a") as f:
            f.write(json.dumps({
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "folder": folder,
                "config": short(fqcn),
                "fqcn": fqcn,
                "seed": seed,
                "returncode": rc,
                "error": err_msg,
            }) + "\n")
    except Exception as e:
        print(f"   (could not write failure log: {e})")


def run_one(fqcn, out_dir, seed, dry_run, timeout):
    """One train_llm.py invocation. Returns True on success."""
    cmd = [
        sys.executable, "train_llm.py",
        "--config_class", fqcn,
        "--output_dir", str(out_dir),
        "--seed", str(seed),
    ]
    print("   $", " ".join(cmd))
    if dry_run:
        return True
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        res = subprocess.run(cmd, cwd=REPO, timeout=timeout)
        if res.returncode != 0:
            log_failure("(unknown)", fqcn, seed, res.returncode, f"exit {res.returncode}")
            return False
        return True
    except subprocess.TimeoutExpired:
        log_failure("(unknown)", fqcn, seed, -1, f"timeout after {timeout}s")
        print(f"   ! TIMEOUT after {timeout}s — moving on")
        return False
    except Exception as e:
        log_failure("(unknown)", fqcn, seed, -1, f"exception: {e!r}\n{traceback.format_exc()}")
        print(f"   ! EXCEPTION: {e!r}")
        return False


def git_commit(folder, dry_run):
    msg = f"results: {folder} ablation runs (metrics.json)"
    print(f"   git commit -> runs/{folder}/")
    if dry_run:
        return
    subprocess.run(["git", "add", f"runs/{folder}"], cwd=REPO)
    # --allow-empty-message guard: skip commit if nothing changed
    staged = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO)
    if staged.returncode == 0:
        print("   (nothing new to commit)")
        return
    subprocess.run(["git", "commit", "-m", msg], cwd=REPO)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folders", nargs="+", default=["all"],
                    help="folders to run, or 'all'. Choices: query output_head muon "
                         "rmsnorm residual_stream attention_output attention_logits "
                         "data_packing optimizer_routing")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0],
                    help="seeds per config (default: 0 — single seed; pass e.g. 0 1 2 for multi-seed)")
    ap.add_argument("--no-commit", action="store_true", help="do not git-commit after each folder")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    ap.add_argument("--no-resume", action="store_true",
                    help="re-run even when metrics.json exists (default: skip = --resume)")
    ap.add_argument("--timeout", type=int, default=None,
                    help="per-run wall-clock cap in seconds (e.g. 1800 for 30 min). Default: no cap.")
    ap.add_argument("--max-failures", type=int, default=None,
                    help="bail out after N consecutive failures (default: never bail)")
    args = ap.parse_args()

    resume = not args.no_resume
    all_folders = list(ABLATION_MODULES.keys()) + ["query"]
    folders = all_folders if args.folders == ["all"] else args.folders

    print(f"Folders: {folders}   seeds: {args.seeds}   resume: {resume}   "
          f"timeout: {args.timeout or 'none'}s")
    plan = build_plan(folders)
    if not plan:
        print("Nothing to run.")
        return

    total = sum(len(c) for c in plan.values()) * len(args.seeds)
    print(f"\nPlanned runs: {total} ({sum(len(c) for c in plan.values())} configs x {len(args.seeds)} seeds)")

    if resume and not args.dry_run:
        for folder, configs in plan.items():
            for fqcn in configs:
                for seed in args.seeds:
                    out = REPO / "runs" / folder / short(fqcn) / f"seed{seed}"
                    if has_metrics(out):
                        print(f"   ✓ already done: {folder}/{short(fqcn)}/seed{seed}")

    consecutive_failures = 0
    failures = []
    for folder, configs in plan.items():
        print(f"\n=== {folder} ({len(configs)} configs incl. control) ===")
        for fqcn in configs:
            for seed in args.seeds:
                out_dir = REPO / "runs" / folder / short(fqcn) / f"seed{seed}"
                if resume and not args.dry_run and has_metrics(out_dir):
                    print(f"   ↷ skip (metrics.json exists): {folder}/{short(fqcn)}/seed{seed}")
                    continue
                ok = run_one(fqcn, out_dir, seed, args.dry_run, args.timeout)
                if not ok:
                    failures.append(f"{folder}/{short(fqcn)}/seed{seed}")
                    consecutive_failures += 1
                    if args.max_failures and consecutive_failures >= args.max_failures:
                        print(f"\n!! {consecutive_failures} consecutive failures — bailing out "
                              f"(set --max-failures higher or 0 for unlimited)")
                        break
                else:
                    consecutive_failures = 0
            if args.max_failures and consecutive_failures >= args.max_failures:
                break
        if not args.no_commit and not args.dry_run:
            git_commit(folder, args.dry_run)
        if args.max_failures and consecutive_failures >= args.max_failures:
            break

    print("\n" + "=" * 60)
    if failures:
        print(f"DONE with {len(failures)} FAILED runs (see runs/failures.jsonl):")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("DONE — all runs completed (or were already done on disk).")


if __name__ == "__main__":
    main()
