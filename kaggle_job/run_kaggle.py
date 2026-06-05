#!/usr/bin/env python3
"""Kaggle headless entrypoint for the universe-lm ablation sweep.

This file is pushed as a Kaggle *script* kernel (see kernel-metadata.json).
It runs top-to-bottom on a Kaggle GPU box: clone repo -> install deps ->
run scripts/run_research.py. Everything written under /kaggle/working is
retrievable later with `kaggle kernels output`.

Edit the CONFIG block, then from your machine:
    bash scripts/kaggle_push.sh        # push + run
    bash scripts/kaggle_push.sh pull   # download results when status==complete
"""
import os
import subprocess
import sys

# ----------------------------- CONFIG -------------------------------------
REPO   = "https://github.com/vukrosic/universe-lm.git"
BRANCH = "main"
# Which research folders to sweep, and seeds. Mirrors run_research.py flags.
# Query research at the Tiny tier: ~0.94M params · 3M tokens, 1 seed.
FOLDERS = ["query_tiny"]
SEEDS   = [0]
# Hard cap per single run (sec) so one wedged config can't eat the session.
PER_RUN_TIMEOUT = 1800
# --------------------------------------------------------------------------

WORK = "/kaggle/working"
SRC  = os.path.join(WORK, "universe-lm")


def sh(cmd, **kw):
    print(f"\n$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=True, **kw)


def main():
    os.chdir(WORK)
    # Fresh clone each run (Kaggle gives a clean box); shallow for speed.
    if not os.path.isdir(SRC):
        sh(f"git clone --depth 1 --branch {BRANCH} {REPO} {SRC}")
    os.chdir(SRC)
    sh("git log --oneline -1")

    # Kaggle ships torch + CUDA already; install only what's missing, quietly.
    sh("pip install -q -r requirements.txt")

    # nvidia-smi sanity (non-fatal).
    subprocess.run("nvidia-smi", shell=True)

    folders = " ".join(FOLDERS)
    seeds   = " ".join(str(s) for s in SEEDS)
    # --no-commit: no git creds on the box; results land in runs/ which is
    # under /kaggle/working (this dir is the repo) -> captured as output.
    sh(
        f"python scripts/run_research.py --folders {folders} "
        f"--seeds {seeds} --timeout {PER_RUN_TIMEOUT} --no-commit"
    )

    # Surface the result tree + any failures in the kernel log.
    sh("find runs -name metrics.json | sort || true")
    sh("cat runs/failures.jsonl 2>/dev/null || echo 'no failures logged'")

    # Mirror results to the working-dir root so `kaggle kernels output`
    # pulls a compact tree even though the repo lives in a subfolder.
    sh(f"cp -r {SRC}/runs {WORK}/runs || true")


if __name__ == "__main__":
    sys.exit(main())
