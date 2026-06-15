#!/usr/bin/env python
"""CPU build-smoke for an _arq treatment stub — run by queue-daemon.sh on the box.

Loads the _arq file as a module (so its `if __name__ == "__main__"` training
block does NOT execute), reads its top-level `C` config class (the RUN-CONTRACT
requires this name), and constructs the repo's model on CPU. This catches the
classic failure where a flag is added to the dataclass but never threaded through
the model — in seconds, before any GPU time is spent.

Also statically guards the launch entrypoint: the stub must drive the repo's real
trainer module and must NOT reference a non-existent legacy entrypoint
(`main.py`, `scripts/train.py`, bare `train.py`). That class of bug — a freeform
run command copied from idea.md — used to fail only on the GPU with
`can't open file '.../main.py'`, wasting a claim. We catch it here on CPU.

Repo-agnostic: the model constructor and trainer name come from env vars the
daemon sets from autoresearch/config.json's `drain.smoke` block, with defaults:
  SMOKE_TRAINER       module the __main__ block must call (default "train_llm")
  SMOKE_MODEL_IMPORT  import line for the model (default "from models.llm import MinimalLLM")
  SMOKE_MODEL_CTOR    callable that builds the model from a config (default "MinimalLLM")

Prints `SMOKE_OK` on success; a traceback + non-zero exit on failure.

Usage:  python _box_smoke.py _arq_157-conv-ffn.py
"""
import importlib.util
import os
import re
import sys

TRAINER = os.environ.get("SMOKE_TRAINER", "train_llm")
MODEL_IMPORT = os.environ.get("SMOKE_MODEL_IMPORT", "from models.llm import MinimalLLM")
MODEL_CTOR = os.environ.get("SMOKE_MODEL_CTOR", "MinimalLLM")

# Common fake entrypoints that do not exist. `\b` boundaries keep `train.py` from
# matching the real `train_llm.py`-style trainer module names.
_FORBIDDEN_ENTRYPOINT = re.compile(r"\bmain\.py\b|scripts/train\.py|\btrain\.py\b")


def check_entrypoint(arq_path: str):
    """Return an error string if the stub's launch entrypoint is wrong, else None."""
    with open(arq_path, encoding="utf-8") as fh:
        src = fh.read()
    bad = _FORBIDDEN_ENTRYPOINT.search(src)
    if bad:
        return (
            f"references non-existent entrypoint '{bad.group(0)}' — the only "
            f"trainer is {TRAINER}; launch via `import {TRAINER}; {TRAINER}.main()`"
        )
    if TRAINER not in src:
        return f"does not reference {TRAINER} — the __main__ block must call {TRAINER}.main()"
    return None


def main(arq_path: str) -> int:
    ep_err = check_entrypoint(arq_path)
    if ep_err is not None:
        print(f"SMOKE_FAIL: {arq_path} {ep_err}")
        return 4

    spec = importlib.util.spec_from_file_location("arqmod", arq_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # top-level only; __main__ guard does not fire

    cfg_cls = getattr(mod, "C", None)
    if cfg_cls is None:
        print(f"SMOKE_FAIL: {arq_path} has no top-level `C` config class")
        return 3

    cfg = cfg_cls()
    # Import + call the repo's model constructor (from config.json / env).
    ns: dict = {}
    exec(MODEL_IMPORT, ns)  # noqa: S102 — trusted, repo-local import line
    ns[MODEL_CTOR](cfg)  # CPU construct; raises if a flag isn't threaded through
    print("SMOKE_OK")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python _box_smoke.py <_arq_file.py>")
        sys.exit(2)
    try:
        sys.exit(main(sys.argv[1]))
    except Exception as e:  # noqa: BLE001 — surface any construction error to the daemon
        import traceback

        traceback.print_exc()
        print(f"SMOKE_FAIL: {type(e).__name__}: {e}")
        sys.exit(1)
