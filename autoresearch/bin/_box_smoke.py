#!/usr/bin/env python
"""CPU build-smoke for an _arq treatment stub — run by queue-daemon.sh on the box.

Loads the _arq file as a module (so its `if __name__ == "__main__"` training
block does NOT execute), reads its top-level `C` config class (the RUN-CONTRACT
requires this name), and constructs MinimalLLM(C()) on CPU. This catches the
classic failure where a flag is added to the dataclass but never threaded through
the model — in seconds, before any GPU time is spent.

Prints `SMOKE_OK` on success; a traceback + non-zero exit on failure.

Usage:  python _box_smoke.py _arq_157-conv-ffn.py
"""
import importlib.util
import sys


def main(arq_path: str) -> int:
    spec = importlib.util.spec_from_file_location("arqmod", arq_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # top-level only; __main__ guard does not fire

    cfg_cls = getattr(mod, "C", None)
    if cfg_cls is None:
        print(f"SMOKE_FAIL: {arq_path} has no top-level `C` config class")
        return 3

    cfg = cfg_cls()
    from models.llm import MinimalLLM

    MinimalLLM(cfg)  # CPU construct; raises if a flag isn't threaded through
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
