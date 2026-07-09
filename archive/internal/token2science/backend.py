from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
TESTBENCH = ROOT / "testbench"


def run(config: dict[str, Any]) -> dict[str, Any]:
    backend = os.environ.get("T2S_BACKEND", "mock")
    if backend == "mock":
        testbench_path = str(TESTBENCH)
        if testbench_path not in sys.path:
            sys.path.insert(0, testbench_path)
        from mock_backend import simulate

        return simulate(config)
    if backend == "gpu":
        raise NotImplementedError("real GPU backend not wired yet")
    raise ValueError(f"unknown T2S_BACKEND: {backend}")
