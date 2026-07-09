#!/usr/bin/env python3
"""Backend-agnostic experiment wrapper."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from backend import run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    result = run(config)
    print(f"RESULT metric={result['metric']} value={result['value']:.10f}")


if __name__ == "__main__":
    main()
