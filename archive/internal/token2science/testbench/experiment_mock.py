#!/usr/bin/env python3
"""Tiny drop-in experiment wrapper around the mock backend."""

import json

from mock_backend import simulate


def main() -> None:
    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    result = simulate(config)
    print(f"RESULT metric={result['metric']} value={result['value']:.10f}")


if __name__ == "__main__":
    main()
