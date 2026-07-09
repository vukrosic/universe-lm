#!/usr/bin/env python3
"""Toy deterministic experiment for a higher-is-better metric."""

import argparse
import json
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    rng = random.Random(cfg["seed"])
    xs = [rng.randint(0, 9) for _ in range(cfg["n"])]
    score = sum(xs) / len(xs)

    print(f"RESULT metric=score value={score:.10f}")


if __name__ == "__main__":
    main()
