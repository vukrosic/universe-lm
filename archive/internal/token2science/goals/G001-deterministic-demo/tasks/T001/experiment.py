#!/usr/bin/env python3
"""Toy deterministic experiment - the stand-in for a real training run.

Same seed + config => same number on every machine (stdlib Mersenne Twister),
so CI can reproduce it for free. Replace the body with a real experiment; keep
the `RESULT metric=<name> value=<float>` contract on the last line.
"""
import argparse
import json
import math
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    rng = random.Random(cfg["seed"])
    xs = [rng.gauss(0.0, 1.0) for _ in range(cfg["n"])]
    rmse = math.sqrt(sum(x * x for x in xs) / len(xs))

    print(f"RESULT metric=rmse value={rmse:.10f}")


if __name__ == "__main__":
    main()
