#!/usr/bin/env python3
"""run_experiment.py — ONE generic entrypoint for every config-row experiment.

The scalable experiment model: an experiment is no longer a hand-written
`_arq_<id>.py` file, it's a JSON of overrides carried in the Neon queue. This
runner merges the live champion (autoresearch/champion.json — the same record
the daemon maintains) with those overrides, builds the config dataclass exactly
as an _arq_ stub would, and hands off to train_llm. Experiments are data; the
code is this one file; a config can be hashed for instant dedup.

The champion is reproduced faithfully from champion.json's three parts:
  - config_class  -> the base dataclass (Tiny1M3MAlibiConfig)
  - flags         -> use_* booleans set True (use_deepnet_alpha, use_poly_alibi)
  - config_overrides + env -> optimizer fields + the combo slope/curvature env
An experiment's own {env, fields, seed} are layered ON TOP.

  EXPERIMENT_CONFIG='{"fields":{"use_layer_scale":true}}' python run_experiment.py
  EXPERIMENT_CONFIG='{...}' python run_experiment.py --dry   # build+print, NO train
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CHAMPION_PATH = ROOT / "autoresearch" / "champion.json"
DEFAULT_DATASET = "processed_data/pretrain_1B"
DEFAULT_CONFIG_CLASS = "configs.llm_config.Tiny1M3MAlibiConfig"


def champion_base() -> dict:
    """The champion as {config_class, env, fields, seed} — flags folded into
    fields as True. This is the floor an experiment delta stacks on.

    champion.json is gitignored daemon-LOCAL state, so it only exists on the
    operator's machine — NOT on a worker box. When it's absent we return an
    empty base: the config row carried in the Neon queue is self-contained
    (the feeder resolves champion+override into a full config), so the box needs
    no local champion state. When it IS present (operator), a row can be a thin
    delta and we fill the rest from here."""
    if not CHAMPION_PATH.exists():
        return {"config_class": DEFAULT_CONFIG_CLASS, "env": {}, "fields": {}, "seed": 42}
    champ = json.loads(CHAMPION_PATH.read_text())
    fields: dict = {f: True for f in champ.get("flags", [])}
    fields.update(champ.get("config_overrides", {}))
    return {
        "config_class": champ.get("config_class", DEFAULT_CONFIG_CLASS),
        "env": dict(champ.get("env", {})),
        "fields": fields,
        "seed": champ.get("seed", 42),
    }


def resolve(overrides: dict) -> dict:
    """Champion base + this experiment's overrides → the fully-resolved spec."""
    base = champion_base()
    return {
        "config_class": overrides.get("config_class", base["config_class"]),
        "env": {**base["env"], **overrides.get("env", {})},
        "fields": {**base["fields"], **overrides.get("fields", {})},
        "seed": overrides.get("seed", base["seed"]),
        "dataset_path": overrides.get("dataset_path", DEFAULT_DATASET),
    }


def build_config_class(config_class: str, fields: dict):
    """Dynamically subclass the champion's base dataclass with field overrides —
    the runtime equivalent of an _arq_ stub's `@dataclass class C(Base): ...`."""
    module_name, class_name = config_class.rsplit(".", 1)
    base = getattr(importlib.import_module(module_name), class_name)
    ns: dict = {"__annotations__": {}}
    for k, v in fields.items():
        ns["__annotations__"][k] = type(v)
        ns[k] = v
    return dataclass(type("C", (base,), ns))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true",
                    help="build + print the resolved config, do NOT train")
    args, _ = ap.parse_known_args()

    overrides = json.loads(os.environ.get("EXPERIMENT_CONFIG", "{}"))
    cfg = resolve(overrides)

    # Combo champion env (slope/curvature) must be set BEFORE the model builds.
    for k, v in cfg["env"].items():
        os.environ[k] = str(v)

    C = build_config_class(cfg["config_class"], cfg["fields"])

    if args.dry:
        inst = C()
        resolved = {
            "config_class": cfg["config_class"],
            "env": cfg["env"],
            "fields": {k: getattr(inst, k) for k in cfg["fields"]},
            "seed": cfg["seed"],
            "dataset_path": cfg["dataset_path"],
        }
        print("RESOLVED_CONFIG " + json.dumps(resolved, default=str, sort_keys=True))
        print("DRY_OK")
        return 0

    import train_llm
    sys.modules["__main__"].C = C
    sys.argv = [
        "train_llm.py",
        "--config_class", "__main__.C",
        "--seed", str(cfg["seed"]),
        "--dataset_path", cfg["dataset_path"],
        "--warmup", "false",
    ]
    train_llm.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
