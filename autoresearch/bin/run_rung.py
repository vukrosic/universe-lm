#!/usr/bin/env python3
"""run_rung.py — run one (arm x rung) of the release ladder and log the point.

Direct `train_llm` invoke (NOT run_experiment.py) so the operator-machine tiny
champion is NOT folded in via champion.json — clean, fully-explicit ladder
configs. See autoresearch/LADDER.md.

  python3 autoresearch/bin/run_rung.py --arm baseline --rung Ladder8M155MConfig
  python3 autoresearch/bin/run_rung.py --arm deepnet   --rung Ladder13M252MConfig --seed 42
  python3 autoresearch/bin/run_rung.py --arm deepnet   --rung Ladder8M155MConfig --check  # build only, no train

Arms carry STRUCTURAL levers ONLY (EXPERIMENT-DESIGN RULE 0 — never optimizer
knobs). The `polyalibi`/`champion` arm is CUT under DECISIONS.jsonl D002: it wins
loss by punishing distant attention, which trades away the release's long-context
capability. Long-context-SAFE levers (RoPE-base / QK-norm / diff-attn — see
autoresearch/LONG-CONTEXT-IDEAS.md) get added here as they earn a scale test.

On completion, reads <output_dir>/metrics.json (history.val_losses[-1] = the
natural-end val loss) and appends one line to autoresearch/ladder/results.jsonl:
  {"arch","N","tokens","val_loss","seed","rung"}
where N is NON-EMBEDDING params for the rung (see LADDER.md table).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))          # .../autoresearch/bin
AUTORESEARCH = os.path.dirname(HERE)                        # .../autoresearch
REPO = os.path.dirname(AUTORESEARCH)                        # repo root
RESULTS = os.path.join(AUTORESEARCH, "ladder", "results.jsonl")

# rung config class -> (non-embedding N, train tokens). Verified by build; see
# the LADDER.md table. N is NON-EMBEDDING params (the 49k-vocab embedding
# dominates total at small N and does not follow the transformer's scaling).
RUNGS = {
    "Ladder8M155MConfig":   (1_450_000,   155_000_000),
    "Ladder13M252MConfig":  (3_170_000,   252_000_000),
    "Ladder23M469MConfig":  (10_900_000,  469_000_000),
    "Ladder52M1042MConfig": (33_200_000,  1_042_000_000),
    "Full135M2700MConfig":  (106_830_000, 2_700_000_000),
}

# arm -> STRUCTURAL flag overrides only (typed dataclass field defaults).
# D002: no distance-punishing attention (no use_alibi_bias / use_poly_alibi /
# kerple / hard sliding-window) — those are disqualified regardless of loss.
ARM_FLAGS = {
    "baseline": {},                              # plain target arch — the control
    "deepnet":  {"use_deepnet_alpha": True},     # DeepNet-a residual init (non-positional, long-context-safe)
    # DeepNet ablations — see autoresearch/DEEPNET-RESEARCH.md. All D002-safe (non-positional):
    "deepnet_ab": {"use_deepnet_alpha": True, "use_deepnet_beta_init": True},  # E3: + canonical beta init downscale ((8L)^-1/4)
    "rezero":     {"use_re_zero": True},                                       # E4: specificity ctrl — learned SCALAR alpha, init 0
    "layerscale": {"use_layer_scale": True},                                   # E4: specificity ctrl — canonical LayerScale, learned per-CHANNEL gamma init 1e-4 (Touvron 2021; NOT the use_layerscale (1+g) variant)
    # Long-context-safe candidates (flags already wired):
    "ropebase100k":  {"rope_base": 100_000},
    "ropebase250k":  {"rope_base": 250_000},
    "ropebase500k":  {"rope_base": 500_000},
    "qknorm":        {"use_qk_norm_post_rope": True},
    "diffattn":      {"use_diff_attn": True},
}

# Per-rung ENGINEERING (NOT architecture): memory-safe micro-batch + grad-accum
# to fit the 12GB box. The logits tensor (micro_batch x seq_len 2048 x vocab
# 49,152) dominates memory — at batch 8 that alone is ~3.2GB and OOMs the 3060 on
# the backward. So the micro-batch is small and grad-accum keeps the EFFECTIVE
# batch (8) and training dynamics constant across every rung and arm (RULE 0:
# engineering re-tuned per tier, not the architecture axis). All arms at a rung
# share these — only the structural ARM_FLAGS differ.
EFFECTIVE_BATCH = 8
RUNG_ENGINEERING = {
    "Ladder8M155MConfig":   {"batch_size": 2, "gradient_accumulation_steps": 4},
    "Ladder13M252MConfig":  {"batch_size": 2, "gradient_accumulation_steps": 4},
    "Ladder23M469MConfig":  {"batch_size": 2, "gradient_accumulation_steps": 4},
    "Ladder52M1042MConfig": {"batch_size": 1, "gradient_accumulation_steps": 8},
    "Full135M2700MConfig":  {"batch_size": 1, "gradient_accumulation_steps": 8},
}


def build_config_class(rung_name, flags):
    """Re-decorated @dataclass subclass with typed field defaults.

    A PLAIN subclass body silently no-ops on a dataclass field (the parent
    __init__ overwrites the class attr with the field default) — verified. So
    overrides MUST be real dataclass fields, which make_dataclass produces.
    """
    sys.path.insert(0, REPO)
    from configs import llm_config
    if rung_name not in vars(llm_config):
        sys.exit(f"unknown rung config {rung_name!r}; known: {', '.join(RUNGS)}")
    base = getattr(llm_config, rung_name)
    fields = [(k, type(v), dataclasses.field(default=v)) for k, v in flags.items()]
    return dataclasses.make_dataclass("C", fields, bases=(base,))


def log_result(output_dir, arch, N, tokens, seed, rung):
    """Append one ladder point to results.jsonl from the run's metrics.json."""
    mpath = os.path.join(output_dir, "metrics.json")
    if not os.path.exists(mpath):
        print(f"[run_rung] WARNING: no metrics.json at {mpath}; not logging.", file=sys.stderr)
        return None
    with open(mpath) as f:
        m = json.load(f)
    vls = (m.get("history") or {}).get("val_losses") or []
    if not vls:
        fm = m.get("final_metrics") or {}
        val = fm.get("val_loss")
    else:
        val = float(vls[-1])
    if val is None:
        print(f"[run_rung] WARNING: no val_loss in {mpath}; not logging.", file=sys.stderr)
        return None
    rec = {"arch": arch, "N": N, "tokens": tokens, "val_loss": round(float(val), 4),
           "seed": seed, "rung": rung}
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[run_rung] logged -> {RESULTS}: {json.dumps(rec)}")
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=sorted(ARM_FLAGS), help="which structural arm")
    ap.add_argument("--rung", default="Ladder8M155MConfig", choices=sorted(RUNGS), help="rung config class")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dataset_path", default="processed_data/pretrain_1B")
    ap.add_argument("--output_dir", default=None, help="default runs/ladder/<rung>_<arm>_s<seed>")
    ap.add_argument("--check", action="store_true", help="build the config + print, do NOT train")
    args = ap.parse_args()

    N, tokens = RUNGS[args.rung]
    # rung engineering (batch/accum) first, then the arm's structural levers on top
    flags = {**RUNG_ENGINEERING.get(args.rung, {}), **ARM_FLAGS[args.arm]}
    C = build_config_class(args.rung, flags)

    if args.check:
        c = C()
        print(f"arm={args.arm} rung={args.rung}  d_model={c.d_model} n_layers={c.n_layers} "
              f"n_heads={c.n_heads} n_kv_heads={c.n_kv_heads} train_tokens={c.train_tokens:,}")
        print(f"  non-embed N (table) = {N:,}   tokens (table) = {tokens:,}")
        for k, v in flags.items():
            got = getattr(c, k)
            print(f"  flag {k} = {got}" + ("" if got == v else f"  !! override DID NOT TAKE (got {got})"))
        # D002 guard: no distance-punishing attention may be on.
        for banned in ("use_alibi_bias", "use_poly_alibi"):
            if getattr(c, banned, False):
                sys.exit(f"D002 VIOLATION: {banned} is True on arm {args.arm} — distance-punishing attention is banned.")
        print("  D002 OK: no distance-punishing attention active.")
        return

    short = args.rung.replace("Config", "").replace("Ladder", "L").lower()
    output_dir = args.output_dir or os.path.join(REPO, "runs", "ladder", f"{short}_{args.arm}_s{args.seed}")
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(REPO)  # train_llm uses repo-relative paths (processed_data/, etc.)

    import train_llm
    sys.modules["__main__"].C = C
    sys.argv = [
        "train_llm.py",
        "--config_class", "__main__.C",
        "--seed", str(args.seed),
        "--dataset_path", args.dataset_path,
        "--output_dir", output_dir,
        "--warmup", "false",
    ]
    print(f"[run_rung] arm={args.arm} rung={args.rung} seed={args.seed} -> {output_dir}")
    train_llm.main()

    log_result(output_dir, args.arm, N, tokens, args.seed, args.rung)


if __name__ == "__main__":
    main()
