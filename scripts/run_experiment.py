"""Generic experiment runner -- the reusable research harness.

An *experiment* compares variants (each = a set of config overrides) across a
grid of LRs x seeds, at a chosen *scale*, and emits graphs + results.json:
  - final_val_vs_lr.png : final val loss vs LR, one line per variant (mean+/-std)
  - curves_grid.png      : per-LR val-loss curves, one line per variant
  - results.json         : machine-readable summary

Designed to be run by anyone with compute, with ONE command:

    python scripts/run_experiment.py --experiment lr_sweep --scale gpu135m
    python scripts/run_experiment.py --experiment gqa      --scale small
    python scripts/run_experiment.py --list                      # show options

Scales (pick with --scale; CLI flags override preset fields):
  toy     ~1.2M, 256 ctx, 40k tok    -> ~minutes on a MacBook (MPS/CPU)
  small   ~6.6M, 512 ctx, 2M tok     -> minutes on a modest GPU
  gpu5m   ~6.6M, 2048 ctx, 50M tok   -> small GPU study
  gpu135m ~134M, 2048 ctx, 300M tok  -> release-scale architecture screen (partial-token proxy)

NOTE: this runner ranks architectures by VAL LOSS at matched compute -- the
cheap, low-variance inner-loop metric. The final compute-optimal train + the
zero-shot benchmark suite (lm-eval) are the separate outer loop; see release docs.

Adding an experiment that needs MODEL changes (e.g. QK-norm): make those changes
on your experiment branch, then register a builder here that flips the flag.
Pure-config experiments (LR, GQA, width/depth, d_ff) need no model changes.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from configs.llm_config import LLMConfig
from models.llm import MinimalLLM
from training.trainer import setup_muon_optimizer
from training.device import resolve_device
from utils.helpers import set_seed

DATA_PATH = "processed_data/pretrain_mix_1000000000"
NATIVE_CHUNK = 2048

# ---- scale presets: arch + data + grid -------------------------------------
SCALES = {
    "toy": dict(
        d_model=24, n_heads=2, n_layers=2, d_ff=96, n_kv_heads=1,
        seq_len=256, train_tokens=40_000, batch_size=8,
        lrs=[0.005, 0.01, 0.02, 0.04, 0.08], seeds=[0, 1, 2],
        eval_every=1, max_val_batches=None,
    ),
    "small": dict(
        d_model=128, n_heads=2, n_layers=2, d_ff=512, n_kv_heads=1,
        seq_len=512, train_tokens=2_000_000, batch_size=16,
        lrs=[0.01, 0.02, 0.04, 0.08], seeds=[0, 1, 2],
        eval_every=20, max_val_batches=20,
    ),
    "gpu5m": dict(
        d_model=128, n_heads=2, n_layers=2, d_ff=512, n_kv_heads=1,
        seq_len=2048, train_tokens=50_000_000, batch_size=24,
        lrs=[0.02, 0.04], seeds=[0, 1, 2],
        eval_every=100, max_val_batches=20,
    ),
    "gpu135m": dict(  # release-scale arch screen (partial-token proxy)
        d_model=576, n_heads=9, n_layers=30, d_ff=2304, n_kv_heads=3,
        seq_len=2048, train_tokens=300_000_000, batch_size=24,
        lrs=[0.02, 0.04], seeds=[0, 1],
        eval_every=200, max_val_batches=20,
    ),
}

# ---- experiment registry: name -> builder(scale_preset) -> [(variant, overrides)] ----
def _divisors(n):
    return [d for d in (1, 2, 3, 4, 6, 8) if n % d == 0]

EXPERIMENTS = {
    # Just sweep LR for a single architecture (find best LR at a scale).
    "lr_sweep": lambda p: [("baseline", {})],
    # GQA ratio: how few KV heads can we use before val loss degrades?
    "gqa": lambda p: [(f"kv{p['n_heads']//g}", {"n_kv_heads": p["n_heads"] // g})
                       for g in _divisors(p["n_heads"])],
    # FFN width multiplier (relative to the preset's d_ff).
    "dff": lambda p: [(f"dff_x{m}", {"d_ff": int(p["d_ff"] * m)})
                       for m in (0.5, 1.0, 2.0)],
}


def build_config(p, overrides, muon_lr, seed) -> LLMConfig:
    base = dict(d_model=p["d_model"], n_heads=p["n_heads"], n_layers=p["n_layers"],
                d_ff=p["d_ff"], n_kv_heads=p["n_kv_heads"], max_seq_len=p["seq_len"],
                train_tokens=p["train_tokens"], vocab_size=49152,
                muon_lr=muon_lr, adamw_lr=0.006, batch_size=p["batch_size"],
                dropout=0.0, compile_model=False, use_amp=False)
    base.update(overrides)
    cfg = LLMConfig(**{k: v for k, v in base.items() if k in LLMConfig.__dataclass_fields__})
    cfg.seed = seed
    for k, v in overrides.items():  # allow flags not in base config
        if k not in LLMConfig.__dataclass_fields__:
            setattr(cfg, k, v)
    return cfg


def load_seqs(seq_len: int, train_tokens: int) -> torch.Tensor:
    from datasets import load_from_disk
    ds = load_from_disk(DATA_PATH)["train"]
    need = int(train_tokens * 1.2) + seq_len
    rows = min(len(ds), math.ceil(need / NATIVE_CHUNK))
    t = torch.tensor(ds[0:rows]["input_ids"], dtype=torch.long).reshape(-1)
    n = (t.numel() // seq_len) * seq_len
    return t[:n].view(-1, seq_len)


def lm_loss(model, x, amp_dtype):
    if amp_dtype is not None:
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            logits = model(x)
    else:
        logits = model(x)
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)).float(), x[:, 1:].reshape(-1))


@torch.no_grad()
def eval_loss(model, loader, device, amp_dtype, max_batches):
    model.eval()
    tot, n = 0.0, 0
    for i, (x,) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        tot += lm_loss(model, x.to(device), amp_dtype).item()
        n += 1
    model.train()
    return tot / max(1, n)


def train_one(seqs, p, overrides, muon_lr, seed, device, amp_dtype):
    set_seed(seed)
    cfg = build_config(p, overrides, muon_lr, seed)
    n_val = max(1, int(0.1 * seqs.size(0)))
    val, train = seqs[:n_val], seqs[n_val:]
    train_loader = DataLoader(TensorDataset(train), batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val), batch_size=cfg.batch_size)

    model = MinimalLLM(cfg).to(device)
    if amp_dtype is not None:
        model = model.to(dtype=amp_dtype)
        amp_dtype = None
    optims = setup_muon_optimizer(model, cfg)
    tps = cfg.batch_size * p["seq_len"]
    ev, mvb = p["eval_every"], p["max_val_batches"]

    steps, val_curve = [], []
    step, tokens = 0, 0
    model.train()
    while tokens < p["train_tokens"]:
        for (x,) in train_loader:
            if tokens >= p["train_tokens"]:
                break
            x = x.to(device)
            loss = lm_loss(model, x, amp_dtype)
            for o in optims:
                o.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            for o in optims:
                o.step()
            if step % ev == 0:
                steps.append(step)
                val_curve.append(eval_loss(model, val_loader, device, amp_dtype, mvb))
            step += 1
            tokens += tps
    steps.append(step)
    val_curve.append(eval_loss(model, val_loader, device, amp_dtype, mvb))
    n_params = sum(pp.numel() for pp in model.parameters())
    return {"steps": steps, "val": val_curve, "final_val": val_curve[-1], "params": n_params}


def make_plots(runs, variants, p, summary, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lrs = p["lrs"]
    cmap = plt.get_cmap("tab10")
    colors = {v: cmap(i % 10) for i, (v, _) in enumerate(variants)}
    exp = summary["experiment"]

    plt.figure(figsize=(7, 4.5))
    for v, _ in variants:
        means = [np.mean([r["final_val"] for r in runs[(v, lr)]]) for lr in lrs]
        stds = [np.std([r["final_val"] for r in runs[(v, lr)]]) for lr in lrs]
        plt.errorbar(lrs, means, yerr=stds, marker="o", capsize=4, color=colors[v], label=v)
    plt.xscale("log")
    plt.xlabel("muon LR (log)"); plt.ylabel("final val loss (mean +/- std over seeds)")
    plt.title(f"{exp} [{summary['scale']}]: final val vs LR")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(out / "final_val_vs_lr.png", dpi=130); plt.close()

    n = len(lrs)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.6), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, lr in zip(axes, lrs):
        for v, _ in variants:
            sr = runs[(v, lr)]
            steps = sr[0]["steps"]
            arr = np.array([r["val"] for r in sr])
            m, sd = arr.mean(0), arr.std(0)
            ax.plot(steps, m, color=colors[v], label=v)
            ax.fill_between(steps, m - sd, m + sd, color=colors[v], alpha=0.18)
        ax.set_title(f"lr={lr}"); ax.set_xlabel("step"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("val loss"); axes[0].legend(fontsize=8)
    fig.suptitle(f"{exp} [{summary['scale']}]: val-loss curves per LR (band=+/-std over seeds)")
    fig.tight_layout(); fig.savefig(out / "curves_grid.png", dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description="Generic experiment runner (toy on Mac, scales to GPU).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="example: python scripts/run_experiment.py --experiment gqa --scale small")
    ap.add_argument("--experiment", choices=list(EXPERIMENTS), default="lr_sweep")
    ap.add_argument("--scale", choices=list(SCALES), default="toy")
    ap.add_argument("--lrs", type=float, nargs="+")
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--tokens", type=int)
    ap.add_argument("--seq-len", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--out", type=str)
    ap.add_argument("--list", action="store_true", help="list experiments + scales and exit")
    args = ap.parse_args()

    if args.list:
        print("experiments:", ", ".join(EXPERIMENTS))
        print("scales     :", ", ".join(SCALES))
        return

    p = dict(SCALES[args.scale])
    if args.lrs: p["lrs"] = args.lrs
    if args.seeds: p["seeds"] = args.seeds
    if args.tokens: p["train_tokens"] = args.tokens
    if args.seq_len: p["seq_len"] = args.seq_len
    if args.batch_size: p["batch_size"] = args.batch_size

    variants = EXPERIMENTS[args.experiment](p)
    out = Path(args.out) if args.out else Path(f"experiments/results/{args.experiment}_{args.scale}")
    out.mkdir(parents=True, exist_ok=True)

    device = resolve_device("auto")
    amp_dtype = torch.bfloat16 if device.type == "cuda" else None
    seqs = load_seqs(p["seq_len"], p["train_tokens"])
    n_runs = len(variants) * len(p["lrs"]) * len(p["seeds"])
    print(f"experiment={args.experiment} scale={args.scale} device={device} "
          f"dtype={'bf16' if amp_dtype else 'fp32'} variants={[v for v,_ in variants]} "
          f"grid={len(variants)}x{len(p['lrs'])}x{len(p['seeds'])}={n_runs}")

    t0 = time.time()
    runs = {}
    for vname, ov in variants:
        for lr in p["lrs"]:
            sr = []
            for s in p["seeds"]:
                r = train_one(seqs, p, ov, lr, s, device, amp_dtype)
                sr.append(r)
                print(f"  {vname:12s} lr={lr:<6} seed={s} -> final_val {r['final_val']:.4f}")
            runs[(vname, lr)] = sr
    elapsed = time.time() - t0
    print(f"all {n_runs} runs in {elapsed:.1f}s")

    summary = {"experiment": args.experiment, "scale": args.scale, "seq_len": p["seq_len"],
               "train_tokens": p["train_tokens"], "lrs": p["lrs"], "seeds": p["seeds"],
               "wall_s": round(elapsed, 1), "table": {}}
    for (v, lr), sr in runs.items():
        fv = np.array([r["final_val"] for r in sr])
        summary["table"][f"{v}@{lr}"] = {"mean": float(fv.mean()), "std": float(fv.std()),
                                          "params": sr[0]["params"]}
    (out / "results.json").write_text(json.dumps(summary, indent=2))
    make_plots(runs, variants, p, summary, out)
    print(f"wrote 2 PNGs + results.json to {out}/")

    print("\n=== final val loss (mean over seeds), best LR per variant ===")
    for v, _ in variants:
        best = min(((summary["table"][f"{v}@{lr}"]["mean"], lr) for lr in p["lrs"]))
        print(f"  {v:12s} best {best[0]:.4f} @ lr={best[1]}")


if __name__ == "__main__":
    main()
