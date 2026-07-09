#!/usr/bin/env python3
"""Scrape every tiny1m run's metrics.json into one dated, param-aware ledger.

Writes runs/tiny1m_0604_results.md — a complete, sorted record of the
2026-06-04 tiny architecture batch (winners AND losers). For each run it
rebuilds the config from the stored `flags` and counts NON-EMBEDDING params,
so val_loss is reported alongside the param cost that produced it (the whole
point: catch param-count confounds). Wall-clock minutes are logged too, so a
mechanism that wins by spending compute (DIFF = 2x attention, NSA = +global
branch) is visible. Re-runnable; safe to call after every run.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: F401,E402  (model build needs it)
from configs.llm_config import Tiny1M3MConfig  # noqa: E402
from models.llm import MinimalLLM  # noqa: E402

RUNS = ROOT / "runs"
OUT = RUNS / "tiny1m_0604_results.md"

SKIP_FLAGS = {"schedule_type", "eval_milestones", "warmup_ratio", "compile_model",
              "batch_size", "train_tokens", "max_seq_len", "seed"}

_param_cache: dict = {}


def nonemb_params(flags: dict) -> int:
    """Rebuild the tiny config from stored flags and count non-embedding params."""
    key = tuple(sorted((k, str(v)) for k, v in flags.items()))
    if key in _param_cache:
        return _param_cache[key]
    c = Tiny1M3MConfig()
    for k, v in flags.items():
        if hasattr(c, k):
            setattr(c, k, v)
    try:
        c.__post_init__()
    except Exception:
        pass
    m = MinimalLLM(c)
    ne = sum(p.numel() for p in m.parameters()) - m.token_embedding.weight.numel()
    _param_cache[key] = ne
    return ne


rows = []
for d in sorted(RUNS.glob("tiny1m_*_full")):
    mp = d / "metrics.json"
    if not mp.exists():
        continue
    try:
        m = json.loads(mp.read_text())
    except Exception:
        continue
    fm = m.get("final_metrics") or {}
    vl = fm.get("val_loss")
    if vl is None:
        continue
    flags = m.get("flags") or m.get("active_flags") or {}
    try:
        ne = nonemb_params(flags)
    except Exception:
        ne = None
    fstr = ", ".join(f"{k}={v}" for k, v in flags.items() if k not in SKIP_FLAGS) or "—"
    name = d.name.replace("tiny1m_", "").replace("_full", "")
    wall = m.get("total_time_minutes")
    rows.append((vl, name, fm.get("val_accuracy"), ne, wall, fstr))

rows.sort(key=lambda r: r[0])
best = rows[0][0] if rows else 0.0
# reference non-emb params = the plain control if present, else the smallest
ref = next((r[3] for r in rows if r[1].endswith("ctrl") or "arch_base" in r[1]), None)
if ref is None:
    ref = min((r[3] for r in rows if r[3]), default=None)

lines = ["# tiny1m architecture batch — results ledger", ""]
lines.append(f"_Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} · {len(rows)} runs · "
             f"every run recorded, winners and losers. Params = NON-embedding._")
lines.append("")
if ref:
    lines.append(f"_Param reference = {ref:,} non-emb params; Δ% is vs that._")
    lines.append("")
lines.append("| rank | val_loss | Δ best | non-emb params | Δ% | wall (min) | val_acc | run | flags |")
lines.append("|---:|---:|---:|---:|---:|---:|---:|---|---|")
for i, (vl, name, acc, ne, wall, fstr) in enumerate(rows, 1):
    acc_s = f"{acc:.4f}" if acc is not None else "—"
    ne_s = f"{ne:,}" if ne else "—"
    dpct = f"{100*(ne-ref)/ref:+.1f}%" if (ne and ref) else "—"
    wall_s = f"{wall:.1f}" if isinstance(wall, (int, float)) else "—"
    lines.append(f"| {i} | {vl:.4f} | {vl-best:+.4f} | {ne_s} | {dpct} | {wall_s} | {acc_s} | `{name}` | {fstr} |")
lines.append("")
OUT.write_text("\n".join(lines))
print(f"wrote {OUT}: {len(rows)} runs; best={best:.4f} ({rows[0][1] if rows else '-'}); ref_params={ref}")
