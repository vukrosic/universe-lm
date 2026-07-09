# ALiBi Deep-Dive — the shape of distance-decay at 1M params

> Independent research thread (track `alibi-deep-dive`). The main track runs the
> broad mechanism-stacking search; **this track does one thing: take the lab's
> flagship record — ALiBi (175) — and go deep on the positional-bias family** to
> find the best *shape of distance-decay* at `tiny1m3m`. Target: a postable study
> + (ideally) a sharper positional kernel to fold back into the champion.

## The record we're studying

`175-alibi-slopes` is the only *unmissable* win the lab has produced: ALiBi
(Press et al. 2022, arXiv:2108.12409) — a per-head linear penalty on the
query–key distance, `scores += −m_h·(i−j)` — beat the RoPE baseline by **Δ−0.15
(~9σ)**, dwarfing every other lever (the rest land at Δ 0.005–0.025). It is the
base of today's champion (`267`: alibi + DeepNet-α + poly-alibi, val **6.2209**).

ALiBi works at this tiny scale because a hard **locality prior** is exactly what
a 0.94M model trained for 92 steps can't learn on its own. That raises the
question this thread exists to answer.

## Research question

**Linear ALiBi is the simplest distance-decay. Is it the best one?** Holding the
rest of the champion architecture and the training recipe fixed, which *shape* of
monotone distance penalty minimizes val loss at `tiny1m3m`:

- **linear** `−m_h·d`  (ALiBi, current champion)
- **polynomial** `−(m_h·d + c_h·d²/L)`  (`use_poly_alibi`, already in champion)
- **log / concave** `−m_h·log(1 + r_h·d)`  (Kerple-log, `use_kerple_log`)
- **exponential** xPos-style magnitude decay (`use_xpos`)

…and *how it's parameterized*: the **slope distribution** (geometric `2^{-8k/H}`
vs learned-per-head vs uniform), **per-head vs tied**, **fixed vs learned**, and
the **head budget** spent on positional bias.

A WIN here is a positional kernel that beats linear ALiBi by more than the noise
band; a NULL is informative — it says linear distance-decay is already the right
shape at this scale and the headroom is elsewhere.

## Levers available today (no new model code)

All of these are live config flags / dedicated configs in
`configs/llm_config.py` + `models/layers.py` — each experiment is one stub on the
frozen recipe, seed 42, judged vs the champion:

| flag / config | kernel | params added |
|---|---|---|
| `use_alibi_bias` (`Tiny1M3MAlibiConfig`) | linear `−m_h·d` | 0 |
| `use_poly_alibi` (`Tiny1M3MPolyAlibiConfig`) | `−(m_h·d + c_h·d²/L)`, `c_h=0` init | +H |
| `use_kerple_log` (`Tiny1M3MKerpleLogConfig`) | `−m_h·log(1+r_h·d)` | +H |
| `use_xpos` | exponential RoPE-magnitude decay | small |

New variants to file as ideas (slope schedule, learned-vs-fixed slopes,
per-head-vs-tied, symmetric-vs-causal) are <200 LoC config/layer edits.

## First experiment batch (file as ideas, run on the box)

1. **Slope schedule** — geometric (default) vs uniform vs learned-per-head slopes for plain ALiBi.
2. **Learned slopes** — make `m_h` a trained parameter (init geometric) instead of fixed.
3. **Curvature sign** — poly-alibi with `c_h` free both signs vs clamped ≥0 (convex-only).
4. **Kerple-log** — concave log-distance kernel as a standalone challenger to linear.
5. **Kernel bake-off** — linear vs poly vs kerple-log vs xPos, same seed, head-to-head.
6. **Head budget** — how many heads need the bias? (bias on all heads vs half vs one.)

## Scope & constraints (this track)

- **Tier:** `tiny1m3m` only (0.94M params · 3M tokens), **seed 42**, one seed for the screen.
- **Baseline to beat:** the champion `267` (val **6.2209**) — every treatment is judged vs it.
- **Changes:** positional-bias mechanism only. No LR/schedule/init sweeps (those are the main track's forbidden zone too).
- **Promotion:** a 1-seed screen-WIN parks in `needs-confirm`; only the paired 3-seed `confirm_paired.py` (drift-free, band 0.018) promotes. Nothing auto-promotes.

## GPU box (this thread)

Dedicated rented box for this track's runs:

```
ssh -p 55010 root@1.208.108.242 -L 8080:localhost:8080
```

- RTX 3060 (12 GB), CUDA 13.0, torch 2.12.0+cu130, Python 3.12.13 — same image as the main box.
- Repo: `/root/universe-lm`  ·  venv: `/venv/main`  ·  data: `processed_data/pretrain_1B`.
- `TORCHDYNAMO_DISABLE=1` (sm_86). A tiny1m3m run is ~6.6 GB / 12 GB → **one run at a time**.
- Set up 2026-06-16 (cloned repo, installed deps, downloaded `vukrosic/blueberry-1B-pretrain`).

## Goal for today

Run experiments 1–6 above, build the val-loss-by-kernel comparison, and write a
post: *"The shape of distance-decay: tuning ALiBi for a 1M-parameter transformer."*
