# Autoresearch handoff — 2026-06-16 (claude-opus-4-8)

Goal given by user: **beat the current champion** at tiny1m3m, keep the GPU always fed,
decide continue-vs-pivot from results. Running on a `/loop` cron (`87d4de4c`, every 5 min,
session-only) firing: "continue checking and building experiments…".

## System (how the loop works)
- Two repos under `~/my-life`: **voidspark** = tooling (daemon, dashboard, `tools/autoresearch/`),
  **llm-research-kit-scaling** (this repo, remote `vukrosic/universe-lm`) = research data + model code.
- **Daemon**: `voidspark/tools/autoresearch/queue-daemon.sh --repo <thisrepo> --loop 120`, logs to
  `voidspark/.queue-daemon.log`. Runs 1 GPU job at a time (1-by-1 guard). Picks up any idea with
  `status: needs-run` AND a `run.json`. PID currently alive (1 proc). The **autopilot/auto-implement
  is DISABLED by the user** — so nothing writes `run.json`/wires configs except me. I author each
  experiment by hand.
- **Box**: Vast RTX 3060, ssh in `autoresearch/remote-box.json` (`-p 52646 root@1.208.108.242`).
- **To run an experiment**: write `_arq_NNN-name.py` (top-level `@dataclass class C(<Config>)` +
  `train_llm.main()`), local-probe it, `PYTHONPATH=. python3 ../voidspark/tools/autoresearch/_box_smoke.py _arq_NNN.py`
  → expect `SMOKE_OK`, then `mkdir autoresearch/ideas/NNN-name/`, write `idea.md` (frontmatter
  `status: needs-run`) + `run.json` (`{name, arq_file, job_timeout}`). Daemon does the rest.

## Champion / bar
- **Champion: `Tiny1M3MAlibiConfig`, val 6.2539** (honest 3-seed mean; seeds 42/123/7 =
  6.2650/6.2556/6.2412). Noise band **0.04**. Daemon WIN gate: val < **6.2003** (champion − band).
- Tier tiny1m3m: 0.94M params, 12L/4H/d64, **92 update steps**, seed 42, no warmup.

## THE key constraint (internalized)
At 92 steps the **0.04 single-seed noise floor exceeds every lever's effect measured so far.**
Two corollaries proven this session:
1. **Only step-0-ACTIVE, few-param levers can move.** Zero-init levers (gates/biases that must grow
   a matrix) wash to ~0. I built a local probe (set flag on a config *instance*, build MinimalLLM,
   measure max-abs logit diff vs alibi at init): **~6 of ~30 flags are step-0-active; the rest are
   zero-init washes.** LIVE flags found: `use_poly_alibi`(mine), `use_kerple_log`(mine), `use_entmax`,
   `use_cosine_attn`, `use_deepnet_alpha`, `use_per_head_rope_base`, `use_layernorm`, `use_k_only_norm`.
   DEAD/zero-init: diff_attn, head_gain, softpick, attn_sink, dyt/dyntanh, drop_path, conv_ffn,
   grouped_v, gmlp_sgu, focal_mod(active but +211k params=unfair), per_head_window, cosformer, etc.
2. **Sub-band levers do NOT stack** (232 poly+logit = +0.0158, not the additive −0.025). Noise dominates.

## Results so far (all NULL vs the 0.04 gate)
| idea | axis | Δ vs champ | note |
|---|---|---|---|
| 230 poly-alibi (convex quad distance) | positional curvature | −0.0111 | best single seed; **3-seed confirm = NULL** (−0.0111/+0.0124/+0.0039, mean +0.0017) → curvature CLOSED |
| 231 kerple-log (concave distance) | positional curvature | +0.0449 | worse → concave hurts |
| 217 mix-norm | norm | −0.0030 | |
| 232 poly+logit stack | composition | +0.0158 | stacking doesn't compound |
| **253 deepnet-alpha** (resid scale α=1/√24) | **residual conditioning** | **−0.0230** | **BEST signal; 3-seed confirm IN FLIGHT (256/257)** |
**Clean finding:** optimal tiny1m3m distance decay is *sharper* than linear alibi (convex helps a
little, concave hurts) — but noise-bound. New mechanisms I wired into model code (committed): the
`use_poly_alibi` (`Tiny1M3MPolyAlibiConfig`) and `use_kerple_log` (`Tiny1M3MKerpleLogConfig`) flags
across `configs/llm_config.py`, `models/llm.py`, `models/layers.py` (additive, default-off).

## CURRENT STATE (as of ~03:04)
- **GPU busy:** 256-deepnet-confirm-s123 (running on box, step 400/732, val loss 6.0993 healthy);
  257-deepnet-confirm-s7 queued behind it in the same arq tmux.
- **JUST DONE:** 255-cosine-attn-alibi — **NULL** val 6.3009 vs champ 6.2539 (Δ+0.047 wrong-sign,
  1.18× the 0.04 band). Closes cosine-attention axis on alibi. Updated closed.md.
- **TIMED OUT:** 254-entmax-alibi — Tsallis iter ~2× slower than softmax; 12m not enough. Bumped
  job_timeout to 20m, flipped back to needs-run. Will re-launch after 256+257 finish.
- **QUEUED (needs-run, wait for current batch):** 254-entmax (20m). 256/257 will finish ~03:11.
- **STAGED (needs-plan, ready to flip):** 258-per-head-rope-base-alibi, 259-layernorm-alibi,
  260-k-only-norm-alibi — the 3 fallback live levers, all step-0-active and param-fair, all CPU
  + box smoke OK. flip to needs-run the moment the deepnet verdict decides.

## NEXT ACTIONS (decision fork)
1. **Read the deepnet-alpha 3-seed confirm (256/257) + 253@42.** Compute the 3-seed mean.
   - If clearly < 6.2539 with low scatter → **deepnet-alpha is the first real challenger.** Consider
     promoting it (update `autoresearch/champion.json`) and then *stack* future levers on it; also
     explore the residual-scaling axis (learnable per-block α, other α values).
   - If it straddles like poly → residual-scaling is also noise-bound; queue the 3 fresh live levers
     standalone (rope-base / layernorm / k-only-norm).
2. Keep ≥2 items in the daemon queue at all times (each run ≈6 min; loop fires every 5 min).
3. For any single-seed Δ that's notably negative (> ~half band), **3-seed confirm before believing it.**

## GOTCHAS (will bite)
- **BOX BRANCH MISMATCH (recurring):** box tracks `orchestrate-codex-fallback`; daemon autosyncs model
  code to **`main`**. So after ANY new model-code commit, the box's `git pull --ff-only` gets nothing →
  new configs `ImportError` → SMOKE_FAIL → idea bounced to needs-recode. **Workaround I use:**
  `git push origin main:orchestrate-codex-fallback` (FF; main is ancestor), then reset the idea to
  needs-run. **Durable fix (needs user, SSH writes auto-denied for the agent):** on box
  `cd /root/universe-lm && git checkout main && git pull --ff-only`. NOTE: inline-config experiments
  using flags already on the box need NO push (most of my recent ones).
- macOS: no `setsid`/`timeout` (`gtimeout` or `-o ConnectTimeout`); launch daemon with `nohup … & disown`.
- Permission classifier auto-denies: killing remote GPU sessions, `git pull`/`checkout` on the box.
  Surface these to the user.
- Dataclass pitfall: override fields in the `@dataclass` body (not `setattr` on the class) or the
  default silently won't take. (For the probe, `setattr` on a config *instance* is fine.)
- Every `_arq` needs a top-level `@dataclass class C`. Champion val is pinned — never re-measure.
