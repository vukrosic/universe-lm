# Idea queue (index)

**Each idea lives in its own folder under `autoresearch/ideas/<NNN-slug>/idea.md`.**
This file is just the index / status board / active-slot tracker. To add a
new idea: create a folder, write `idea.md`, append to the table below.

**Pipeline:** the review→revise→implement→run loop and the canonical `status`
vocabulary are defined in [`PIPELINE.md`](PIPELINE.md). The routing
truth is the `status:` frontmatter in each `idea.md`; the tables here are a
human-readable view of it. Regenerate with
`grep -H "status:" autoresearch/ideas/*/idea.md`.

## Active remote queue (FIFO, always 3)

Tracks which folder occupies which **GPU slot** — remote execution, *not* the
pipeline `status` field. "Run state" below = idle/in-progress/done on the metal.
Owned by the **runner** (`prompts/runner.md`): it fills/updates these rows as it
launches and pulls. Raw results land in `remote-results/<date>-vast-<tier>/`.

| Slot | Folder | Run | Run state |
|---|---|---|---|
| — | — | — | **GPU IDLE since 2026-06-10T11:27Z** (arq queue hit QUEUE_DONE; no needs-run candidates to launch) — incident, upstream must push 026-030 through code loops OR runner fills with ctrl variance bracket for 10M/30M tier prep |

> **arq queue (10 jobs, detached tmux) — launched 2026-06-10T08:03:46Z on box vast-81.45.65.189 (V100-PCIE-32GB, compute_cap 7.0):**
>
> ```
> 1. ctrl_fire   — _arq_020_ctrl.py  (Tiny1M3MVQGainSWAHighRoPE250KConfig + use_fire_pe=True; ctrl for 020-023)
> 2. 020-fox     — _arq_020.py       (Tiny1M3MFOXOnFireConfig)
> 3. 021-vres    — _arq_021.py       (Tiny1M3MVResidualOnFireConfig)
> 4. 022-soft    — _arq_022.py       (Tiny1M3MSoftpickOnFireConfig)
> 5. 023-canon   — _arq_023.py       (Tiny1M3MCanonOnFireConfig)
> 6. 024-ctrl    — _arq_024_ctrl.py  (Tiny1M3MConfig + use_fire_pe=True; plan-024 ctrl)
> 7. 024-gated   — _arq_024.py       (Tiny1M3MGatedAttnOnFireConfig)
> 8. 025-ctrl    — _arq_025_ctrl.py  (Tiny1M3MConfig plain; plan-025 ctrl)
> 9. 025-ssmax   — _arq_025.py       (Tiny1M3MSSMaxConfig)
> 10. ctrl_fire2 — _arq_020_ctrl.py  (variance bracket)
> ```
>
> Box env: `/venv/main/bin/python` + `PYTHONPATH=/usr/local/lib/python3.12/dist-packages` (torchtune). All 9 configs build-smoke OK on CPU (params 949K-954K). Wall-clock target: ~5 min/job × 10 = ~50 min.

> 006-schedule-free-adamw finalized 2026-06-09T12:01Z — NULL (trt 6.8056 vs ctrls
> 6.5953/6.6091, +0.21 worse). 010-polyloss finalized 2026-06-09T12:11Z — NULL
> (trt 6.5938 vs ctrls 6.5991/6.6050, inside variance). ⚠️ both batches: session
> ctrl drifted +0.19 vs prior days — within-session A/B valid, cross-day not (see
> evidence.md / closed.md; suspected baseline pollution from wholesale file sync).

> 2026-06-09 ~06:27-06:44Z batch ran ctrl+001+003+004+005+009+ctrl2 on
> vast-34386 (RTX 3060). 003-orphan FAIL; 001/004/005 NULL; **009 WIN** (largest
> of the day). All 4 are now `done`. **Pipeline is empty of `needs-run`
> candidates** — 010 (needs-review), 006 (planning), 003 (recoding) are mid-loop.
> Trigger the relevant agents to push them through; the runner will pick them
> up once they hit `needs-run`.

> ## 🔴 THE GPU MUST NEVER BE IDLE
> Hard invariant, not an aspiration. Keep **≥3 ideas at `needs-run`/`running`** so
> the remote always has work the moment a slot frees. An idle box = wasted rented
> compute = an **incident**: find the upstream stage starving the queue (taste →
> definition → code → mining) and drain it immediately. If all slots are empty and
> nothing is `needs-run`, the upstream agents are behind — launch them *now*, don't
> wait for a human to notice. See the prime directive in [`PIPELINE.md`](PIPELINE.md).

## Ideas board (in folder-number order)

Pure index — **no status column on purpose.** The `status:` frontmatter in each
`idea.md` is the *only* source of truth (see [`PIPELINE.md`](PIPELINE.md)). To
see live status: `grep -H "status:" autoresearch/ideas/*/idea.md`. Never copy
status into this file — that is exactly the drift that breaks the loop.

| # | Folder | One-liner | Expected Δ |
|---|---|---|---|
| 001 | `001-cautious-muon/` | sign-mask on Muon ortho'd update | −0.01 to −0.05 |
| 002 | `002-cautious-adamw/` | sign-mask on AdamW (1D params) | −0.005 to −0.02 |
| 003 | `003-soap/` | Shampoo + Adam hybrid in eigenbasis | −0.02 to −0.05 |
| 004 | `004-retnet-retention/` | linear-attention retention (parallel/recurrent) | −0.02 to −0.06 (if transfers) |
| 005 | `005-decoupled-qkv-muon/` | split fused qkvo into 4 matrices for Muon routing | −0.005 to −0.02 |
| 006 | `006-schedule-free-adamw/` | AdamW w/o LR schedule, iterate-averaging | −0.005 to −0.02 |
| 007 | `007-sigmoid-loss/` | per-token sigmoid + z-loss replacing softmax CE | −0.005 to −0.02 |
| 008 | `008-gated-deltanet/` | linear attention w/ delta rule + input gate (O(n), no softmax) | −0.02 to −0.06 (if transfers) |
| 009 | `009-fire-pe/` | learnable position bias w/ fixed decay kernel (drop-in for RoPE) | −0.005 to −0.02 |
| 010 | `010-polyloss/` | CE + ε·(1−p_t) polynomial correction (label-smoothing generalization) | −0.005 to −0.02 |
| 011 | `011-cautious-lion/` | sign-mask on Lion's sign-update (Cautious generalized to Lion) | −0.01 to −0.03 |
| 012 | `012-gated-deltanet/` | linear attention with delta rule + input gate (subsumes 008) | −0.02 to −0.06 (if transfers) |
| 013 | `013-cope/` | content-conditional position offset (probe-based, drop-in for RoPE) | −0.005 to −0.02 |
| 014 | `014-sigmoid-loss/` | sigmoid + z-loss replacing softmax CE (distinct from 010) | −0.005 to −0.02 |
| 015 | `015-moonlight-muon-rms/` | Muon per-tensor RMS rescale `c·√max(d_in,d_out)` (Moonlight paper) | −0.01 to −0.05 |
| 016 | `016-qk-norm/` | LayerNorm on Q,K head-dim (per-head logit bounding) | −0.005 to −0.02 |
| 017 | `017-sub-ln-sandwich/` | LN_post wrap on each sublayer output (DeepNet §3.1) | small win or null at 6L |
| 020 | `020-forgetting-attn/` | per-head, per-token learnable forget-gate (multiplicative decay on softmax) | small win or null on top of 009 FIRE |
| 025 | `025-scalable-softmax/` | SSMax — per-head learnable `s·log(n)` temperature on attention logits (arXiv:2501.19399) | −0.01 to −0.03 or informative null |
| 026 | `026-fire-x-qknorm/` | Composition: FIRE positional bias × QK-Norm — stack both attention levers (009 × 016) | −0.07 to −0.09 (additive) or informative null |
| 027 | `027-moonlight-x-qknorm/` | Composition: Moonlight Muon RMS × QK-Norm — stack optimizer scale-align + per-head logit bound (015 × 016) | −0.02 to −0.03 (additive) or informative null |
| 028 | `028-deep-thin-config/` | Deep-and-thin config: more layers, smaller d_model at fixed ~0.94M param budget (MobileLLM ICML 2024) | +2.7% on benchmarks at 125M per paper |
| 029 | `029-v-norm/` | V-Norm: per-head LayerNorm on Value projections before AV product (symmetric to QK-Norm 016) | −0.005 to −0.015 or informative null |
| 030 | `030-unet-skip-sigmoid/` | U-Net skip gates with sigmoid(−1.5) init fix — ~5 LoC fix to existing unet_skip_gates code (modded-nanogpt PR #125) | +1.25% speedrun equivalent |
| 081 | `081-scale-invariant-attn/` | scale-invariant attention logit transform / p-RoPE | −0.01 to −0.03 or informative null |
| 082 | `082-diff-transformer/` | differential attention with noise-canceling twin softmax | −0.01 to −0.04 or informative null |
| 083 | `083-root-optimizer/` | orthogonalized optimizer + soft-threshold outlier suppression | −0.01 to −0.03 or informative null |
| 084 | `084-peri-ln/` | peri-normalization around sublayers | −0.005 to −0.02 or informative null |
| 085 | `085-hybridnorm/` | QKV norm in attention + Post-Norm FFN | −0.005 to −0.02 or informative null |
| 086 | `086-muddformer/` | MUDD residual connections to break bottlenecks | −0.01 to −0.03 or informative null |
| 087 | `087-entropy-guided-attn/` | entropy-guided head-diversity regularization | −0.005 to −0.02 or informative null |
| 088 | `088-rodimus/` | linear recurrent attention with DDTS selection | −0.01 to −0.03 or informative null |
| 089 | `089-pi-attention/` | periodic sparse ring-local + stride-skip fusion | −0.01 to −0.03 or informative null |
| 090 | `090-tapernorm/` | gated normalization removal with foldable affine tail | −0.005 to −0.02 or informative null |
| 091 | `091-scale-anchor-loss/` | fixed-target residual scale anchor near logits | −0.005 to −0.015 or informative null |
| 092 | `092-seednorm/` | input-conditioned normalization scale coefficient | −0.005 to −0.02 or informative null |
| 093 | `093-derf/` | erf-based pointwise norm replacement | −0.005 to −0.02 or informative null |
| 094 | `094-keel-highway-postln/` | Post-LN with highway residual connection | −0.01 to −0.03 or informative null |
| 095 | `095-bhyt/` | bounded tanh Pre-LN alternative with one-time stats | −0.005 to −0.02 or informative null |
| 096 | `096-siamesenorm/` | dual pre/post-norm streams with shared parameters | −0.005 to −0.02 or informative null |
| 097 | `097-asentmax/` | adaptive sparse entmax attention for long context | −0.01 to −0.03 or informative null |
| 098 | `098-lp-qknorm/` | QK geometry via Lp norm family | −0.005 to −0.02 or informative null |
| 099 | `099-double-p/` | hierarchical top-p sparse attention | −0.01 to −0.03 or informative null |
| 100 | `100-mud/` | cheaper Muon whitening surrogate | −0.005 to −0.02 or informative null |
| 101 | `101-trasmuon/` | orthogonalized momentum plus trust-region scaling | −0.005 to −0.02 or informative null |
| 102 | `102-sf-normuon/` | schedule-free spectral optimizer | −0.005 to −0.02 or informative null |
| 103 | `103-momentum-streams/` | optimizer-like residual streams inside the model | −0.005 to −0.02 or informative null |
| 104 | `104-post-norm-resharpen/` | post-norm to counter attention dispersion | −0.005 to −0.02 or informative null |
| 105 | `105-retrospective-sparse-attn/` | sparse attention that revises old outputs | −0.005 to −0.02 or informative null |
| 106 | `106-mvn-grad/` | variance-normalized gradients with post-normalization momentum | −0.005 to −0.02 or informative null |

## PENDING — not yet foldered (migrate on first touch)

These are tracked here as a quick-lookup; copy to a numbered folder when
the idea is about to be run. Full spec lives in the table at the end of
this file until then.

Optimizer: Decoupled Q/K from V in Muon routing · Schedule-free AdamW · EMA-of-orthogonalized-matrix · AdEMAMix dual-EMA AdamW (018).
Architecture: Soft MoE (fallback) · BigBird sparse · Product-key FFN.
Loss/objective: Sigmoid loss / ET loss · PolyLoss.
Positional: FIRE · CoPE.
Attention stability: Forgetting Transformer per-head decay (020).
Normalization: Dynamic Tanh / DyT (019).
Architecture: Value Residual Learning · cross-layer V shortcut (021) · arXiv:2410.17897.
Attention: Softpick rectified-softmax · sink-free normalization (022) · arXiv:2504.20966.
Architecture: Canon layers · gated depthwise causal Conv1d on residual stream (023) · Griffin arXiv:2402.19427 / Physics-of-LMs Canon.
Attention: Gated Attention · per-head sigmoid output gate post-AV (024) · arXiv:2505.06708.
Attention: Scalable-Softmax / SSMax · length-aware attention temperature (025) · arXiv:2501.19399.

## Scale-tier backlog (NOT tiny1m3m ideas — tested on the 10M+ ladder)

Data-mix, LR-schedule, and tokenizer levers land here, not in idea folders
(see `plans/beat-smollm2-135m.md` Phases 1-2). Format: `<lever> · <source> ·
needs ≥10M tier`.

## CLOSED ideas (do not re-propose)

Moved to [`closed.md`](closed.md) — the loop's dedup list (miner reads it,
reviewer appends on `reject`). `LEADERBOARD.md` remains the full human results
record.

## External sources to mine (refresh weekly)

- arXiv: `cs.LG`, `cs.CL` — filter "Muon", "orthogonal", "spectral", "MoE", "state space", "Mamba", "DeltaNet", "linear attention", "cautious"
- **科学空间 / kexue.fm (Su Jianlin 苏剑林)** — https://kexue.fm — originator of RoPE; deep mechanism-level posts on attention, optimizers (Muon, Tiger), normalization, length extrapolation. Chinese-language; the agent reads it natively. Browse `https://kexue.fm/archives/<id>` and the front-page index. (Anti-bot JS wall blocks plain WebFetch/curl — agent may need a browser tool or the user pastes the text.)
- X follows: @kellerjordan0, @borisdayma, @arankomatsuzaki, @_akhaliq, @hardmaru, @StasBekman, @cloneofsimo
- Repos: modded-nanogpt, picoGPT, llm.c, nanogpt-speedrun, PaLM-pytorch, mamba
- HF papers: https://huggingface.co/papers
- Papers With Code: https://paperswithcode.com/task/language-modelling

## Remote run log

| Date | Slot | Folder | Run | Status | Val loss | Δ vs ctrl |
|---|---|---|---|---|---|---|
| 2026-06-08 | — | — | tiny1m3m ctrl (1B data, T4) | DONE | 6.4287 | — |
| 2026-06-09 | — | — | tiny1m3m ctrl (vast-34386, RTX 3060) — arq-r1 first ctrl | DONE | 6.3875 | — |
| 2026-06-09 | — | — | tiny1m3m ctrl (vast-34386, arq-r1 second ctrl, variance bracket) | DONE | 6.4050 | — |
| 2026-06-09 | 1 | `001-cautious-muon/` | tiny1m3m + cautious-muon, s42 (vast-34386) | DONE (NULL) | 6.4125 | +0.025 / +0.0075 |
| 2026-06-09 | — | `003-soap/` | tiny1m3m + SOAP, s42 — orphan smoke (003 was `recoding` not `needs-run`, runner violation) | FAILED (rc=1) | — | — |
| 2026-06-09 | — | `004-retnet-retention/` | tiny1m3m + retention probe, s42 (v1 kernel+probe, v2 not wired) | DONE (NULL) | 6.4162 | +0.029 / +0.011 |
| 2026-06-09 | — | `005-decoupled-qkv-muon/` | tiny1m3m + decoupled-QKV, s42 (code loop skipped) | DONE (NULL) | 6.3909 | +0.003 / -0.014 |
| 2026-06-09 | — | `009-fire-pe/` | tiny1m3m + FIRE, s42 | DONE (**WIN**) | 6.3234 | **-0.064 / -0.082** |
| 2026-06-09 | — | `006-schedule-free-adamw/` | tiny1m3m + SF-AdamW, s42 (vast-34386) | DONE (NULL) | 6.8056 | +0.21 / +0.20 — clear negative |
| 2026-06-09 | — | `010-polyloss/` | tiny1m3m + PolyLoss, s42 (vast-34386) | DONE (NULL) | 6.5938 | −0.0053 / −0.0112 — inside ctrl-pair gap (0.0059) |
| 2026-06-09 | — | `011-cautious-lion/` | tiny1m3m + cautious-lion, s42 (vast-34386) | DONE (**WIN**) | 6.3941 | **−0.0312 / −0.0321** ≫ gap 0.0009 — in-session WIN |
| 2026-06-10 | 1 | `020-forgetting-attn/` | tiny1m3m + FoX, s42 (vast-81.45.65.189, V100) | **FAILED (NaN both runs)** | NaN | needs-recode — FoX row-renorm blow-up at step ~400/732 |
| 2026-06-10 | 2 | `021-value-residual/` | tiny1m3m + V-residual, s42 (vast-81.45.65.189) | DONE (WIN w/ caveat) | 6.3075 | −0.0344 vs shared fire-ctrl 6.3419 (joint V-res+FIRE vs no-FIRE; shared ctrl buggy) |
| 2026-06-10 | 3 | `022-softpick-attention/` | tiny1m3m + softpick, s42 (vast-81.45.65.189) | **FAILED (NaN both runs)** | NaN | needs-recode — softpick row-renorm blow-up |
| 2026-06-10 | — | `023-canon-conv/` | tiny1m3m + Canon, s42 (vast-81.45.65.189) | DONE (WIN w/ caveat) | **6.2581** | **−0.0838** vs shared fire-ctrl 6.3419 (best of day; after stripping FIRE ~−0.07 from 009, Canon effect still ~−0.06 ≫ WIN bar) |
| 2026-06-10 | — | `024-gated-attention/` | tiny1m3m + gated, s42 (vast-81.45.65.189) | DONE (WIN w/ caveat) | 6.3316 | −0.0953 vs plan-024 ctrl 6.4269 (plan-024 ctrl buggy — no fire); isolated gated effect TBD |
| 2026-06-10 | — | `025-scalable-softmax/` | tiny1m3m + SSMax, s42 (vast-81.45.65.189) | DONE (WIN w/ caveat) | 6.3359 | −0.0910 vs plan-025 ctrl 6.4269 (TWO bugs: plan-025 ctrl missing use_fire_pe + SSMax trt config missing use_fire_pe); spec'd SSMax+FIRE vs FIRE unmeasured |
| 2026-06-10 | — | shared fire-ctrl ×2 | `Tiny1M3MVQGainSWAHighRoPE250KConfig + use_fire_pe=True` subclass | ok_but_buggy | 6.3419 / 0.1511 | **WIRING BUG** — subclass override silently dropped `use_fire_pe=False`; 4 -sh reruns (ctrl_fire, ctrl_fire2, 024-gated-sh, 025-ssmax-sh) all produced identical 6.3419 to 4 dp → confirms bug, not noise |

## Protocol (per-idea folder)

When an idea moves to implementation, add these files alongside `idea.md`:

| File | When | Contents |
|---|---|---|
| `plan.md` | promoting to implementation | implementation spec, flags, controls, expected cost |
| `review.md` | parallel-AI code review | critique, suggestions, sign-off |
| `evidence.md` | after a run | val loss, commit link, Δ vs ctrl, verdict |
| `notes.md` | anytime | scratchpad, dead-ends, follow-up ideas |

To re-prioritize the queue: renumber folders. To close: set the `idea.md`
frontmatter `status` to `rejected` (killed in review) or leave at `done` (ran),
move a `rejected` folder to `autoresearch/ideas/_closed/`, and append a line to the
CLOSED section below. See [`PIPELINE.md`](PIPELINE.md) for the full
state machine.
