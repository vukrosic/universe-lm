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
| 1 | empty | — | — (was 001, finalized 2026-06-09, NULL) |
| 2 | empty | — | — |
| 3 | empty | — | — |

> 2026-06-09 ~06:27-06:44Z batch ran ctrl+001+003+004+005+009+ctrl2 on
> vast-34386 (RTX 3060). 003-orphan FAIL; 001/004/005 NULL; **009 WIN** (largest
> of the day). All 4 are now `done`. **Pipeline is empty of `needs-run`
> candidates** — 010 (needs-review), 006 (planning), 003 (recoding) are mid-loop.
> Trigger the relevant agents to push them through; the runner will pick them
> up once they hit `needs-run`.

Rule: aim for 3 in the queue at all times so the remote never idles.

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

## PENDING — not yet foldered (migrate on first touch)

These are tracked here as a quick-lookup; copy to a numbered folder when
the idea is about to be run. Full spec lives in the table at the end of
this file until then.

Optimizer: Moonlight per-param RMS scaling · Decoupled Q/K from V in Muon routing · Schedule-free AdamW · EMA-of-orthogonalized-matrix.
Architecture: Soft MoE (fallback) · BigBird sparse · Sandwich block · Product-key FFN.
Loss/objective: Sigmoid loss / ET loss · PolyLoss.
Positional: FIRE · CoPE.

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
