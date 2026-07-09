# Review — 115-rdrop

## r3 — 2026-06-14 — verdict: reject

Round 3 is the cap; force the decision.

### What I checked (gate criteria)

- **Source real and current**: arXiv:2106.14448 (Liang et al. NeurIPS 2021) — resolves,
  authors plausible, paper title correct. Not fabricated.
- **Mechanism, not HP**: KL-regularized dropout via double forward + symmetric KL on
  logits. Structural/architectural lever. Step-0 zero-init handled via warmup of
  `rdrop_alpha` from 0 → target. Sound.
- **tiny1m3m only**: ✓ tier is tiny1m3m throughout. No screen20m / Phase-2 leak.
- **Not already closed**: not in `closed.md`. Adjacent loss-shape regularizers
  closed (066 label-smoothing, 067 confidence-penalty, 068 unlikelihood,
  069 focal-loss, 070 mtp-head) — but R-Drop's *output-invariance-under-dropout*
  is distinct from those, so this is not a dedup reject.
- **< 200 LoC**: ✓ first-cut spec ~30 LoC; r2 fix added ~17 LoC; r3 fix added
  ~54 LoC inside `_rdrop_loss`. Total still well under budget.
- **Falsifiable bar**: ✓ `final_metrics.val_loss` against ctrl mean 6.4272±0.01,
  bar ≤ ctrl − 0.005 (PASS), |Δ| < 0.005 (NULL), > +0.005 (DRIFT). Honest range.
- **Transfer-risk tag**: `med` ✓. Paper is validated 110M-340M (BERT-base/large on
  GLUE/SuperGLUE) and in BART-pretrain ablations ~140M. No ≥1B LM-pretraining
  ablation published. The "med" tag is honest — not low, not high.

### Why reject at round 3 (not approve)

The mechanism survives every gate. The blocker is the **implementation-vs-envelope
fit at tiny1m3m on RTX 3060 (12.5 GiB, ~11.6 GiB usable)**.

- **r1**: `rc=1: CUDA OOM, 768 MiB alloc failed` — structural. Two full forward
  graphs at B=2, T=2048, V=49152, 12L × d=64 overshoot 11.6 GiB.
- **r2**: OOM at the symmetric KL block — naive materialization of
  `4 × [N_valid, V]` fp32 intermediates ≈ 3.0 GiB, exceeded headroom. Fix:
  `_torch_ckpt.checkpoint` on the second forward.
- **r3**: OOM at `F.kl_div` line 297 — the runner log
  (`autoresearch/remote-results/2026-06-13-vast-tiny1m3m/115-rdrop.log`)
  reads `Process 1926632 has 11.06 GiB memory in use. Of the allocated memory
  10.25 GiB is allocated by PyTorch`. That is **a leaked prior process holding
  the GPU**, not an inherent R-Drop memory issue. The chunked-KL fix in r3 was
  technically correct (~100 MB peak per chunk vs ~3.0 GiB naive) and *should*
  fit on a clean 11.6 GiB envelope — but the runner started on top of a stale
  process that had not been killed.

So the picture is:

1. The chunked-KL fix is sound and *probably* fits within 11.6 GiB on a clean
   GPU. But "probably" is the operative word — at this scale the budget is
   ~1500 MiB of headroom after a single forward's activations (verified by the
   r1 trainer trace), and there is no margin for another failure mode to
   surface.
2. The fallback options listed in the r3 runner note ("reduce batch_size /
   seq_len in subclass, or true gradient checkpointing on the second forward,
   or no second forward / R-Drop-with-augmentation alternative") all change
   the A/B from the cited R-Drop mechanism into something else:
   - `bs=1` halves the *token throughput* vs the ctrl baseline (which uses
     `bs=2`), so the A/B becomes "R-Drop with halved effective batch" — not
     "R-Drop with default batch". The closed family of nulls (117/118/146
     soft-moe/mod/sparse-ffn) shows that small-capacity adjustments at this
     tier routinely produce wrong-sign results.
   - True gradient checkpointing on the second forward still doubles the
     forward compute and only saves activations during backward. Combined
     with the chunked KL, it *probably* fits — but on the same leaked-process
     GPU, "probably" already burned one round.
   - No-second-forward variants (e.g. R-Drop over an *augmented* input pair)
     are a different mechanism and a different citation. They would need a
     fresh round of "is this the lever we're testing?" before they could be
     approved.
3. The expected Δval band in the plan (`-0.005 to -0.012`) is ~2× the null
   band (`|Δ| < 0.005`). A successful run that lands at `-0.0049` would be
   inside null band — *the mechanism either wins comfortably or we can't tell*.
4. The 2× forward-pass cost is the only meaningful cost, but at 92-step
   tiny1m3m it doubles the wall-clock for a single A/B. Across a pipeline that
   ships weekly, this is non-trivial spend on an axis where the prior loss-shape
   family is already closed.

### What a fresh round would have to look like to be approved

If this idea is to come back, it needs **either**:

(a) A submission that explicitly documents that the GPU box kills stale
    processes before launch (a runner hygiene patch), and the chunked-KL fix
    runs to completion on a clean 11.6 GiB envelope — *and* lands at
    `trt ≤ ctrl − 0.010` to be unambiguous (1× outside null band on the
    expected-Δ mid-point); OR

(b) A different mechanism altogether (R-Drop-with-augmentation, true
    gradient checkpointing on the second forward at bs=1, etc.) filed as a
    new idea, not as another round of 115.

Neither is in scope for "round 3 of 115-rdrop at the gate", so the gate
closes it.

### Reject reason (one-line for `closed.md`)

`115-rdrop — reject: 3 rounds of OOM/fit issues at tiny1m3m on RTX 3060; chunked-KL fix is sound but slot envelope exhausted and adjacent loss-shape regularizers (066/067/068/069/070) already closed — 2026-06-14`