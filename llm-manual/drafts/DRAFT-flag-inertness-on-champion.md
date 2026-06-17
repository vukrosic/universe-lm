# DRAFT — Most architecture flags are step-0 inert on the ALiBi champion path

**Hypothesis.** On the tiny1m3m ALiBi champion path (combo-296 + DeepNet-α + poly-ALiBi),
the **majority of `use_*` mechanism flags (~152 of ~200) produce zero step-0 logit
difference** — they are no-ops or zero-init grow-ins; only **~48 fire at init**. Because a
zero-init mechanism has only ~732 steps to grow in, step-0-inert flags overwhelmingly return
**sub-band** in a 3-seed confirm — which is *why* the loop kept drifting into hyperparameter
search (the architecture levers mostly don't move the needle on this base).

**Believed because.** The "safe init" pattern (zero-init gates, residual scalars init 0,
identity-init convs) makes a flag a literal identity map at step 0; on a base already shaped
by ALiBi, many flags add a dormant module the short schedule can't activate. Measured via the
deterministic CPU step-0 probe (under the identical-construction discipline of
[[DRAFT-identical-construction-differencing]]): max|logit_diff| at init across all flags;
~48 fire >1e-6, the rest are 0.0. The two best novel results so far are *right-sign but still
sub-band* (329-gmlp-sgu Δ−0.0123, 334-tied-output-mlp Δ−0.0164), consistent with "fires a
little, grows in a little, doesn't clear the band."

**Test.** The CPU probe (rerunnable, ~minutes, no GPU): for each `use_*` flag, build champion
and champion+flag by the identical path, record max|logit_diff| at init. Cross with closed.md
(which firing flags already lost) and with confirm results (do firing flags clear the band
more often than inert ones?).

**Predicted.** ~48/~200 fire. Among GPU-run flags, P(clears 0.02 band | step-0 inert) ≈ 0;
P(clears band | fires) is higher but still small at this token budget.

**Promotes to.** **L?** (tentative). The flag census is a reproducible *measurement*, but the
inference "step-0 inert ⇒ sub-band" is correlational (confounded with token budget and
param count), and the exact count depends on the flag list. → **L!** needs the firing-vs-inert
split to predict band-clearance at a stated rate with a CI.

**Falsifier.** A **step-0-inert** (zero-init) flag that nonetheless **clears the 0.02 band in
a 3-seed confirm** → breaks "inert ⇒ sub-band"; the mechanism grew in fast enough, and step-0
inertness is not a useful screen. (gmlp-sgu / tied-output-mlp are the live stress test: both
near-inert and right-sign — if either clears the band stacked, this draft is bounded.)

**Scope.** tiny1m3m, combo-296+DeepNet-α+poly-ALiBi champion, 732 steps. Says nothing about
these flags on a different base (e.g. a non-ALiBi recipe) or a longer schedule — both are
separate measurements.

**Why it matters.** Triages the search: spend GPU on flags that *fire*, and prefer mechanisms
that are active at init (not zero-init-gated) so the short schedule can use them. Explains the
HP drift mechanistically rather than as a failure of discipline.

**Evidence so far.** partial — probe transcript this session; needs the firing×band-clearance
table to make the correlation quantitative.

**Blocked on.** A logged table: per GPU-run flag {fires?, 3-seed Δ, cleared band?}.
