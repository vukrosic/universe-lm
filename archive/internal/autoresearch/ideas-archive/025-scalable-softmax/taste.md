# 025 — taste review

## r1 — 2026-06-10 — verdict: accept

- **Leverage is real and falsifiable.** At `max_seq_len=2048`, vanilla softmax
  over the attention logits provably flattens as n grows (denominator scales
  with key count, logit variance is fixed), and a 0.94M-param model with a
  handful of heads cannot afford to waste that concentration budget. SSMax's
  `s · log(n)` pre-softmax temperature is the textbook counter-measure: one
  learnable scalar per head (init at 1.0 = identity), fires every step, no
  schedule, no extra params, no new infrastructure. Drop-in ~20 LoC.

- **Information value is high in both directions.**
  - **Win:** 20-LoC transferable sharpening lever, orthogonal to every
    closed/active attention-stability idea in the queue (FIRE 009 = position
    bias, qk-norm 016 = head-dim norm bound, FoX 020 = content probability
    decay, logit-softcap = tanh clamp, softpick 022 = sink-free, gated-attn
    024 = output gate). Stacks on top of FIRE cleanly because it acts on the
    `QKᵀ/√d` line, not the position bias.
  - **Null:** Confirms that softmax flattening is *not* the binding constraint
    at 2048/0.94M — useful knowledge, would let us de-prioritize any future
    "sharpen the attention" lever. Inside ctrl-pair variance is still a
    *result*, not waste.

- **Spirit-check vs closed list passes cleanly.**
  - Closed `logit softcap` (closed.md:23) — *clamps* logit range, different
    mechanism.
  - Closed `020-FoX` (decay on probabilities) — *content* decay, different
    mechanism.
  - Active `016-qk-norm` (WIN at tiny1m3m, Δ-0.014) — bounds norms
    *pre-softmax*; 025 scales logits *by length* pre-softmax. Same line in
    the forward, different operator. Both can stack.
  - Active `022-softpick` — replaces softmax distribution shape entirely
    (rectified softmax / sink-free). Different family.
  - Active `024-gated-attention` — post-AV sigmoid output gate. Different
    family.
  - 025 is not a tweak-of-a-tweak; it is its own lever on its own axis.

- **Niche fit is solid.**
  - Mechanism, not HP ✓
  - Identity/zero-init-able (scalar s=1.0) ✓
  - tiny1m3m-scoped (pathology appears at n=2048, our max_seq_len) ✓
  - One seed (42) per protocol ✓
  - ~20 LoC, drop-in, no new trainer plumbing beyond a flag ✓

- **Crisp bet, one sentence.** "We expect a val-loss drop because at
  `max_seq_len=2048` late-position queries attend over hundreds-to-thousands
  of keys, vanilla softmax provably flattens, and SSMax's per-query
  `s·log(n)` temperature restores per-position sharpness for the few heads
  the tiny model can afford — all in one learnable scalar per head."

- **Portfolio fit is acceptable.** The active queue is currently sparse
  (GPU invariant violated: 0/3 needs-run after 011 WIN), and 025 is a
  low-cost, high-info-whether-it-wins-or-loses idea. Accepting helps fill
  the upstream pipe.

Routing: definition gate (`needs-review`, round reset to 1).
