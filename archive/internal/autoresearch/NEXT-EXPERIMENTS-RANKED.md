# Next Experiments — Ranked by Priority

**Strategy:** DeepNet is closed (Muon-redundant). Release ladder needs a scaling lever. Long-context mechanisms are the only remaining class Muon does NOT replace.

**Selection criteria:** Already wired in code, fast screen (8M is ~50 min per arm), high long-context upside.

---

## TIER 1: Fast Track (No New Params, Step-0 Active)

### Experiment 1: RoPE Base Scaling (`rope_base=100k/250k/500k`)
**Hypothesis:** Larger RoPE base de-aliases low-frequency position encoding over long contexts → steeper exponent.

**Mechanism:** RoPE rotary base controls how many "turns" the lowest-frequency dims complete across sequence length. Base=10k wraps position info after ~1k tokens. Larger base preserves distance information at range.

**What to test:**
- Baseline: `rope_base=10000` (default)
- Arm 1: `rope_base=100000`
- Arm 2: `rope_base=250000`
- Arm 3: `rope_base=500000`

**Ladder runs:** 8M/13M/23M/52M/135M (full 5-rung ladder)

**GPU cost:** ~50 min per arm at 8M → ~4 hrs for quick screen (baseline + 3 arms at 8M)

**Expected outcome:** 
- ✓ Steeper exponent (de-aliasing pays off more as model/context grows)
- ✓ Better needle-in-haystack retrieval at 2×–4× train length
- ✗ Or: all arms collapse to same curve (base doesn't matter at 8M depth)

**Success criteria:** Δ > 0.02 vs baseline at 8M, positive trend across rungs

**Failure criteria:** All bases equivalent, or worse at higher base

---

### Experiment 2: QK-Norm Post-RoPE (`use_qk_norm_post_rope=True`)
**Hypothesis:** RMSNorm on Q/K before dot-product prevents attention logit magnitude blow-up → stabilizing, may also be a steeper exponent if entropy-collapse is scale-dependent.

**Mechanism:** Long sequences suffer "logit saturation" where a few high-norm tokens dominate softmax → distant tokens starved of probability. QK-norm caps logit scale, keeps softmax able to attend far.

**What to test:**
- Baseline: no QK-norm
- Arm 1: `use_qk_norm_post_rope=True`

**Ladder runs:** 8M/13M/23M/52M/135M (full ladder)

**GPU cost:** ~50 min at 8M (one flag, no new params)

**Expected outcome:**
- ✓ Constant intercept shift (stabilizing, maybe small constant loss win)
- ✓ Or steeper exponent (if entropy-collapse is a scale-dependent failure mode)
- ✓ Better training stability late (prevents divergence on upper rungs)

**Success criteria:** Δ > 0.005 vs baseline (even intercept shift earns the slot if stable)

**Failure criteria:** No change, or regression

---

### Experiment 3: Screening Both at 8M (RoPE + QK-Norm)
**Why test together:** Both are step-0 active, no new params, safe to run in parallel. One might be the winner, both might be needed.

**GPU cost:** 2 additional arms at 8M = ~100 min total

---

## TIER 2: Parallel Discovery (New Operator, Watch Convergence)

### Experiment 4: Differential Attention (`use_diff_attn=True`)
**Hypothesis:** softmax₁ − λ·softmax₂ (learnable λ per head) cancels attention noise → steeper exponent, better needle-in-haystack.

**Mechanism:** Two attention maps, subtraction removes diffuse background noise that dilutes signal tokens. Paper result: "lost-in-the-middle" rescue.

**What to test:**
- Baseline: standard attention
- Arm 1: `use_diff_attn=True` (λ init 0.5)

**Ladder runs:** 8M/13M only at first (watch tiny screen for convergence; new operator = risk)

**GPU cost:** ~100 min at 8M + 13M (~2 hrs to screen)

**Expected outcome:**
- ✓ Steeper exponent (noise floor scales with sequence length)
- ✓ Better "lost-in-middle" performance on long-doc QA
- ✗ Or: convergence issue on tiny screen (new operator = watch for divergence)

**Success criteria:** Δ > 0.01 at 8M, clean convergence, 13M confirms

**Failure criteria:** Divergence, or null at 8M/13M

---

## TIER 3: Heaviest Lift (After Tier 1+2 Confirm)

### Experiment 5: Intra-Doc Mask (`use_intra_doc_mask=True`)
**Hypothesis:** Forbid cross-document attention in packed training → model learns to use distant context instead of ignoring it as noise → steeper exponent on long-range evals.

**Mechanism:** Standard packing lets attention bleed across concatenated docs. Model learns that far-back = usually irrelevant noise. Masking to true document boundaries fixes that prior.

**What to test:**
- Baseline: standard packing (cross-doc allowed)
- Arm 1: `use_intra_doc_mask=True` (intra-doc causal only)

**Ladder runs:** 8M/13M/23M (requires collate + kernel changes, heavier wiring)

**GPU cost:** ~2 hrs for wiring + 150 min for 8M/13M screen

**Expected outcome:**
- ✓ Steeper exponent on *long-context eval* (loss may stay flat if eval is on-distribution)
- ✓ Better needle-in-haystack vs distractor docs
- ✗ Or: null on loss (benefit only on capability eval, not perplexity)

**Success criteria:** Δ > 0.01 on long-range benchmark (even if loss is flat)

**Failure criteria:** Regression on loss or eval

---

## Decision Tree

```
START
├─ Run Experiment 1 (RoPE) + Experiment 2 (QK-norm) in parallel at 8M
│  ├─ IF either Δ > 0.02 → carry to full ladder (8M/13M/23M/52M/135M)
│  ├─ IF both Δ < 0.005 → move to Tier 2
│  └─ IF Δ 0.005–0.02 → 3-seed confirm at 8M before full ladder
│
├─ (Parallel) Run Experiment 4 (Diff-Attn) at 8M/13M
│  ├─ IF Δ > 0.01 + clean convergence → carry to full ladder
│  ├─ IF divergence or Δ < 0.005 → deprioritize
│  └─ IF Δ 0.005–0.01 → 3-seed confirm before ladder
│
└─ After Tier 1+2 winners confirmed:
   ├─ Experiment 5 (Intra-Doc) if time/GPU allows
   └─ If two levers both win, judge by exponent steepness at 135M
```

---

## Resource Estimates

| Experiment | Rung | GPU Cost | Timeline |
|---|---|---|---|
| RoPE (8M screen) | 8M | ~50 min | 1 hr |
| QK-norm (8M screen) | 8M | ~50 min | 1 hr |
| Diff-Attn (8M/13M screen) | 8M/13M | ~100 min | 2 hrs |
| Winner (full ladder) | 8M/13M/23M/52M/135M | ~12–15 hrs | 1 day |

**Fast path:** RoPE + QK-norm screen at 8M (~2 hrs) → winner to ladder (~12 hrs) = **~14 hrs total**

---

## Success Definition for Release

A lever earns the 135M release run if:
1. **Fitted L(N) sits below baseline at 135M** (exponent comparison, preferably steeper α)
2. **Does NOT regress long-context eval** (needle-in-haystack, long-doc QA, code retrieval)
3. **Verdict is robust across all 5 rungs** (not a lucky 8M artifact)

---

## Why NOT Other Candidates

- **Optimization levers (deepnet/rezero/layerscale):** Muon-redundant, closed
- **Positional levers (ALiBi/poly-ALiBi/KERPLE):** Distance-punishing, banned by D002
- **Model-capacity levers (width/depth):** Ladder controls, not a lever
- **HP search (LR/wd/momentum):** RULE 0 forbids this; only structural mechanisms

---

## My Recommendation

**Run Experiments 1 + 2 in parallel at 8M tonight** (~2 hrs GPU).
- If either clears screen (Δ > 0.02) → carry both to full ladder
- If both null → Experiment 4 (Diff-Attn) is next
- If mixed → 3-seed confirm the borderline, prioritize the winner

This gets you to a release-lever decision by EOD tomorrow.
