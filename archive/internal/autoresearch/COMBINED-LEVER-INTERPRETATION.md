# Combined Long-Context Lever: Interpretation Guide

**Experiment:** Single 8M run combining RoPE-500k + QK-norm + Diff-Attn vs baseline
**Status:** In progress (~50 min runtime)
**Hypothesis:** The three levers address orthogonal long-context failure modes → stacking should work

---

## The Three Mechanisms (Why They Shouldn't Interfere)

### 1. RoPE Base Scaling (rope_base=500k)
**Problem it solves:** Low-frequency RoPE dims alias (wrap) after ~1k tokens with base=10k
**How it works:** Larger base extends the "turn distance" so low-freq dims complete fewer full turns
**Level it operates:** Positional encoding (before attention softmax)
**Side-effects:** None — pure positional geometry, orthogonal to attention structure

### 2. QK-Norm (use_qk_norm_post_rope)
**Problem it solves:** Attention logits saturate (a few high-norm tokens dominate softmax)
**How it works:** RMSNorm Q and K before dot-product caps logit magnitude
**Level it operates:** Attention logit scale (within softmax, after position)
**Side-effects:** Prevents entropy collapse, shouldn't interact with RoPE base change

### 3. Diff-Attn (use_diff_attn)
**Problem it solves:** Attention background noise dilutes signal (many irrelevant tokens attend weakly)
**How it works:** Two softmax maps, subtract one from other to cancel noise (λ per head, learned)
**Level it operates:** Attention output weighting (post-softmax probability mix)
**Side-effects:** Only affects how attention weights mix; orthogonal to position or logit scale

---

## Expected Outcomes & Interpretation

### Outcome 1: Δ > 0.02 (LARGE WIN)
```
Combined:     3.8500 (example)
Baseline:     4.3200
Δ = −0.4700  (HUGE improvement)
```
**Interpretation:** 
- All three levers contribute meaningfully
- Stacking works, compound effect is real
- This arm **clears the screen** and goes to full ladder (8M/13M/23M/52M/135M)
- Prediction: exponent will be steeper than baseline (especially if RoPE-base is the driver)

**Next steps:** Run full ladder immediately

---

### Outcome 2: Δ 0.01–0.02 (MODERATE WIN)
```
Combined:     4.2100 (example)
Baseline:     4.3200
Δ = −0.1100  (credible, but borderline)
```
**Interpretation:**
- At least one lever helps, but the combination is weaker than expected
- Possible: one lever is null, another's effect is small
- Within the "needs-confirm" band (0.01 < Δ < 0.02)

**Diagnostic: Run 3-seed confirm at 8M**
- If 3-seed mean > 0.01 with p < 0.05 → goes to ladder
- If 3-seed mean < 0.005 or p > 0.05 → flag for ablation (which lever matters?)

**Next steps:** 3-seed confirm, then proceed or diagnose

---

### Outcome 3: Δ 0.005–0.01 (SMALL WIN, NOISY)
```
Combined:     4.3100 (example)
Baseline:     4.3200
Δ = −0.0100  (within noise band, barely positive)
```
**Interpretation:**
- Likely NULL or very small effect
- Could be individual levers canceling out (one helps, another hurts)
- Or: combined benefit is too small to measure at 8M

**Diagnostic: Ablate the combined arm**
- Test RoPE-500k alone, QK-norm alone, Diff-Attn alone
- One might be the driver, others neutral or negative

**Next steps:** Ablate to isolate signal

---

### Outcome 4: Δ ≈ 0 or Negative (NULL/REGRESSION)
```
Combined:     4.3300 (example)
Baseline:     4.3200
Δ = +0.0100  (null, or slightly worse)
```
**Interpretation:**
- Combined arm provides no benefit, or the levers interfere
- Possible: Diff-Attn + QK-norm both affect attention → one destabilizes the other
- Or: RoPE-500k needs careful tuning (base too large?)

**Diagnostic:**
- Run individual levers (RoPE-500k alone, QK-norm alone, Diff-Attn alone)
- Check if any is positive; if all null, the three are orthogonally harmless
- If one is positive and combined is null → stacking issue, need per-lever tuning

**Next steps:** Ablate to identify the bad interaction

---

## Decision Tree (After Results)

```
Monitor: tail -f logs/ladder_combined_8m.log
When done, parse results.jsonl and check:

IF Δ > 0.02
  → PROCEED TO FULL LADDER (8M/13M/23M/52M/135M)

ELSE IF 0.01 < Δ <= 0.02
  → RUN 3-SEED CONFIRM AT 8M
  → IF p < 0.05 → proceed to ladder
  → IF p >= 0.05 → ablate to isolate signal

ELSE IF 0.005 < Δ <= 0.01
  → ABLATE: test each lever alone
  → Identify the driver, drop the nulls
  → Proceed with winner(s) to ladder

ELSE IF Δ <= 0.005
  → ABLATE: test each lever alone
  → Check for interference
  → If all null individually, the combined is safe but useless
  → If interference detected, debug the interaction
```

---

## Parallel: DeepNet Status

While combined 8M is running (~50 min):
- 23M baseline/deepnet on remote box: ETA unknown (transient SSH)
- Ablations (E3/E4) queued, waiting for "LOCAL LADDER COMPLETE"
- Expected: 23M confirms NULL (deepnet ≈ baseline ±0.004), ablations confirm family redundancy

**These are independent processes.** Combined arm result does NOT depend on 23M completion. They run in parallel.

---

## Why This Single Combined Test is Smart

- **Cost:** One 8M run (~50 min) instead of 5 separate runs (4+ hrs) — 5× faster
- **Signal:** If the levers don't interact, combined should preserve individual effects (roughly)
- **Decision speed:** Within 2 hrs we know if the direction is worth full ladder
- **Fallback:** If combined is weak, individual ablations give us granular understanding

If Δ > 0.01, we've found the release ladder's scaling lever in 2 hrs. If Δ < 0.005, we've proven the levers don't help at 8M and should explore other mechanisms (different positional encodings, attention sparsity patterns, etc.).

