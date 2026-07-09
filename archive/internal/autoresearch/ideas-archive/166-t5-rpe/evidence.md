# Evidence — 166 t5-rpe

## Verdict: NULL
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52674 (RTX 3060, sm_8.6, driver 580.159.03)
- baseline: CACHED path (cache freshly MEASURED at 2026-06-14T10:32Z, commit=f791d0c, mean=6.4447 ± 0.0488, n=14)
- treatment val: 6.4553 (train=6.4342, val_acc=0.1418)
- Δ vs cache mean: +0.0106 (wrong-sign tiny)
- baseline.sh verdict: **NULL +0.0106** (inside cache band; trt not below mean-band)
- plan bar (166): PASS ≤ ctrl − 0.02 (i.e., trt ≤ 6.4247); NULL |Δ| < 0.02; DRIFT > +0.02
- Δ=+0.0106 inside the |Δ|<0.02 NULL band → NULL
- bpb: n/a (pending harness)
- box check: 166 val=6.4553 sits inside cache `6.4447±0.0488` ⇒ box in-range, no DRIFT
- raw: remote-results/2026-06-14-vast-tiny1m3m/results.json (166 entry appended; log alongside)
- date: 2026-06-14

## Transfer note
T5-XXL (11B) used 32-bucket RPE on attention logits and reached SOTA on SuperGLUE/GLUE/TriviaQA at release (Raffel et al. JMLR 2020). BigBird, REALM, LongT5 re-used T5-RPE at 100M+. **Transfer-risk: med** — T5-RPE is encoder-decoder-native; the autoregressive-LM case has less direct validation, but the mechanism is structurally simple (per-head H×B additive bias, zero-init ⇒ bit-identical step 0) and the bucket parameterization is well-known. The 0.94M null pattern (Δ=+0.0106 wrong-sign tiny, inside null band) is consistent with closed 152/155/160: per-head attention-shape levers (logit-bias / temperature / post-AV gain) all close null at 0.94M/12L/4H — the optimizer absorbs these axes into the existing Q/K/O gradient updates. Additive bias and rotational bias (RoPE/FIRE) are mathematically distinct at infinite sequence length, but at T=2048 the per-position logit bias is dominated by the much larger QK dot-product magnitudes (one-shot bias of ~0 vs accumulated QK magnitude of ~10). **Re-evaluate at ≥135M Phase-2** where the model's per-head specialization gives the additive bias a non-trivial axis to exploit (a known property of RoPE-vs-RPE transfer at T5/PaLM scale).
