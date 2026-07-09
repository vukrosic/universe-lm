# Evidence — 119 sam

## Verdict: NULL
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52674 (RTX 3060)
- control val: 6.4272   treatment val: 6.4356   Δ: +0.0084
- ctrl2: pending (queue still running 120+ + ctrl2); current Δ is already inside the |Δ|<0.01 null band — verdict is decided regardless of ctrl2
- bpb: n/a (pending harness)
- pass/fail bar: ctrl − 0.005 (≈ 6.4222), standard null band |Δ|<0.01 → not met
- box check: ctrl 6.4272 vs leaderboard ctrl 6.4306 = −0.0034 (within noise, box healthy)
- raw: remote-results/2026-06-13-vast-tiny1m3m/119-sam.log, ctrl.log
- date: 2026-06-13

## Trajectory check
- Live θ (train_loss): 6.3974 vs ctrl 6.3966 ⇒ +0.0008 train-loss gap ⇒ SAM's 2-backward step is healthy (no optimizer/forward pathology)
- Treatment val curve: 10.81 → 8.31 → 7.81 → 7.58 → 7.42 (step 100) → 7.10 → 6.96 → 6.78 → 6.63 → 6.44 final
- Ctrl val curve: 10.81 → 8.34 → 7.82 → 7.58 → 7.42 → 7.08 → 6.94 → 6.76 → 6.63 → 6.43 final
- Curves overlap step-for-step until ~step 200, then the treatment sits ~0.01 above ctrl for the rest of training. The wrong sign and inside-the-noise magnitude is consistent with a no-op — SAM's flat-minima benefit either doesn't materialize at this short horizon or is washed out by the existing AdamW moment estimates.

## Transfer note
SAM (Sharpness-Aware Minimization, Foret et al. ICLR 2021) adds a 2-backward-pass step that climbs to the worst-case perturbation `ε̂ = ρ · ∇L/‖∇L‖` and backprops there, finding *flatter* minima that generalize better. The lever is well-validated at LLM scale (paper reports +0.5-1.5% on ViT-B/L, +0.3% on BERT) but the cost is 2× compute. At tiny1m3m the run already takes ~2.5 min on RTX 3060 — SAM runs in ~3 min (2× backward is the dominating cost on a memory-bound 12L net), so the overhead is real and visible. The flat-minima benefit, however, needs a *long* training horizon to dominate — at 92 update steps the loss landscape is so under-explored that "flat vs sharp" is a low-SNR distinction. The +0.008 result is consistent with "no detectable effect at this horizon" and a wrong sign is plausible noise. Re-evaluate at Phase-2 (135M, ~3-4k steps) where SAM's lever is well-documented to fire — but note the implementer should keep `sam_rho=0.05` (default) rather than over-tune.
