# Evidence — 156 Mixture-of-Attentions (MoA)

## Verdict: NULL
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52674 (RTX 3060, sm_86, 12GB)
- baseline: cached mean=6.4394 ±0.04 (box 5b8a7fea8963, 3 ctrls measured 2026-06-14T00:13:56Z)
- treatment val: 6.4516   Δ vs baseline: +0.0122
- train: 6.3930   val_acc: 0.1406   step-0 val: 10.8125 (bit-identical to ctrl)
- bpb: n/a (pending harness)
- pass/fail bar (plan.md): PASS trt < cached_mean - 0.04; NULL inside band; DRIFT > +0.01 → **not met** (inside band)
- box check: cached mean 6.4394 vs leaderboard 6.4306 (within 0.009 of noise band, healthy)
- raw: remote-results/2026-06-14-vast-tiny1m3m/results.json (log: 156-moa_52674.log)
- date: 2026-06-14

## Transfer note
The MoA lever adds E=2 parallel attention computations per layer with a per-token learned router `W_g ∈ R^{E×d_model}`. Step-0 is bit-identical to baseline (router bias one-hot to expert 0 + zero-init extra-expert K/V ⇒ single-attention output). At 0.94M/12L/4H the two experts do not differentiate in any useful way: train_loss is 0.03 below the ctrl cluster (right sign) but val_loss is 0.04 above (overfit), and the per-token router's ~16K params (≈1.7% overhead) is not productively allocated at this scale. This is the same "capacity-injection levers don't amortize at 0.94M" pattern that closed 117-soft-moe (+0.14) and 118-MoD (+0.10) on the MoE/FFN axis, and 146-sparse-ffn (+0.006) on the FFN-side axis. The mechanism differs from those — MoA mixes full attention computations, not softmaxes or FFNs — but the binding constraint at this tier is the depth-12 residual chain, not the per-layer attention pattern, so adding more parallel paths to the attention does not help. Transfer to >=135M Phase-2: more attention dim gives the router a non-trivial axis to break ties on, and the depth-24+ chain gives the experts time to specialize (e.g., one expert on syntax, one on content). A win at 135M would tell us the binding constraint was capacity-not-depth at 0.94M; the current null is consistent with the closed MoE/FFN-side findings and should be re-evaluated once we have a 135M Phase-2 ladder.
