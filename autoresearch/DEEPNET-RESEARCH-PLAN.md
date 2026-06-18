# Autonomous DeepNet Research Execution Plan

**Current status:** Main ladder 8M/13M complete (4 points, both NULL). 23M in flight. Ablations queued and waiting.

## Event Timeline (Real-Time Triggers)

### Event 1: 23M Baseline Completion (ETA ~10–12 hrs from run start)
When: box completes the 23M baseline rung and logs the point
Trigger: `ladder_status.py --no-pull` sees 6 points (baseline+deepnet × 8M/13M/23M)
Action:
1. Harvest box results via SSH: `ladder_status.py` (pulls remote results.jsonl)
2. Check 23M verdict: baseline vs deepnet delta
3. IF 23M delta is NULL (|Δ| ≤ 0.02): **exponent comparison is already conclusive**
   - Parallel curves (baseline and deepnet) indicate H0/H2 (null / constant shift)
   - Ladder is done with the 8M/13M/23M canonical fit; can proceed to ablations
4. Log point to local results.jsonl

### Event 2: "LOCAL LADDER COMPLETE" Marker (box writes it)
When: 23M deepnet finishes, `run_ladder_local.sh` writes marker to logs/ladder_driver.log
Trigger: `run_deepnet_ablations.sh` polls for this line
Action:
1. Ablation driver unblocks and starts sequentially:
   - E3: `run deepnet_ab` (α + β init)
   - E4: `run rezero` (learned scalar)
   - E4: `run layerscale` (learned per-channel)
2. Each arm runs at the 8M rung (cheap, ≥3 seeds for stability)
3. Results logged to ladder/results.jsonl

### Event 3: Auto-Fit Trigger (≥3 Rungs Per Arm)
When: ladder_status.py sees 3 or more distinct N for any arm (baseline/deepnet both have 3)
Trigger: automatic; happens after 23M logs
Action:
1. `ladder_status.py` calls `scaling_fit.py --baseline baseline`
2. Fit L(N) = E + A·N^(-α) for all arms
3. Print exponent comparison table:
   ```
   baseline: α = X.XXX
   deepnet:  α = X.XXX (Δα = ±Y.YYY)
   deepnet_ab: α = X.XXX
   rezero:   α = X.XXX
   layerscale: α = X.XXX
   ```
4. If all exponents cluster (Δα < 0.01): **family redundancy CONFIRMED**
5. If deepnet α > baseline α by > 0.05: **steeper exponent, warrants investigation**

### Event 4: Ablation Completion
When: E3/E4 arms finish (residual 3 runs × ~50 min = ~150 min total)
Trigger: ablation_driver writes "DEEPNET ABLATIONS COMPLETE"
Action:
1. Run `finalize_deepnet_synthesis.py` (checks readiness + prints comprehensive table)
2. Examine E3/E4 deltas at 8M:
   - E3 (deepnet_ab): predicted NULL (β adds nothing on top of α)
   - E4 (rezero, layerscale): predicted NULL (family-wide redundancy)
3. If all 5 arms (deepnet + 4 ablations) land within ±0.005 of baseline:
   → **full redundancy confirmed**, write final synthesis update

## Synthesis Update Workflow (Post-Events)

After 23M lands AND ablations complete, manually run:
```bash
python3 autoresearch/bin/finalize_deepnet_synthesis.py
```

This prints:
- Readiness check (has 23M? has ablations?)
- Comprehensive table (all 6 arms × 3 rungs)
- 23M/ablation verdicts
- Recommendation for next step

Then update `DEEPNET-SYNTHESIS.md`:
1. Fill in E1 final 23M row (baseline, deepnet, Δ)
2. Fill in E3 result (deepnet_ab comparison)
3. Fill in E4 results (rezero, layerscale comparison)
4. Add exponent fit table from `scaling_fit.py` output
5. Write final conclusion:
   - IF all exponents match: "H0 confirmed, deepnet does not bend scaling curve"
   - IF deepnet exponent is steeper: investigate the mechanism
   - Recommendation for release (no vs yes, with caveats)

## Next Research Direction (Post-Synthesis)

**AFTER deepnet closes** (synthesis finalized), pivot immediately to long-context levers:

1. **Wire arms 1 & 3** (RoPE-base, QK-norm) — see `NEXT-LADDER-ARMS.md`
   - Trivial config subclasses, no new params, step-0 active
   - Run at 8M/13M parallel with ablations (no VRAM contention)
2. **Parallel:** run arm 2 (diff-attn) at 8M/13M, watch tiny screen for convergence
3. **Selection:** fitted L(N) at 135M target picks the winner

## Success Metrics

DeepNet study is successful if:
1. ✓ Mechanism is **mechanistically understood** (Muon-redundancy via init-probes)
2. ✓ Empirical verdict is **replicated across rungs** (8M/13M/23M all NULL or all non-NULL)
3. ✓ Exponent comparison is **definitive** (fitted curves either parallel or clearly separated)
4. ✓ Family specificity is **confirmed** (ablations show E3/E4 are also NULL)
5. ✓ Strategy is **locked** (next direction pivots to long-context, not more optimization-stability)

Current status: 1–2 complete, 3–4 in flight, 5 ready on signal.

## Operational Notes

- Box is unreachable intermittently (transient SSH timeouts). These self-recover; do not panic.
- The finalization script (`finalize_deepnet_synthesis.py`) is idempotent — run it repeatedly, it updates on new data.
- The synthesis update is **manual** (not auto-scripted) to preserve narrative quality. But the data table updates are templated for speed.
- Keep `DEEPNET-RESEARCH.md` and `DEEPNET-SYNTHESIS.md` in sync as the final write-up evolves.
