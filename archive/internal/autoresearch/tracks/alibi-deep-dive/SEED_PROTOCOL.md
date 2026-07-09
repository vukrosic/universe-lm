# Seed protocol — how many seeds before you trust a result

*Operating rule for any agent running tiny1m3m (or any single-box A/B) in this lab.*

## The one number that governs everything

**Noise band = ±0.02 val loss** (paired, same-seed, tiny1m3m at 3M tokens).
Any val-loss gap **smaller than 0.02 is not real** until proven across seeds.
A single run's val loss has run-to-run scatter on the order of the band itself —
so one seed can flip the ranking of two arms that are actually tied.

## Tiered seeding

Don't seed everything the same. Match seeds to the decision you're making.

| Tier | Seeds | When | What it lets you say |
|---|---|---|---|
| **SCREEN** | 1 | Triage a new arm; kill obvious losers | "direction only" — never a claim |
| **CONFIRM** | 3 | Any arm within ~0.02 of the leader, OR anything you will publicly claim | "real win / real tie / real loss" |
| **ARBITRATE** | 5 | A 3-seed result still straddles the band, or the claim is load-bearing | tie-break a within-noise call |

### The rule in one line
**Screen at 1 seed. Promote anything within 0.02 of the leader to 3 seeds. Never publish a 1-seed number as a result.**

## Decision procedure

1. Run the new arm at **1 seed**.
2. Compare its val loss to the current leader:
   - **Worse by > 0.02** → kill it. Done. (Don't waste GPU 3-seeding a clear loser.)
   - **Within 0.02 (either side)** → it's inside noise. You *cannot* rank it yet. Promote to **3 seeds**.
   - **Better by > 0.02** → promising, but still promote to **3 seeds** before claiming — a single lucky seed can clear the band by itself (see [[autoresearch-champion-pin-optimistic]]).
3. At 3 seeds, compare **means**, and run the proper paired test rather than eyeballing:
   ```
   python confirm_paired.py --a <leader_cfg> --b <arm_cfg> --seeds 42 123 7
   ```
   - mean gap > 0.02 **and** every seed agrees in sign → real.
   - mean gap < 0.02 or seeds disagree in sign → **call it a tie / null**, report it as such.
4. If 3 seeds still straddle the band on a claim you must stand behind → **5 seeds (ARBITRATE)**.

## Anti-patterns (these have bitten this lab)

- **Publishing a 1-seed number.** RQ2's seed-42 spread was 6.240–6.253 = 0.013 — *entirely inside the band*. At 1 seed those four arms are unrankable; the "winner" is noise until 3 seeds say otherwise.
- **Eyeballing means against the band instead of running `confirm_paired.py`.** The band is a screen, not a promotion gate. The paired test is the gate.
- **Lucky-seed promotion.** A 1-seed WIN that clears the band is *not* a confirmed win — it parks in needs-confirm until 3 seeds agree. See [[autoresearch-champion-pin-optimistic]] and the queue-daemon lucky-seed guard.
- **3-seeding obvious losers.** Wastes the single GPU. Kill > 0.02 losers at 1 seed.
- **Garden of forking paths.** Many tiny comparisons → something clears the band by chance. The 1→3 confirm gate is the guard; honor it for *every* claim, not just the convenient ones.

## Why 3 (not 2, not 10)

- 1 seed ≈ the band itself → can flip a true tie.
- 2 seeds → if they disagree in sign you learn nothing and need a third anyway.
- 3 seeds → first count where "all agree in sign" is meaningful and the mean is stable enough to clear a 0.02 gap.
- 5 seeds → only when 3 still straddles a load-bearing claim. Beyond 5 the GPU cost rarely pays for itself at this scale.

## Scale caveat (state this in any writeup)

All of the above is calibrated for **tiny1m3m: 0.94M params, 3M tokens, 732 steps, seq_len 2048**.
The band and the seed counts are *not* assumed to transfer to larger models or longer
context. A result confirmed here is a **tiny-scale** result. Say so.

Related: [[autoresearch-noise-band-too-wide]] · [[autoresearch-champion-pin-optimistic]] · [[baseline-retrained-every-run]]
