# QK-Gain Scaling Experiment Protocol

## Rules

1. **Nano first, then prune up** — run dense experiments on Nano3M32KConfig (fastest, cheapest). Use those results to narrow the gain range for Micro7M1MConfig, then Mini10M20MConfig. Do NOT run full sweeps on larger configs — just test the top 2–3 candidates from the previous tier.

2. **Baseline is reusable** — same config + seed (42) = same baseline. Run once, store in `baselines/`, reuse for all future sweeps on that config. No need to re-run baseline before each sweep.

3. **Sweeps** — for Nano3M32KConfig: run 10–14 gains broadly across the range. For Micro7M1MConfig: run 5–7 gains around the Nano optimal. For Mini10M20MConfig: run 2–3 gains (exploitation only, based on Micro findings).

4. **Memory** — stay within 7.5GB GPU: batch=2, drop to batch=1 if OOM, reduce eval frequency accordingly.

5. **Remote GPU** — vast.ai CUDA instance `root@154.12.38.116`, port 50670, conda env `/venv/main`:
   - rsync config + script → remote → `python3 <script>.py` via background ssh
   - Pull results with `scp` when done

## Configs (defined in `configs/llm_config.py`)

Use these exact config names — do not create new ones:

| Config | Params | Tokens | Steps | Eval every | Baseline |
|--------|--------|--------|-------|------------|----------|
| `Nano3M32KConfig` | 3.2M | 32k | 8 | every step | gain=0.0 → 9.3769 |
| `Micro7M1MConfig` | 6.7M | 1M | ~244 | every 20 steps | gain=0.0 → 6.4675 |
| `Mini10M20MConfig` | 10M | 20M | ~4880 | every 200 steps | gain=0.0 → 5.2041 |

## Baseline storage

All results stored in `baselines/`:
- `nano3m32k_fine_sweep.json` — 14 Nano3M32K gains (0.75–3.5)
- `micro7m1m_sweep.json` — 5 Micro7M1M gains (0.0–4.0)
- `micro7m1m_fine_sweep.json` — 6 Micro7M1M gains (2.0–6.5)
- `mini10m20m_gain4.json` — Mini10M20M gain=4.0 → 4.9816
- `mini10m20m_baseline.json` — Mini10M20M gain=0.0 → 5.2041