# Claim Lock Stress Test

This directory contains a small simulation that hammers `token2science/claim.py`
from many worker processes and checks the audit log for overlapping task holds.

## What it proves

The simulator repeatedly:

1. Picks a random task from `SIM-1` through `SIM-M`.
2. Tries to claim it with a short lease.
3. If the claim is granted, records the hold interval in a per-process audit
   file, sleeps briefly, then releases the task.
4. Merges every audit file at the end and asserts that no two hold intervals
   overlap for the same task.

If the final summary reports `overlaps_detected=0`, the claim lock held up for
the entire run.

## How to run

From `token2science/`:

```bash
python sim/simulate.py
```

The defaults are:

- `--agents 20`
- `--tasks 5`
- `--rounds 5`

You can override them, for example:

```bash
python sim/simulate.py --agents 20 --tasks 5 --rounds 5
```

The script writes temporary state to:

- `sim/.tmp-claims`
- `sim/.tmp-audit`

Both directories are cleared before each run.
