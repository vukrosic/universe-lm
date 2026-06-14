# Run contract — the deterministic GPU handoff

The GPU last-mile (run → poll → pull → judge → flip) is drained by a **plain
script**, [`bin/queue-daemon.sh`](bin/queue-daemon.sh) — no LLM in the hot loop.
That is only possible because every `needs-run` idea arrives in a **fixed,
machine-readable shape**. The implementer ([`prompts/code-implementer.md`](prompts/code-implementer.md))
produces that shape; the daemon consumes it. This file is the contract between
them.

> AI still owns everything *upstream*: mining, taste, review, and writing the
> code. The daemon owns only the mechanical last mile, where the verdict is pure
> arithmetic ([`bin/baseline.sh verdict`](bin/baseline.sh)) and nothing needs
> judgment.

## What the implementer must emit (at release to `needs-run`)

Two files, alongside the idea's existing `idea.md` / `plan.md`:

### 1. `autoresearch/ideas/<idea>/run.json`

The deterministic descriptor the daemon reads. Minimal, extensible:

```json
{
  "name": "157-conv-ffn",
  "arq_file": "_arq_157-conv-ffn.py",
  "job_timeout": "12m"
}
```

| field | required | meaning |
|---|---|---|
| `name` | ✅ | run/log name. Use the idea slug. Logs are named `<name>.log`. |
| `arq_file` | ✅ | path **relative to repo root** of the treatment entry the daemon runs as `python <arq_file>`. By convention `_arq_<idea>.py`. |
| `job_timeout` | optional | per-job `timeout(1)` cap (default `12m`). A timed-out job is logged `FAIL` and bounced. Bump only for a legitimately heavy idea (MoS etc.). |

The daemon **skips** any `needs-run` idea without a valid `run.json` + an
existing `arq_file`, and says so in its report (that idea never reached the box —
an implementer bug, not a run result).

### 2. `_arq_<idea>.py` (repo root) — the treatment entry

Self-contained, seed-42, flag-ON. **Must define a top-level config class named
`C`** (a tier-config subclass with the idea's flag(s) on) so the daemon can do a
CPU build-smoke without training. The established pattern:

```python
#!/usr/bin/env python
"""Bootstrap for 157-conv-ffn."""
from configs.llm_config import Tiny1M3MConfig


class C(Tiny1M3MConfig):            # <-- top-level `C`, the build-smoke target
    use_conv_ffn: bool = True
    conv_ffn_kernel: int = 3


if __name__ == "__main__":
    import sys, train_llm
    sys.modules["__main__"].C = C
    sys.argv = [
        "train_llm.py", "--config_class", "__main__.C",
        "--seed", "42",
        "--dataset_path", "processed_data/pretrain_1B",
        "--warmup", "false",
    ]
    train_llm.main()
```

Why `_arq` and not raw CLI flags: `train_llm.py` argparse is a hand-maintained
allowlist, so **new idea flags are silently ignored on the CLI**. The subclass
stub is the only reliable way to toggle a new flag. (See `[[vast-runner-harness]]`.)

## What the daemon relies on (don't break these)

- **Build-smoke target.** The daemon loads `<arq_file>` as a module (so its
  `__main__` block does *not* run), reads `C`, and constructs `MinimalLLM(C())`
  on CPU. A flag added to the dataclass but not threaded through the model
  crashes here in seconds → the idea is bounced `needs-recode` before any GPU
  time is spent. **Keep the class named `C`.**
- **Final readout.** `train_llm.py` ends every run with these exact lines; the
  daemon greps them — do not change the wording:
  ```text
  Final Train Loss:                6.4004
  Final Val Loss:                  6.4316
  Final Val Accuracy:              0.1429
  ```
- **Control is the daemon's, not the idea's.** The baseline ctrl is always the
  bare tier config (`configs.llm_config.Tiny1M3MConfig`, seed 42, flag OFF). The
  daemon prepends ctrls itself **only** when [`baseline.sh check`](bin/baseline.sh)
  returns `MEASURE`. An idea never ships a ctrl.

## The daemon's loop (for reference — see the script for the truth)

1. **Reclaim** `running` ideas whose `updated` is stale and whose `arq` tmux is
   dead → back to `needs-run`.
2. **Claim** every `needs-run` idea with a valid `run.json` → `running`.
3. **Sync + smoke** on the box: `git pull` the repo, `scp` each `arq_file`, CPU
   build-smoke each `C`. Smoke fail → `flip <idea> needs-recode`.
4. **Baseline:** `baseline.sh check` → `CACHED` (treatment-only) or `MEASURE`
   (prepend N≥3 ctrls).
5. **Launch** one guarded, fail-isolated queue in a **detached `arq` tmux**.
6. **Poll** `STATUS` each tick: finalize every `OK`, bounce every `FAIL`.
7. **Judge** each finished treatment with `baseline.sh verdict` (`mean ± band`),
   write `evidence.md`, `flip … done`, append NULLs to `closed.md`.
8. **Cache:** `baseline.sh measure` (MEASURE path) or `bump` (CACHED path).

Idempotent and cron-safe: re-invoking never relaunches a live queue or
re-finalizes a done idea. No auto-push — local working tree only.
