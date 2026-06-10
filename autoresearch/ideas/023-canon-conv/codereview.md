# Code-review log — 023 Canon conv

## r1 — 2026-06-10 — verdict: revise

Mechanism (module): faithful — `models/canon_conv.py:53-58` builds `nn.Conv1d(d_model, d_model, kernel_size=3, padding=0, groups=d_model, bias=False)` + scalar `nn.Parameter(torch.zeros(1))`. Forward `models/canon_conv.py:62-70` left-pads `(K-1, 0)` along the time axis and returns `x + gate * h.transpose(1, 2)`. Causality, step-0 identity, depthwise property all hold (6/6 tests pass: `pytest tests/test_canon_conv.py -v`).

Placement: correct in all 3 block branches — `models/layers.py:2325-2326` runs `x = self.canon_conv(x)` BEFORE the pre-norm `self.norm1(x)` call (pre-norm `:2378`, post-norm `:2353`, parallel-block `:2331`). Matches spec `idea.md:90-95`.

Flag-off path: bit-identical — `models/layers.py:2187-2189` only constructs `self.canon_conv` when `use_canon_conv=True`; `:2325-2326` is gated. No params, no FLOPs in baseline.

LoC: ~50 prod LoC (module 30 + wiring 8 + flag 8 + config class 22). Well under the 200 ceiling.

Coordination: surgical — adds new file `models/canon_conv.py`, one import at `:9`, one new kwarg at `:2032`, one wiring block at `:2187-2189`, one forward branch at `:2325-2326`, one flag at `configs/llm_config.py:205`, one config class at `:912-934`, one wiring line at `models/llm.py:248,368`. No stomp on the parallel 021/022/024/025 additions visible in the same diff.

### Findings (must fix before run)

- **F1 — silent multi-axis HP drift, trt vs ctrl** (`configs/llm_config.py:912`). The trt class is `class Tiny1M3MCanonOnFireConfig(Tiny1M3MConfig):` but the spec at `idea.md:57-67` pins the trt as "**same config** + `use_canon_conv=True`" where the ctrl is `Tiny1M3MVQGainSWAHighRoPE250KConfig + use_fire_pe=True`. The diff between `Tiny1M3MConfig` and `Tiny1M3MVQGainSWAHighRoPE250KConfig` is 4 booleans + 1 int that flip silently:

  ```
  axis                       trt (now)   ctrl
  use_value_embed            False       True
  use_q_gain                 False       True
  use_sliding_window         False       True
  rope_base                  10000       250000
  ```

  That is an A/B over 5 axes, not 1 — exactly the "silent HP drift smuggled in alongside" pattern the code-reviewer prompt flags as `revise`. The plan handwaves this as "the existing 020/021/022 pattern" (`plan.md:222-229`), but that precedent has been REVISED: `Tiny1M3MSoftpickOnFireConfig` at `configs/llm_config.py:856` was just changed to inherit from `Tiny1M3MVQGainSWAHighRoPE250KConfig` for exactly this reason. **Fix:** change the parent class at `configs/llm_config.py:912` to `Tiny1M3MVQGainSWAHighRoPE250KConfig` (drop the redundant `use_fire_pe: bool = True` — it's already there via the new parent's `use_fire_pe` inherits from `LLMConfig`, so set it explicitly on the trt class only). Final class body:

  ```python
  class Tiny1M3MCanonOnFireConfig(Tiny1M3MVQGainSWAHighRoPE250KConfig):
      use_fire_pe: bool = True
      use_canon_conv: bool = True
  ```

- **F2 — runner harness scripts missing** (`_arq_023.py` and `_arq_023_ctrl.py`). Plan §Run at `plan.md:198-208` pins the runner contract: both scripts must exist before `needs-run` is meaningful (the runner reads them, not the config class). Neither file is on disk (`ls _arq_023*.py` → no matches). **Fix:** create the two scripts, mirroring `_arq_020.py` / `_arq_020_ctrl.py` exactly:

  ```python
  # _arq_023.py — trt
  import sys
  from configs.llm_config import Tiny1M3MCanonOnFireConfig
  class C(Tiny1M3MCanonOnFireConfig): pass
  if __name__ == "__main__":
      import train_llm
      sys.modules["__main__"].C = C
      sys.argv = ["train_llm.py", "--config_class", "__main__.C",
                  "--seed", "42", "--dataset_path", "processed_data/pretrain_1B",
                  "--warmup", "false"]
      train_llm.main()
  ```

  ```python
  # _arq_023_ctrl.py — ctrl (FIRE-equipped 009 WIN signature)
  import sys
  from configs.llm_config import Tiny1M3MVQGainSWAHighRoPE250KConfig
  class C(Tiny1M3MVQGainSWAHighRoPE250KConfig):
      use_fire_pe: bool = True
  if __name__ == "__main__":
      import train_llm
      sys.modules["__main__"].C = C
      sys.argv = ["train_llm.py", "--config_class", "__main__.C",
                  "--seed", "42", "--dataset_path", "processed_data/pretrain_1B",
                  "--warmup", "false"]
      train_llm.main()
  ```

After F1+F2 land, this is `accept` — the mechanism, placement, identity-init, flag wiring, and tests are all clean.
