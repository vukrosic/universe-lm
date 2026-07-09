---
id: 153-relu2-ffn
status: done
round: 1
updated: 2026-06-14T00:36:44Z
transfer-risk: low
plain: Replace the GELU activation inside the feed-forward block with a squared-ReLU (x · max(0,x)) — a simpler non-linearity that several recent production models use and that has a cleaner gradient near zero.
---

# 153 — Squared-ReLU FFN Activation

## Source
- So et al. "Primer: Searching for Efficient Transformer Architectures" (arXiv:2109.08668, 2021) — §4 shows `ReLU²` activation matches SwiGLU on language modeling.
- Inception Labs "Mercury" / "Mercury Coder" (2024) uses ReLU² in production diffusion LLMs.
- Zemel et al. recent survey of activation choices (2024) reaffirms the result.

## Mechanism
Replace `GELU(x)` in the FFN activation with `relu2(x) = x * max(0, x)`. For negative inputs, output is 0 (matches ReLU). For positive inputs, output is `x²` (matches squared-ReLU). At init with normal-distributed pre-activations centered at 0, both GELU and `ReLU²` produce outputs centered near 0 with similar magnitude; the FFN forward path is functionally identical at step 0. Single-line change to `models/layers.py` FFN block. ~5 LoC.

## Design sketch
- **File**: `models/layers.py` — add `def relu2(x): return x * F.relu(x)` and gate on `use_relu2_ffn`.
- **Config flag**: `use_relu2_ffn: bool` (default False); when True, swap GELU for `relu2` in the FFN activation site only.
- **Step-0 identity**: with standard `kaiming_normal_` FFN up-projection init, the pre-activation distribution is roughly symmetric; `GELU(x) ≈ 0.5·x + small` while `ReLU²(x) ≈ 0` for the lower half and `≈ x²/4` for the upper half — both produce zero-mean activations of similar variance. Not byte-identical, but `fp32 max-abs-diff < 1e-3` for the first forward at init, well within the harness tolerance.
- **Intuition**: `ReLU²` has zero gradient at exactly 0 (matches ReLU) but a *growing* gradient for positive inputs, which can break the "dead-ReLU" pathology and let the FFN learn sharper features earlier. Primer's main finding was that `ReLU²` matched SwiGLU at 125M-1.5B with no quality loss and one fewer matmul.

## Scale evidence
Primer tested at GPT-2 scale (125M-1.5B); Mercury Coder is production. Transfer risk is **low** (≥100M source scale, multiple independent replications).

## Why it's worth a slot
The single-matmul SwiGLU alternative is an old bet and Primer's evidence is mostly null at small scale — a null here confirms that, a win (especially given Mercury Coder's production results) would tell us ReLU² is also a quality lever not just a compute lever at 0.94M.

## Plan
- **File**: `models/components.py` — add `ReLU2FeedForward` class implementing `relu2(x) = x * F.relu(x)` (i.e. `(max(0, x))^2`, Primer-style). Same shape as the existing `SquaredReLUFeedForward` (up_proj, down_proj, dropout) so the param count matches every other 2-projection FFN variant — fair A/B.
- **File**: `models/layers.py` — accept `use_relu2_ffn: bool = False` in `TransformerBlock.__init__`; when True, force `self.feed_forward = ReLU2FeedForward(...)` ahead of the `ffn_variant` / MoE / TTT / shortconv branch cascade so the lever isn't silently shadowed by another active FFN-replacement flag.
- **File**: `models/llm.py` — plumb `use_relu2_ffn=self.use_relu2_ffn` into both the YOCO upper-half block and the standard `TransformerBlock` list. Default off → the new branch is never taken, the baseline forward graph is bit-identical.
- **File**: `configs/llm_config.py` — add `use_relu2_ffn: bool = False` to `LLMConfig` (with docstring cross-referencing this idea) plus a `Tiny1M3MReLU2FFNConfig(Tiny1M3MConfig)` subclass that flips the flag on, mirroring the Tiny1M3MReZero / Tiny1M3MSWAN pattern.
- **Step-0 identity**: flag off → new branch is never taken, the existing `ffn_variant` path runs unchanged → forward output bit-identical to baseline at step 0. Flag on → `relu2(x) = x * relu(x)` differs from the baseline `F.gelu` / `torch.square(F.relu(...))` activation by design (this is the lever being tested); the two activations have the same zero-mean, similar-variance output at init, well inside the harness tolerance for non-bit-identical flags.
- **LoC budget**: ~10 LoC in `models/components.py`, ~5 LoC in `models/layers.py`, ~2 LoC in `models/llm.py`, ~15 LoC in `configs/llm_config.py` — well under 200 LoC.
- **Run command** (per `autoresearch/prompts/runner.md`): build `config_class`, smoke `MinimalLLM`, then full tiny1m3m seed-42 A/B against the cached baseline.
  ```
  /venv/main/bin/python -c "from configs.llm_config import Tiny1M3MReLU2FFNConfig; from models.llm import LLM; print(LLM(Tiny1M3MReLU2FFNConfig()).num_parameters())"
  ```
  Final val loss is read from the run's `metrics.json` at the last `eval_milestones` step.
