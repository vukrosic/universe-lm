# Plan — 145-expert-choice

## Flag
- `configs/llm_config.py:831-832`:
  - `use_expert_choice_moe: bool = False`
  - `n_moe_experts: int = 4`
- `configs/llm_config.py:4905-4932`:
  - `Tiny1M3MExpertChoiceConfig(Tiny1M3MConfig)` with `use_expert_choice_moe: bool = True`
  - `n_moe_experts: int = 4`

## Change
- `models/expert_choice_moe.py`: add the expert-choice MoE wrapper with a zero-init router and `n_experts` full-width FFNs.
- `models/layers.py`: import `ExpertChoiceMoE`, add `use_expert_choice_moe` / `n_moe_experts` kwargs to `TransformerBlock`, and select it in the FFN branch after `use_switch_ffn`.
- `models/llm.py`: read the two flags from config and thread them into both block-construction sites.
- `configs/llm_config.py`: add the base flags on `LLMConfig` and the `Tiny1M3MExpertChoiceConfig` subclass with the lever on.
- `train_llm.py`: add `--use_expert_choice_moe` / `--n_moe_experts` to the CLI allowlist.
- `_arq_145-expert-choice.py`: seed-42 launcher stub with top-level `C`.
- `autoresearch/ideas/145-expert-choice/run.json`: daemon descriptor for the run handoff.

## Control
- Control: `Tiny1M3MConfig`, tier `tiny1m3m`, seed `42`.
- Treatment: `Tiny1M3MExpertChoiceConfig`, tier `tiny1m3m`, seed `42`, `use_expert_choice_moe=True`, `n_moe_experts=4`.

## Cost
- Params: `+1,182,720` at tiny1m3m, roughly `4x` the FFN budget.
- FLOPs: higher on the routed FFN path, worst-case about `4x` the FFN compute.
- Memory: one router plus `n_moe_experts` full-width expert FFNs and routed-token buffers.

## Run
- `python train_llm.py --config tiny1m3m --seed 42 --use_expert_choice_moe true --n_moe_experts 4`
- Tier: `tiny1m3m`
- Seed: `42`
- Expected wall-clock: one tiny1m3m job
- Pass/fail bar: PASS `<= ctrl - 0.01`; NULL band `|Δ| < 0.01`; DRIFT `> +0.01`.
