# Plan — 107 exclusive-self-attn

## Flag
- `use_exclusive_self_attn: bool = False` on `LLMConfig` in `configs/llm_config.py`.
- Threaded through `models/llm.py -> TransformerBlock -> MultiHeadAttention`.

## Change
- `models/layers.py`
  - `MultiHeadAttention.__init__`: add `use_exclusive_self_attn`; create a zero-init
    per-head coefficient `self.exclusive_self_attn = nn.Parameter(torch.zeros(n_heads))`
    only when the flag is on.
  - `MultiHeadAttention.forward`: after the attention output is formed and before the
    output-side gates / merge, subtract the projection of `attn_output` onto the current
    token's value vector `V`:
    `o <- o - a_h * ((o·V)/(||V||^2 + eps)) * V`.
    `a_h` starts at 0, so step 0 is baseline.
  - `TransformerBlock.__init__`: accept and pass the new flag to MHA.
- `models/llm.py`
  - read `self.use_exclusive_self_attn = getattr(config, "use_exclusive_self_attn", False)`.
  - forward it into every `TransformerBlock(...)` construction.
- `configs/llm_config.py`
  - add `Tiny1M3MExclusiveSelfAttnOnFireConfig(Tiny1M3MConfig)` with
    `use_fire_pe = True` and `use_exclusive_self_attn = True`.
- `configs/__init__.py`
  - export `Tiny1M3MExclusiveSelfAttnOnFireConfig`.

## Step-0 / identity
- Default path stays byte-identical because the new field defaults to `False` and the
  model only allocates the extra parameter behind that flag.
- Flag-on step 0 is also baseline because the coefficient is zero-init, so the extra term
  is multiplied by 0 before it can change the output.

## Run
- Control: `Tiny1M3MConfig` with `use_fire_pe = True`.
- Treatment: `Tiny1M3MExclusiveSelfAttnOnFireConfig`.
- Seed: `42`.
- Command:
  `python train_llm.py --config_class configs.llm_config.Tiny1M3MExclusiveSelfAttnOnFireConfig --seed 42 --dataset_path processed_data/pretrain_1B --warmup false`
- Final val loss source: `training/trainer.py` writes `metrics.json`; read
  `final_metrics.val_loss` there (the same value is also printed as `Final Val Loss`).
