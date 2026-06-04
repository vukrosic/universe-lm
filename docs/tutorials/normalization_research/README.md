# Normalization as a small research project

This note is for the person who wants to add a new normalization idea without getting fooled by a lucky stack.

The short version:

- A normalization idea is not "good" just because it helps once.
- The same norm can look different in a richer stack and in a stripped baseline.
- If you want a result people can trust, test the norm in both settings.

## What "full stack" means

In this repo, the "full stack" is not the norm itself.
It is the larger recipe around it:

- `value embeddings`
- `q-gain`
- `SWA384`
- `RoPE250k`

The full-stack run asks:

> If we keep the whole recipe fixed, which normalization is best?

That is how we found that `pnorm1.5` looked strong.

But the clean-baseline run asked a different question:

> If we remove the helpers and leave a plain full-attention transformer, which normalization still wins?

That is where the answer shifted toward `LayerNorm` and `pnorm1.75`.

The lesson is not that one result was wrong.
The lesson is that a norm is part of a system, not a universal law.

## What the code is doing

Three files matter most:

- [train_llm.py](/Users/vukrosic/my-life/llm-research-kit-scaling/train_llm.py)
- [models/layers.py](/Users/vukrosic/my-life/llm-research-kit-scaling/models/layers.py)
- [configs/llm_config.py](/Users/vukrosic/my-life/llm-research-kit-scaling/configs/llm_config.py)

`train_llm.py` exposes the knobs:

```text
--norm_type
--qk_norm_type
--v_norm_type
--use_layernorm
```

`models/layers.py` turns those strings into actual modules.
The factory is simple on purpose:

```python
if nt.startswith("pnorm"):
    return PNorm(dim, p)
if nt.startswith("clipnorm"):
    return ClipNorm(dim, k)
if nt in _NORM_REGISTRY:
    return _NORM_REGISTRY[nt](dim)
if nt == "layernorm" or use_layernorm:
    return nn.LayerNorm(dim, elementwise_affine=True)
return nn.RMSNorm(dim)
```

That means the pattern for a new norm is clear:

1. Write a small `nn.Module`.
2. Register it in the factory.
3. Add a CLI string if you need one.
4. Run it against `RMSNorm`.

## What the data says

The current tiny1m evidence is already enough to teach something useful.

On the richer stack:

| norm | val_loss | read |
|---|---:|---|
| `pnorm1.5` | `6.3063` | best full-stack norm |
| `pnorm1` | `6.3156` | strong |
| `channelscale` | `6.3253` | strong |
| `RMSNorm` | `6.3584` | baseline |

On the clean baseline:

| norm / placement | val_loss | read |
|---|---:|---|
| `LayerNorm` | `6.3628` | best clean-baseline run |
| `pnorm1.75` | `6.4088` | best plain p-norm |
| `pnorm1.5` | `6.4387` | beats RMSNorm, but not best here |
| `RMSNorm` | `6.4516` | clean baseline |

So the useful idea is not "replace RMSNorm with one magic norm".
It is:

- Mild robustness helps.
- The exact best norm depends on the architecture around it.
- Attention-side placement can matter as much as the body norm.

## How to test a new norm properly

This is the workflow I would teach another engineer:

1. Start with the full stack.
2. Test one norm change at a time.
3. Compare against `RMSNorm`.
4. Re-run the same norm on a clean baseline.
5. If the result still looks good, repeat it on more than one seed.

That progression matters because it separates three questions:

- Does the norm help inside the richer recipe?
- Does it still help when the recipe is stripped down?
- Is it real signal or just seed noise?

The repo already has scripts that follow that pattern:

- `scripts/queue_tiny1m_norm4.sh`
- `scripts/queue_tiny1m_norm5.sh`

## A tiny example

Suppose you want to try a new norm called `robustcenter`.

The minimum path is:

```python
class RobustCenter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xc = x - x.mean(dim=-1, keepdim=True)
        denom = xc.abs().mean(dim=-1, keepdim=True) + 1e-6
        return self.weight * (xc / denom)
```

Then add it to the norm factory:

```python
_NORM_REGISTRY["robustcenter"] = RobustCenter
```

Then run it like a real research question, not a vibes question:

```bash
python train_llm.py --config tiny1m --norm_type robustcenter
```

If it wins once, test it again on the clean baseline.
If it still wins, run more seeds.
If it only wins in one setting, write that down honestly.

## What to teach readers

If this becomes an article for other people, the main teaching should be:

- `RMSNorm` is not sacred.
- `full stack` is a context, not a theorem.
- A norm can look excellent in one stack and merely decent in another.
- The best normalization work is not a single result. It is a controlled comparison across contexts.

That is the real lesson behind the ablations.

## What generalizes beyond norms

The norm result is specific, but the research habit is broader.
These are the parts I would reuse on other model changes:

- Test in a rich stack and a stripped baseline.
- Keep the true default in the table.
- Change one thing at a time.
- Use paired seeds before claiming a win.
- Prefer mild changes over aggressive ones.
- Record where the change is applied, not just what it is.

That workflow catches the difference between a real improvement and a trick that only works because of the surrounding architecture.

## What the paired-seed check taught us

The follow-up run on seeds 43 and 44 was useful because it made the result less
vibe-based:

| norm | seed 43 | seed 44 | read |
|---|---:|---:|---|
| layernorm | 6.3594 | 6.3644 | stable best |
| rmsnorm | 6.3931 | 6.3953 | baseline |
| pnorm1.75 | 6.4019 | 6.4013 | stable, but behind LayerNorm |
| pnorm1.5 | 6.3963 | 6.5822 | rerun needed before promotion |

The key research lesson is the same one we keep seeing in small-compute work:
if a change only looks good on one seed or one stack, it is not ready to teach
as a general principle.

The newest partial sweep on seed 45 adds a fresh candidate, `channelscale`,
which beat both `LayerNorm` and `RMSNorm` on that run. That is exciting, but it
still needs the second seed before it can graduate from "lead" to "result."

The more universal workflow is:

1. Test the idea in the richer stack.
2. Test it in the stripped baseline.
3. Re-run the winners on a second seed.
4. Only then write the takeaway.
