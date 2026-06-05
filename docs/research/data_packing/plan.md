# Data / sequence-packing ablations — research plan

**For the implementing AI.** Self-contained. The one point we're poking is
how tokens are batched into the model — sequence length, document packing,
and curriculum. Promotes + expands the C13 backlog entry in
[../../research/README.md §5](../../research/README.md#5-candidate-loci--not-started-the-backlog).

---

## The one point we're poking

```text
data/loader.py:  tokenize_and_chunk(...)  ->  group_texts(examples)
                 arrays = {k: np.concatenate(examples[k]) for k in examples.keys()}
                 trunc  = (total // block_size) * block_size
                 return {k: v[:trunc].reshape(n, block_size) for k, v in arrays.items()}
```

`group_texts` already concatenates documents and chunks at `seq_length` — i.e. the
baseline is *implicitly* doc-packed. Every lever here changes ONE batching decision
above that line: what to track across the doc boundary, how long the chunks are,
or what curriculum to apply.

## Critical wiring note

The data loader feeds the model; changes here affect every batch but not the model
itself. **Clean A/B territory** — no architecture changes, no optimizer routing
to worry about, no extra parameters. The interesting questions are about the
*information* exposed to the model (cross-doc boundaries) and the *distribution*
of what the model sees (length, curriculum).

A second wiring note: the existing `group_texts` already concatenates documents.
So the "doc pack" lever (D1) is not a switch from truncate → pack — it is a
switch from *implicit* pack (no boundary info) to *explicit* pack (doc_id
column). See "Batch 1" notes for the honest A/B.

## Implementation contract

- Edit [data/loader.py](../../../data/loader.py) — add one conditional branch in
  `group_texts` (read `config.use_doc_pack`). Default = `False` = byte-identical
  to baseline.
- Add `use_doc_pack: bool = False` to [configs/dataset_config.py](../../../configs/dataset_config.py)
  `DataConfig` (with `__post_init__` validation, type-checked).
- One `class Tiny1M3M<Name>Config` per lever (this implementation: `Tiny1M3MDocPackConfig`).
- Run: `python train_llm.py --config <name> --seed 42`.
- **Identity:** with `use_doc_pack=False` (the default) the loader output is
  byte-identical to baseline. The lever only fires when True.

---

## Batches (2 batches, 4 levers total)

### Batch 1 — packing (D1 / D2)

| # | Name | Change | Step-0 == base | Note |
|---|---|---|---|---|
| D1 | `DocPack` | emit a `doc_id` column marking which document each token came from (the baseline *already* concatenates documents; this exposes the boundary) | yes (flag off by default → no doc_id column produced) | Free info. The A/B: with the boundary info, downstream code (curriculum, attention masking) can reason about cross-doc transitions. |
| D2 | `NoCrossDoc` | same as D1, plus a no-attend token (or a doc_id-aware attention mask) between documents | yes (loader-level change; the actual mask would be a follow-up wiring) | Tests whether D1's value is the boundary info or the cross-doc signal itself. |

### Batch 2 — sequence length / curriculum (D3 / D4)

| # | Name | Change | Step-0 == base | Note |
|---|---|---|---|---|
| D3 | `SeqLenSweep` | sweep `seq_length` ∈ {256, 512, 1024, 2048} | yes (truncation already in baseline) | Basic knob. Note: changing `seq_length` interacts with `max_seq_len` and the RoPE cache — flag with care. |
| D4 | `ShortSeqFirst` | curriculum: start at `seq_length=128`, grow to full over first 10% of steps | yes (loader-level change) | Easier samples first. Implementation: dynamic `seq_length` in the trainer loop, not the loader. |

## Protocol (what counts)

- Control = clean `Screen10M20MConfig` (val_loss **4.7984**, `s_ctrl_full`).
- This is a **loader-level** locus, not an architecture locus. The control is
  the same as for the architecture loci — the lever changes *what the model
  sees per batch*, not the model itself.
- tiny → screen 3-seed (42/43/44) if the lever looks alive at tiny.
- The data axis is **lower-variance** than norm/architecture, but still
  3-seed-mandatory on the screen tier. Single-seed wins at tiny are the
  filter; do not promote from a single seed.

---

## Run guidance

- **D1 first** — cheapest, most concrete. It only changes one column in the
  dataset; the trainer's input pipeline doesn't even need to consume it for
  the lever to "fire" (the data is in the dataset, the model can use it
  later). Run D1 tiny → screen to establish the data-axis baseline.
- **D2 isolates the cross-doc question.** If D1 doesn't move and D2 doesn't
  move, the data axis is closed at this scale. If D1 moves and D2 doesn't,
  the win is the boundary info, not the masking. If D2 moves and D1 doesn't,
  the win is the masking and the boundary is incidental.
- **D3–D4 are sweeps.** They cost more runs and overlap with
  `max_seq_len`/`rope_base` concerns. Run after D1/D2 settle, only if a
  winner is found.

## When a batch finishes

1. Numbers → [tutorial/results.md](tutorial/results.md) — 3-seed mean + std
   (mandatory on the screen tier).
2. Status → [tutorial/experiments.md](tutorial/experiments.md).
3. Clear story → draft [tutorial/README.md](tutorial/README.md), house style
   of the other [../](../) tutorial stubs (terse, one mechanism, one
   baseline, one result).
4. Commit `metrics.json`, re-run `runs/make_evidence_index.py`.
