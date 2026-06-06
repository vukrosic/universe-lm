# Research goal list — loci → experiments → tutorials

The working method: pick **one point** in the base transformer, design many cheap
identity/zero-init levers around it, screen tiny→`Screen10M20M` (3-seed), promote
winners to the full ladder, then write the tutorial. Each topic below is one folder:
`plan.md` (for the implementing AI) + `tutorial/` (filled after runs).

Control everywhere = clean `Screen10M20MConfig`, val_loss **4.7984** (`s_ctrl_full`).

---

## Status legend
`tutorial` = published · `folder` = plan+scaffold built, runs pending · `idea-bank` =
loose plan exists, no folder · `gap` = experiments already run, no writeup ·
`candidate` = not started.

---

## 1. Shipped tutorials (done)

| Topic | Path |
|---|---|
| Embedding factorization + depth | `../tutorials/embedding_factorization_depth/` |
| QK gain | `../tutorials/qk_gain/` |
| QKV embeddings | `../tutorials/qkv_embeddings/` |
| Value embeddings | `../tutorials/value_embeddings/` |
| Normalization | `../tutorials/normalization/` |

## 2. Research folders built (plan ready)

`code` = all levers wired in `models/layers.py` + configs in `llm_config.py`, runs
pending · `scaffold` = plan + tutorial scaffold only, code-wiring unverified.

| Locus | Folder | Levers | Code | Flagship |
|---|---|---|---|---|
| Query (pre-dot-product) | [query/](query/) | 29 | **code** (Q1–Q29 wired, 0 runs) | talking-heads, ALiBi bias |
| Residual add | [residual_stream/](residual_stream/) | 10 | scaffold | ReZero, ResidMix |
| Attention logits (pre-softmax) | [attention_logits/](attention_logits/) | 10 | code (configs exist) | z of softcap/temp/entropy |
| Output head + loss | [output_head/](output_head/) | 8 | code (configs exist) | z-loss, vocab bias |
| RMSNorm op | [rmsnorm/](rmsnorm/) | 17 + zoo sweep | scaffold | partial-norm mix, RMS+bias, DynTanh |
| Muon optimizer | [muon/](muon/) | 14 | scaffold | no-ortho A/B, ns_steps speed cut |
| U-Net skips | [unet_skips/](unet_skips/) | gate/init + skip-count | scaffold | sigmoid gate init vs raw zero gate |

> Note: **code-wired ≠ run.** Every `results.md` in §2 is still `pending` — the levers
> are implemented and launchable but no val_loss numbers exist yet.

## 3. Idea banks ready to promote to folders (next up)

| Locus | Source | Why |
|---|---|---|
| Tied output MLP vs embed residual | `../research-plans/tied-output-mlp/plan.md` | A-vs-B inductive bias question, ties into output_head OH8 |

## 4. Gaps — experiments already run, tutorial unwritten

| Topic | Evidence (runs/) | Story |
|---|---|---|
| **RoPE base tuning** | rope125k/250k/375k/500k/750k sweep | clean curve, optimum *scales with model size* |
| **Sliding-window attention** | `*_swa256/384/512/768/1024`, `s_swa_only` | window sweep, folder `swa_record/` exists but unwritten |
| **LR warmup-decay schedule** | `issue30` warmup_decay | was the #1 10m record; folder `lr_schedules/` unwritten |
| KV-head sharing (MHA/GQA/MLA) | `s_mha/gqa1`, `*_mha/gqa1/mla` | wash at scale; GQA1+MLA hurt — good negative result |
| Tied QK (PaLM) | `tiny1m_arch_tiedqk` vs `*_tiedqk` | wins at tiny, fades at screen — scale-dependent |
| Pre vs post-norm | `*_postnorm`, `qkpostnorm` | post-norm collapses — fold into normalization or standalone |

## 5. Candidate loci — not started (the backlog)

Single points, each rich enough for ~8–20 cheap levers.

| # | Locus | The one point | Sample cheap levers | Novelty |
|---|---|---|---|---|
| C1 | **Attention output proj `W_O`** | per-head outputs → mixed back into residual | per-head output gate, talking-heads-on-output (post-softmax mix), output LayerScale | high |
| C2 | **Input embedding `x0`** | token vector before block 0 | embedding scale, embedding norm, input dropout, scaled-init | med |
| C3 | **Keys `K`** | pre-dot-product on K | mirror of query batch (k_gain exists) | low |
| C4 | **Value aggregation `V`** | the `softmax·V` weighted sum | per-head value gate, value norm, value temperature | med |
| C5 | **Positional / RoPE internals** | the rotation applied to Q,K | partial rotary, per-head base, NoPE-mix, learned freqs | med (some in query/) |
| C6 | **KV-head structure** | how many K/V heads, how shared | GQA ratio sweep, per-group gates, learned head merging | med (gap exists) |
| C7 | **Depth / layer schedule** | which block does what | layer-wise width, early-exit, block reordering, per-depth LR | med |
| C8 | **Init scheme** | weight init at t=0 | per-matrix std sweep, zero-init projections, μP-style scaling | med |
| C9 | **Optimizer routing** | which params → Muon vs AdamW | routing sweep, per-group LR, Muon for 1D params | high (repo-specific) |
| C10 | **Attention temperature / scaling** | the `1/√d` factor | learnable global scale, per-layer schedule, length-aware scaling | med (overlaps attention_logits) |
| C11 | **Token mixing alternatives** | replace/augment attention | short conv mix, MLP-mixer block, hybrid conv-attn | high lift |
| C12 | **Embedding tying geometry** | the shared `E` / `emb_proj` | rank sweep, separate in/out rank, orthogonal init | med (factor tutorial adjacent) |
| C13 | **Data / sequence packing** | how tokens enter | seq-len sweep, doc packing, curriculum | med (non-arch) |
| C14 | **Regularization** | what's penalized | weight decay sweep, dropout placement, stochastic depth | low-med |

---

## Build order recommendation

1. **Promote the two ready idea banks** (RMSNorm, tied-output-mlp) — least work, plans exist.
2. **Close the three gaps** with the strongest stories: RoPE base, SWA, LR schedule —
   experiments are *already run*, just need the writeup (claim the half-built folders).
3. **New folders**: C1 (`W_O`) and C4 (`V` aggregation) — highest novelty, same cheap pattern.

Everything else is backlog. Pull from §5 when §3–4 are cleared.
