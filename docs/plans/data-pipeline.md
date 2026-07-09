# Data pipeline — Phase 3 of docs/plans/beat-smollm2-135m.md

Status: v1 (2026-06-10). Owner: Phase 3.
Hard ceiling: 4–5 TB on the Vast box. Must fit pre-tokenized 2T tokens.

## 1. Datasets — HF IDs, sizes, tokens

| Dataset | HF ID (config) | Rows | Disk (GB) | Tokens | Source / cite |
|---|---|---|---|---|---|
| FineWeb-Edu (full) | `HuggingFaceFW/fineweb-edu` (`default`) | 1.53 B | ~1.4 TB | ~1.3–1.5 T | [HF card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) |
| FineWeb-Edu sample-10BT | `HuggingFaceFW/fineweb-edu` (`sample-10BT`) | 9.67 M | 28.5 | 10 B | [HF dir](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/tree/main/sample/10BT) |
| FineWeb-Edu sample-100BT | `HuggingFaceFW/fineweb-edu` (`sample-100BT`) | 97.3 M | 286 | 100 B | [HF dir](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/tree/main/sample/100BT) |
| FineWeb-Edu sample-350BT | `HuggingFaceFW/fineweb-edu` (`sample-350BT`) | 339 M | ~1.0 TB | 350 B | HF card |
| DCLM-baseline-1.0 | `mlfoundations/dclm-baseline-1.0` (`default`) | ~2.73 B | 7.42 TB | 4.0 T | [HF card](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) (mirror: `…-parquet`) |
| Stack-Edu (code) | `HuggingFaceTB/stack-edu` | 167 M | 17.5 (SWHIDs) + ~150 (content) | 125 B | [HF card](https://huggingface.co/datasets/HuggingFaceTB/stack-edu) |
| SmolLM-Corpus (FWE-dedup) | `HuggingFaceTB/smollm-corpus` (`fineweb-edu-dedup`) | 190 M | ~600 | 220 B | [HF card](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) |
| SmolLM-Corpus (cosmopedia-v2) | `HuggingFaceTB/smollm-corpus` (`cosmopedia-v2`) | 39.1 M | ~60 | 30 B | HF card |
| SmolLM-Corpus (python-edu) | `HuggingFaceTB/smollm-corpus` (`python-edu`) | 7.68 M | (S3 fetch) | ~5 B | HF card |

**Tokenizer:** `HuggingFaceTB/cosmo2-tokenizer`, **vocab = 49,152**, trained on 70% FineWeb-Edu / 15% Cosmopedia-v2 / 8% OpenWebMath / 5% StarCoderData / 2% StackOverFlow. Reuse as-is (no retrain) per Phase-0 plan. [Tokenizer card](https://huggingface.co/HuggingFaceTB/cosmo2-tokenizer), [SmolLM2 paper §4.1](https://arxiv.org/abs/2502.02737).

**Gap:** DCLM has no public sub-sample; we cannot get <1 T of DCLM without filtering. For Phase-1/2 proxies we either (a) downsample to 1-of-N rows or (b) live with ≥1 TB of DCLM held in storage.

## 2. What SmolLM2-135M actually used (from the paper)

Paper: *"SmolLM2: When Smol Goes Big — Data-Centric Training of a Small Language Model"*, [arXiv:2502.02737](https://arxiv.org/abs/2502.02737), §6.

- **Architecture:** Llama2-style, GQA; 30L / 576h / 9 q-heads / 3 kv-heads; context 2k (base), 8k after the context-extension step; 2T tokens; WSD scheduler, **20% decay**, **peak LR 3.0e-3**; bfloat16, nanotron. (See `docs/plans/beat-smollm2-135m.md` for our pinned target spec.)
- **Single-stage training** for 135M / 360M (the 1.7B used the 4-stage curriculum in §4). The 135M mix:
  - **English web:** DCLM re-filtered through the FineWeb-Edu classifier — drop `int_score=0`, downsample `1` and `2`.
  - **Code:** Stack-Edu, from the start (not introduced at Stage 3 like the 1.7B).
  - **Math:** InfiMM-WebMath + FineMath, from the start.
  - **Synthetic:** Cosmopedia (v2), from the start.
- **SmolTalk SFT** (filtered version) then **DPO with UltraFeedback** post-training.
- **What the paper does NOT publish:** exact per-component percentages for the 135M. §6 says they "re-ran data ablations at the target training length" but the table is in the appendix, not in the main text. **Honest position: we set the 135M-specific ratios via our own Phase-1/2 ablations, not by copying the paper.**

For reference, the 1.7B four-stage ratios (§4.2–§4.5, Fig. 2): Stage 1 (0–6T) = ~54% FWE + 36% DCLM + 10% StarCoderData; Stage 2 (6–8T) = 75% web (60/40 FWE/DCLM) + 20% code + 5% OWM; Stage 3 (8–10T) = ~10% math, FWE/DCLM flipped to 40/60, StarCoderData→Stack-Edu; Stage 4 (10–11T, decay) = 58% web (40/60 FWE/DCLM) + 24% code (Stack-Edu expanded) + 14% math (OWM 0.08% + InfiMM-WebMath-3+ + FineMath-3+ + AugGSM8K 0.02%) + 4% Cosmopedia v2.

## 3. Tokenize-and-shard plan

**Tokenizer:** `HuggingFaceTB/cosmo2-tokenizer`, loaded via `transformers.AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer", use_fast=True)`. Use_fast=True gives Rust BPE at ~1 M tok/s/core on a modern CPU.

**Shard format:**
- `uint16` token IDs (2 bytes/token) — needs vocab < 65,536 (cosmo2 = 49,152, fits).
- Append `<eos>` (id 2 in cosmo2) at end of every doc.
- Pack to fixed sequence length **2048** with cross-doc boundaries masked by `-100` on the label axis (HF `DataCollatorForLanguageModeling` with `mlm=False`).
- **Shard size: 256 M tokens → 512 MB raw → round to 1 GB per shard** (256 M tokens + 16 MB index header = ~528 MB; pad to 1 GB with `pad_to_multiple_of=8M` for fast aligned reads). Easier mental math than nanoGPT's 100M.
- File: `shard_{rank:05d}.bin` (raw uint16, little-endian, no compression — mmap-friendly) + `shard_{rank:05d}.idx.json` (offsets, doc-count, sha256 of `.bin`).
- Layout per chunk dir:
  ```
  data/
    v1_fwe2_dclm_stackedu/
      manifest.json           # sources, ratios, total tokens, tokenizer hash, git sha
      train/
        shard_00000.bin shard_00000.idx.json
        ...  (~7800 shards for 2T tokens)
      val/  (~50M tokens held out)
  ```
- **Checksums:** `manifest.json` records SHA-256 of every `.bin` and SHA-256 of the tokenizer files. Tokenizer-pinned is mandatory — any retrain of cosmo2 invalidates the manifest.

**Disk math for 2T tokens:**
- Raw: 2 × 10¹² × 2 B = **4.0 TB**.
- +1% for EOS + indices + `val/`: ~4.05 TB.
- **Comfortably under the 4–5 TB ceiling.** Add ~5% headroom for re-tokenization attempts and we're at 4.3 TB.

**Streaming loader (train side):**
- mmap a random `.bin`, seek to a random 2k-token aligned offset, return the next 2048 tokens.
- Reshuffle on epoch boundary by randomising shard order + within-shard offset.
- Val loader: sequential, deterministic.

**Tokenization wall-clock — where to run it:**
- Pure-Rust BPE: ~0.5–1.0 M tok/s/core on a modern CPU. [HF fast-tokenizer benchmarks](https://github.com/huggingface/tokenizers) put GPT-2 BPE at ~1.5 M tok/s/core on c6i.2xlarge.
- 2 × 10¹² tok ÷ (32 cores × 0.75 M tok/s) ≈ **23 hours** theoretical.
- Bound in practice by (a) parquet decode + HF datasets overhead → expect **1.5–3× slowdown**, (b) disk I/O → needs NVMe scratch.
- **Recommended:** spin a **Vast CPU-only instance** (e.g. 32 vCPU / 64 GB RAM / 1 TB NVMe, ~$0.10/hr) for ~2–3 days. Avoid the GPU box — we'd be paying $2/hr for cores we'd be wasting on HTTP/SSL.
- Local-Mac alternative: do not. Apple Silicon tokenization is fast but the local disk is too small to stage 4 TB; download + tokenize on Vast, ship `.bin` shards back to object storage or keep on the GPU box's volume (one-time ~$5–10 egress).

**Phased tokenization to derisk:**
1. Tokenize Phase-1/2 proxy set first (see §4) — should take <30 min.
2. Smoke-test the loader on it: 1 epoch of a 10M-param model, BPB moves sensibly.
3. Only then scale to the 2 T job, in batches of 100 B tokens. Each batch is a 1.5 h job; we can stop after any batch.

## 4. Staged download plan — Phase 1/2 proxies (30–50 B tokens)

Goal: recipe work at 10M–135M starts within 1–2 days, no 2 T commit.

| Tier | Subset | Tokens | Disk | Notes |
|---|---|---|---|---|
| Web-A | FineWeb-Edu `sample-10BT` | 10 B | 28.5 GB | one-shot, mirror or `huggingface-cli download` |
| Web-B | DCLM filtered, **first ~60 GB of parquet** (≈ 30 B tokens at ~2 GB/B tokens) | 30 B | 60 GB | partial mirror; index by `mlfoundations/dclm-baseline-1.0-parquet` |
| Code | Stack-Edu SWHIDs + Python subset content (S3 Software Heritage) | 5 B | 17.5 + ~6 GB | `HuggingFaceTB/stack-edu` "Python" subset |
| Math | FineMath-3+ (`HuggingFaceTB/fine-math`) + InfiMM-WebMath-3+ (`opendataaimath/InfiMM-WebMath-3Plus`) | 5 B | ~10 GB | direct from HF |
| Synthetic | Cosmopedia v2 (`HuggingFaceTB/cosmopedia`, 100k sample) | 0.1 B | ~1 GB | cheap |
| **Total** | | **~50 B** | **~120 GB** | fits in any 256 GB Vast volume |

Phase-1/2 runs (10M × 5B, 30M × 10B, 60M × 15B, 135M × 20B) all live inside this 50 B. Use BPB on a held-out FineWeb-Edu slice as the decision metric (already pinned in `beat-smollm2-135m.md`).

Only after the recipe is locked do we trigger the **2 T download**:
- 1.4 TB FineWeb-Edu full → ~$0.50 ingress (HF is free egress for these public datasets via `huggingface-cli`).
- 7.4 TB DCLM → too big; **do NOT mirror the full DCLM.** We only need enough to fill the 2 T mix. If 135M web ratio is 60% (1.2 T tokens) split ~50/50 FWE/DCLM, that's 600 B DCLM tokens ≈ 1.1 TB. **Plan to mirror ~1.1 TB of DCLM, not 7.4 TB.** (DCLM is `mlfoundations/dclm-baseline-1.0`; use the `-parquet` mirror for direct file access.)
- 125 B Stack-Edu → 17.5 GB SWHIDs + ~150 GB content.
- FineMath + InfiMM-WebMath + Cosmopedia → small (~100 GB total).
- **Net: ~1.4 TB (FWE) + 1.1 TB (DCLM) + 170 GB (code) + 100 GB (math/synth) = ~2.8 TB of raw corpus → 4.05 TB of uint16 shards.** Fits.

## 5. Cost table (USD, all approximate; verify in Phase 0)

| Item | Unit | Quantity | Unit cost | Subtotal | Source |
|---|---|---|---|---|---|
| HF dataset download (egress) | GB | 2,800 | $0 | $0 | [HF Hub pricing](https://huggingface.co/docs/hub/storage) — public datasets free egress |
| Tokenization compute (Vast CPU, 32 vCPU) | hour | ~70 (2.5 days × 1 worker) | ~$0.10/hr | **$7** | [Vast billing](https://docs.vast.ai/documentation/reference/billing) — host-set $/hr; community rate for 32-vCPU is $0.05–0.15/hr |
| Tokenization compute (alt: 64 vCPU) | hour | ~40 | ~$0.20/hr | **$8** | same |
| Tokenization compute (alt: local Mac M-series) | hour | ~120 | $0 (electric ~$0.50) | **< $1** | not recommended; disk-bound |
| Storage on Vast instance (4 TB) | GB-month | 4,000 × 2 months (recipe + flagship) | $0.002/GB/hr (≈ $1.45/TB/mo) | **$12** | Vast docs: storage is $/GB/hr, host-set, "considerable variation"; community range $0.001–0.003/GB/hr |
| Egress: ship shards to GPU box | TB | 4 | $10–20/TB | **$40–80** | Vast docs: bandwidth is $/TB, host-set |
| Egress: HF release of weights/logs | GB | ~10 | $0 | $0 | [HF Hub pricing](https://huggingface.co/docs/hub/storage) |
| **Phase 3 total (not counting GPU)** | | | | **≈ $60–100** | matches `beat-smollm2-135m.md` Phase-3 budget of "~$100 + 4–5 TB storage" |

**Caveats:**
- Vast storage and bandwidth are **set per-offer by hosts**, not by Vast; the billing reference doc states rates "vary considerably from machine to machine." The $/GB/hr and $/TB above are community-observed midpoints, not published list prices. [Source](https://docs.vast.ai/documentation/reference/billing).
- The flagship 8×H100 GPU cost is in `beat-smollm2-135m.md` Phase 4, separate from this plan.
- The plan's 4–5 TB ceiling assumes only the pre-tokenized corpus + working scratch. Add ≥1 TB free for the HF cache, OS, and conda envs.

## 6. Open decisions to resolve in Phase 0

1. **DCLM row count to mirror.** Will be set by Phase-1 winner ratio; current working assumption is 1.1 TB.
2. **DCLM subset strategy** — do we use the raw 4 T pool, or filter with the FineWeb-Edu classifier at tokenize time (matching the SmolLM2-135M recipe, §6)? Recommend: filter at tokenize time, save 70% of the DCLM disk.
3. **Code content licensing.** Stack-Edu file content lives in Software Heritage's S3 — verify download is compatible with our HF release (Apache-2.0 for our outputs).
4. **Shuffle strategy.** Decide: random shard order + random offset (nanoGPT-style) vs deterministically chunked for resume. Default: random.
5. **Tokenizer hash pin.** Lock the cosmo2 tokenizer at the byte level, not just the HF repo tag — repo tags can be re-pointed.

## 7. Cited URLs

- SmolLM2 paper: https://arxiv.org/abs/2502.02737
- FineWeb-Edu: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- FineWeb-Edu sample-10BT dir: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/tree/main/sample/10BT
- FineWeb-Edu sample-100BT dir: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/tree/main/sample/100BT
- DCLM-baseline-1.0: https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0
- DCLM parquet mirror: https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0-parquet
- Stack-Edu: https://huggingface.co/datasets/HuggingFaceTB/stack-edu
- SmolLM-Corpus: https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus
- cosmo2 tokenizer: https://huggingface.co/HuggingFaceTB/cosmo2-tokenizer
- HF Hub storage pricing: https://huggingface.co/docs/hub/storage
- Vast billing reference: https://docs.vast.ai/documentation/reference/billing
- HF fast-tokenizers (perf reference): https://github.com/huggingface/tokenizers
