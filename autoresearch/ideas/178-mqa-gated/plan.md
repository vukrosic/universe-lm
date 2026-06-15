# Plan — 178 — Gated MQA Probe (Per-Head Mix Between Head-Local and Shared K/V)

## Flag
- `use_mqa_gated: bool` (default `False`) on `MultiHeadAttention.__init__` and threaded through `TransformerBlock` / `MinimalLLM`.
  - Field: `models/layers.py:1232` (MHA kwarg), `models/llm.py:313` (block reads from config).
  - A/B subclass: `Tiny1M3MMQAGatedConfig(Tiny1M3MConfig)` with `use_mqa_gated: bool = True` at `configs/llm_config.py:6182` (current diff).

## Change
- `models/layers.py` — `MultiHeadAttention.__init__` (lines ~2077-2149):
  - `self.use_mqa_gated = use_mqa_gated`.
  - When on, allocate:
    - `self.W_K_shared = nn.Parameter(zeros(shared_kv_dim, d_model))` where `shared_kv_dim = n_kv_heads * d_k`.
    - `self.W_V_shared = nn.Parameter(zeros(shared_kv_dim, d_model))`.
    - `self.mqa_gate_k = nn.Parameter(zeros(n_kv_heads))` — per-KV-head scalar.
    - `self.mqa_gate_v = nn.Parameter(zeros(n_kv_heads))` — per-KV-head scalar.
  - All zero-init. The `nn.Parameter(torch.zeros(...))` choice (NOT `nn.Linear`) keeps the construction from consuming RNG, preserving the `qkvo_proj` random init alignment with the no-flag baseline (any RNG advance would shift later blocks' qkvo_proj init and break step-0 byte-identity).
  - When off: stub the four attrs to `None` so attribute lookups are always valid; `forward()` never references them.
- `models/layers.py` — `MultiHeadAttention.forward` (lines ~2781-2806):
  - After the head-local K, V reshape (`[B, T, n_kv_heads, d_k]`), gated branch:
    ```python
    if self.use_mqa_gated:
        K_shared = F.linear(x, self.W_K_shared).reshape(B, T, n_kv_heads, d_k)
        V_shared_n = F.linear(x, self.W_V_shared).reshape(B, T, n_kv_heads, d_k)
        beta_k = self.mqa_gate_k.view(1, 1, n_kv_heads, 1)
        beta_v = self.mqa_gate_v.view(1, 1, n_kv_heads, 1)
        K = K + beta_k * (K_shared - K)
        # Mega-aware V mix: only the first n_kv_heads slots are blended.
        V = V + beta_v * (V_shared_n - V)  # when V_n_kv_heads == n_kv_heads
    ```
  - **Step-0 byte-identity (verified)**: β=0 init ⇒ `K = K + 0·(K_shared − K) = K` exactly in fp32; W_K_shared is also zero-init so `K_shared = F.linear(x, 0) = 0` regardless of input. **CPU build smoke confirms `max-abs-diff = 0.0` between flag-on and flag-off forward at the same seed.**
- `models/layers.py` — `TransformerBlock` (line 4016) and pass-through at line 4493: `use_mqa_gated=use_mqa_gated` to the inner MHA.
- `models/llm.py` — `MinimalLLM.__init__` (line 313): `self.use_mqa_gated = getattr(config, "use_mqa_gated", False)`. Pass-throughs at lines 698 (YOCO upper-half) and 994 (standard block) thread the flag to every block.
- `configs/llm_config.py` — `Tiny1M3MMQAGatedConfig` (current diff) flips the flag on.

## Control
- Treatment: `configs.llm_config.Tiny1M3MMQAGatedConfig` (flag on), seed 42, tier tiny1m3m, dataset `processed_data/pretrain_1B`, warmup off.
- Control (owned by the daemon): `configs.llm_config.Tiny1M3MConfig` (flag off), seed 42, same tier / dataset / warmup.
- Verdict path: `autoresearch/bin/baseline.sh verdict` (mean ± band) on `Final Val Loss`. The cached tiny1m3m baseline at box key `5b8a7fea8963`: val_mean = 6.3988, val_std = 0.0088, noise_band = 0.04 → WIN iff `Final Val Loss < 6.3588`. **Val-loss is informational only** (review.md finding #4); the primary signal is the per-KV-head β trajectory (see "Runner hooks" below).

## Cost
- Params: +2·(n_kv_heads·d_k·d_model) shared K, V = +2·(2·16·64) = +4,096 params/block (one `nn.Parameter` per K, V per block); +2·n_kv_heads gate scalars = +4 params/block. Total per block = 4,100 params. Across 12 blocks: **+49,200 params** at tiny1m3m, ~+5.2% of the 0.94M baseline.
  - (Review.md finding #3 noted the correct count is **48 total gate scalars** across 12 blocks at n_kv_heads=2, not 96. The plan reflects this.)
- FLOPs: 2 extra `d_model → n_kv_heads·d_k` matmuls per block per forward (one K, one V) on the full `x` tensor → +2·(B·T·d_model·n_kv_heads·d_k) per block per step → ~+2·(B·T·64·2·16) = +4,096·B·T flops/block, ~+5% over baseline.
- Memory: 4,100 extra weight params per block (negligible). No activation memory delta.

## Run
- Artifact: `_arq_178-mqa-gated.py` (repo root) defines top-level `class C(Tiny1M3MMQAGatedConfig): pass` and dispatches `train_llm.main()` with `--config_class __main__.C --seed 42 --dataset_path processed_data/pretrain_1B --warmup false`.
- Descriptor: `autoresearch/ideas/178-mqa-gated/run.json` — `{"name": "178-mqa-gated", "arq_file": "_arq_178-mqa-gated.py", "job_timeout": "12m"}`.
- Daemon (`autoresearch/bin/queue-daemon.sh`): scp's the stub, runs the CPU build-smoke (`python _box_smoke.py _arq_178-mqa-gated.py` → `SMOKE_OK`), then launches the run in the `arq` tmux.
- Pass/fail bar (from `idea.md`): **probe, not a lever** — the primary signal is the per-KV-head β trajectory, not val loss. PASS (mechanistic) = ≥1 β_h moves measurably off zero in any block, OR all β_h collapse to/stay at zero (clean mechanistic close); null = noisy half-move with no clean signal. Val-loss is recorded as the secondary column.

### Runner hooks (review.md finding #1 — primary signal)
The runner must record at end-of-training:
- Per-block, per-KV-head final `β_k_h` and `β_v_h` values (12 blocks × 2 KV-heads × 2 = 48 scalars). Written to `evidence.md` (or `results.json`) so the reviewer can read the trajectory without paying for a second run.
- Cost: a single tensor dump at end-of-run, ~one line per block. Trivial.

### Step-0 byte-identity smoke (review.md finding #2)
CPU build smoke runs `MinimalLLM(C())` with `seed=42` and a `(1, 16)` integer input, compares the flag-on logits to a no-flag `MinimalLLM(Tiny1M3MConfig())` build at the same seed. **Verified locally: `max-abs-diff = 0.0`**. The shared K, V projection allocation does not mutate the parameter state — zero-init + zero-gate ⇒ strict bit-identity on the forward.
