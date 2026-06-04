"""Regenerate runs/EVIDENCE_INDEX.md — maps each committed metrics.json to what it is.

Run: python runs/make_evidence_index.py

Context: runs/ is gitignored (checkpoints + logs stay local/remote-only), but the
small metrics.json files ARE force-committed (`git add -f`) so every LEADERBOARD.md
"Evidence" path resolves from the repo. metrics.json has no run-name field, so this
index supplies the human-readable mapping run-dir -> what it is -> leaderboard row.

Numbers below are read live from the JSON; descriptions are curated. Add a line to
DESC when you commit a new run's metrics.json.
"""
import json, glob, os

HERE = os.path.dirname(__file__)

# curated: run-dir -> (leaderboard ref, one-line description)
DESC = {
    # --- tiny1m3m fast idea screen ---
    "tiny1m_ctrl_full": ("tiny1m ctrl", "Tiny1M3M control"),
    "tiny1m_qgain_full": ("tiny1m", "Tiny1M3M + q_gain"),
    "tiny1m_vqgain_full": ("tiny1m", "Tiny1M3M + V-embed + q_gain"),
    "tiny1m_swa_full": ("tiny1m", "Tiny1M3M + SWA(window=512) only"),
    "tiny1m_vqgain_swa_highrope_full": ("tiny1m", "Tiny1M3M + V+q+SWA+HighRoPE"),
    "tiny1m_vqgain_highrope_swa384_full": ("tiny1m", "Tiny1M3M + V+q+HighRoPE+SWA384"),
    "tiny1m_vqgain_swa_rope250k_full": ("tiny1m", "Tiny1M3M + V+q+SWA+RoPE base 250k"),
    "tiny1m_vqgain_swa_rope125k_full": ("tiny1m refine", "Tiny1M3M + V+q+SWA + RoPE base 125k"),
    "tiny1m_vqgain_swa_rope375k_full": ("tiny1m refine", "Tiny1M3M + V+q+SWA + RoPE base 375k"),
    "tiny1m_vqgain_swa_rope750k_full": ("tiny1m refine", "Tiny1M3M + V+q+SWA + RoPE base 750k"),
    "tiny1m_vqgain_rope250k_swa256_full": ("tiny1m refine", "Tiny1M3M + V+q+RoPE base 250k + SWA256"),
    "tiny1m_vqgain_rope250k_swa384_full": ("tiny1m refine", "Tiny1M3M + V+q+RoPE base 250k + SWA384"),
    "tiny1m_vqgain_rope250k_swa768_full": ("tiny1m refine", "Tiny1M3M + V+q+RoPE base 250k + SWA768"),
    "tiny1m_swa_rope250k_full": ("tiny1m refine", "Tiny1M3M + SWA512 + RoPE base 250k, no V/q"),
    "tiny1m_qgain_swa_rope250k_full": ("tiny1m refine", "Tiny1M3M + q_gain + SWA512 + RoPE base 250k, no V-embed"),
    # --- tiny1m architecture sweep on the current best tiny baseline ---
    "tiny1m_arch_base_full": ("tiny1m arch", "Tiny1M3M current best baseline: V+q+SWA384+RoPE250k"),
    "tiny1m_arch_mha_full": ("tiny1m arch", "Tiny1M3M + full MHA (n_kv_heads=4)"),
    "tiny1m_arch_gqa1_full": ("tiny1m arch", "Tiny1M3M + GQA1 (n_kv_heads=1)"),
    "tiny1m_arch_tiedqk_full": ("tiny1m arch", "Tiny1M3M + tied QK"),
    "tiny1m_arch_mla_full": ("tiny1m arch", "Tiny1M3M + MLA latent K/V"),
    "tiny1m_arch_layernorm_full": ("tiny1m arch", "Tiny1M3M + LayerNorm"),
    "tiny1m_arch_postnorm_full": ("tiny1m arch", "Tiny1M3M + post-norm"),
    "tiny1m_arch_gelu_full": ("tiny1m arch", "Tiny1M3M + GELU FFN"),
    "tiny1m_arch_swiglu_full": ("tiny1m arch", "Tiny1M3M + SwiGLU FFN"),
    "tiny1m_arch_linearattn_full": ("tiny1m arch", "Tiny1M3M + linear attention"),
    "tiny1m_arch_qkpostnorm_full": ("tiny1m arch", "Tiny1M3M + QK norm after RoPE"),
    # --- screen20m natural-end tier (step 4,883) ---
    "s_ctrl_full":        ("screen20m ctrl",  "Control, no flags — screen20m baseline"),
    "s_vqgain_swa_full":  ("#51 s42",  "V+q_gain + SWA(window=512), seed 42 (multi-seed mean 4.6676)"),
    "s_vqgain_swa_s43":   ("#51 s43",  "V+q_gain + SWA, seed 43 (multi-seed confirm)"),
    "s_vqgain_full":      ("#39 s42",         "V-embed + per-head q_gain, seed 42"),
    "s_vqgain_s43":       ("#39 s43",         "V+q_gain, seed 43"),
    "s_vqgain_s44":       ("#39 s44",         "V+q_gain, seed 44 (3-seed mean 4.6815, std 0.0057)"),
    "s_voqgain_full":     ("#38 s42",         "V+O + q_gain, seed 42 (O redundant with q_gain)"),
    "s_voqgain_s43":      ("#38 s43",         "V+O+q_gain, seed 43"),
    "s_voqgain_s44":      ("#38 s44",         "V+O+q_gain, seed 44"),
    "s_vqkgain_full":     ("#43 s42",         "V+q_gain+k_gain, seed 42 (k_gain anti-additive on V+q)"),
    "s_vqkgain_s43":      ("#43 s43",         "V+q+k_gain, seed 43"),
    "s_vqkgain_s44":      ("#43 s44",         "V+q+k_gain, seed 44 (3-seed mean 4.6949)"),
    "s_qgain_full":       ("#41",             "q_gain alone, no embeds — standalone lever"),
    "s_kgain_full":       ("#42",             "k_gain alone, no embeds — weaker standalone lever"),
    "s_qkgain_full":      ("#44",             "q_gain + k_gain, no embeds"),
    "s_swa_only_full":    ("#52 s42",         "SWA(window=512) only, no embeds/gains, seed 42"),
    "s_swa_only_s43":     ("#52 s43",         "SWA only, seed 43 (2-seed mean 4.7552)"),
    "s_mha_full":         ("#58",             "Full MHA (n_kv_heads=6) — GQA not a lever at this scale"),
    "s_vqgain_nope_full": ("#54 CLOSED",      "V+q_gain + NoPE — catastrophic, RoPE is load-bearing"),
    "s_vqgain_tied2_full":("#56 CLOSED",      "V+q_gain + ALBERT layer-tying(group=2) — anti-additive"),
    "s_deepv_full":       ("#45",             "2-layer non-linear V-embed (GELU)"),
    "s_deepvqgain_full":  ("#46",             "Deep V-embed + q_gain — anti-additive"),
    "s_valembed_full":    ("#29",             "V-embed alone"),
    "s_vqembed_full":     ("#32",             "V+Q embed combo"),
    "s_voembed_full":     ("#35",             "V+O embed combo"),
    "s_vokembed_full":    ("#36",             "V+O+K embed combo (K neutral here)"),
    "s_vqkembed_full":    ("#34",             "V+Q+K embed (K anti-additive)"),
    "s_vqqgain_full":     ("#40",             "V+Q embed + q_gain (Q-embed redundant with q_gain)"),
    "s_oembed_full":      ("#33",             "O-embed alone"),
    "s_qembed_4k":        ("#30",             "Q-embed, natural-end read"),
    "s_kembed_full":      ("#31",             "K-embed, natural end"),
    # --- variants not yet on the main leaderboard table ---
    "s_ffnembed_full":      ("variant",       "FFN-embed injection"),
    "s_vqgffnembed_full":   ("variant",       "V+q_gain + FFN-embed"),
    "s_vqgain_swiglu_full": ("variant",       "V+q_gain + SWiGLU FFN"),
    "s_vqgain_swa_gelu_full":("#62",  "V+q+SWA + GELU FFN — single-seed 4.6608 (was best until #64)"),
    "s_vqgain_swa_highrope_full":("#64",  "V+q+SWA + RoPE_base=500000 — current screen20m best 4.6364, single-seed"),
    "s_vqgain_swa_highrope_gelu_full":("#65",  "V+q+SWA+HighRoPE + GELU — 4.6527, GELU anti-additive on HighRoPE (closed)"),
    "s_vqgain_swa_highrope_tied2_full":("#66",  "V+q+SWA+HighRoPE + layer tying(2) — 4.7133, tying anti-additive (closed)"),
    "s_vqgain_swa_highrope_mha_full":("#67",  "V+q+SWA+HighRoPE + full MHA — 4.6384, GQA ratio is a wash (closed)"),
    "s_vqgain_swa_highrope_postnorm_full":("#75",  "V+q+SWA+HighRoPE + post-norm — 5.3816, destabilizes training (closed)"),
    "s_vqgain_swa_highrope_gqa1_full":("#76",  "V+q+SWA+HighRoPE + GQA=1 — 4.6761, max KV sharing hurts (closed)"),
    "s_vqgain_swa_highrope_noembscale_full":("#77",  "V+q+SWA+HighRoPE + embedding_scale=1.0 — q6 pending"),
    "s_vqgain_swa_highrope_fullwin_full":("#78",  "V+q+SWA+HighRoPE + SWA window=2048 — q6 pending"),
    "s_vqgain_swa_highrope_layernorm_full":("#79",  "V+q+SWA+HighRoPE + LayerNorm — q6 pending"),
    "s_vqgain_swa_highrope_linearattn_full":("#80",  "V+q+SWA+HighRoPE + positive-feature linear attention — q6 pending"),
    "s_vqgain_swa_highrope_qkpostnorm_full":("#81",  "V+q+SWA+HighRoPE + QK norm after RoPE"),
    "s_vqgain_highrope_swa384_full":("#82",  "V+q+HighRoPE + SWA(window=384)"),
    "s_vqgain_highrope_swa768_full":("#83",  "V+q+HighRoPE + SWA(window=768)"),
    "s_vqgain_swa_rope250k_full":("#84",  "V+q+SWA(window=512) + RoPE base 250k"),
    "s_vqgain_swa_rope1m_full":("#85",  "V+q+SWA(window=512) + RoPE base 1M"),
    "s_vqgain_swa_highrope_linearattn_fixed_full":("#80-fixed",  "V+q+SWA+HighRoPE + actual positive-feature linear attention"),
    "s_vqgain_swa_highrope_tiedqk_full": ("#72",  "V+q+SWA+HighRoPE + Tied QK (PaLM) — 4.6500, closed"),
    "s_vqgain_swa_highrope_mla_full":     ("#73",  "V+q+SWA+HighRoPE + MLA (DeepSeek-V2) — 4.7269, closed"),
    "s_vqgain_swa_highrope_dilated_full": ("#74",  "V+q+SWA+HighRoPE + dilated (d=2) — 5.2494, closed"),
    "s_vqgain_swa_highrope_softcap_full": ("#71",  "V+q+SWA+HighRoPE + logit softcap=15 — 4.6777, closed"),
    "s_vqgain_highrope_swa256_full":     ("#68",  "V+q+HighRoPE + SWA(window=256) — 4.6672, closed"),
    "s_vqgain_highrope_swa1024_full":    ("#69",  "V+q+HighRoPE + SWA(window=1024) — 4.6517, closed"),
    "s_vqgain_highrope_noswa_full":      ("#70",  "V+q+HighRoPE + NO SWA — 4.6841, closed"),
    "s_vqgain_gelu_full":("standalone",   "V+q_gain + GELU FFN, no SWA/HighRoPE"),
    "s_gelu_full":       ("standalone",   "GELU FFN only"),
    "s_highrope_full":   ("standalone",   "RoPE_base=500000 only"),
    "s_gqa1_full":       ("standalone",   "GQA=1 only"),
    "s_vqgainqkpostnorm_full":("variant",     "V+q_gain + QK post-norm"),
    # --- 4k gated screen tier ---
    "s_ctrl":  ("screen16m ctrl", "4k gated control (5.0078)"),
    "s_kembed":("#31 4k",         "K-embed, 4k gated"),
    "s_qembed":("#30 alt-seed",   "Q-embed, alt seed"),
    # --- early-kill structural screens (3-5M, ablation log) ---
    "emb_resid":   ("issue#20 §1", "Embedding residual (early kill)"),
    "s_embresid":  ("issue#20 §1", "Embedding residual (early kill, -0.069)"),
    "s_zeroinit":  ("issue#22 §2", "Zero-init residual projections (null)"),
    "s_outadapter":("ablation §3", "Low-rank output adapter (collapsed)"),
    "s_smeargate": ("issue#27 §4", "SmearGate (weak +0.0053, below bar)"),
    "s_unetskip":  ("issue#23 §5", "U-Net skip connections (-0.0003)"),
    "s_attngate":  ("issue#28 §6", "Attention output gate (killed @2k)"),
    "s_layerscale":("issue#21 §7", "LayerScale (+0.0106, below bar)"),
}


def load(mp):
    d = json.load(open(mp))
    fm = d.get("final_metrics", {})
    return (
        fm.get("val_loss"),
        d.get("actual_steps"),
        d.get("gated"),
        (d.get("git_commit") or "")[:8],
        # Forward-only identity fields (added 2026-06-03, post ef5a523).
        # Old metrics.json won't have them — DESC is the fallback.
        d.get("config_name"),
        d.get("seed"),
        d.get("flags") or {},
        d.get("run_name"),
    )


def auto_label(config_name, seed, flags):
    """Build a one-line self-description for a run that lacks a DESC entry.
    Pulls the architecture-relevant flags out of the dict so a run that
    has never been hand-curated still surfaces "what was on" in the
    index. Pure presentation — does not modify the JSON.
    """
    if not config_name:
        return "(undocumented — add to DESC in make_evidence_index.py)"
    # Pick the most architecture-relevant flags
    interesting = []
    for k in ("use_value_embed", "use_query_embed", "use_key_embed",
              "use_output_embed", "use_deep_value_embed", "use_ffn_embed",
              "use_q_gain", "use_k_gain", "use_qk_norm_post_rope",
              "use_sliding_window", "sliding_window_size", "use_nope",
              "use_layerscale", "use_attn_output_gate", "use_embed_residual",
              "tie_layer_groups", "ffn_variant", "rope_base",
              "n_kv_heads"):
        if k in flags and flags[k] not in (False, 0, 1, None, ""):
            v = flags[k]
            if v is True:
                interesting.append(k.removeprefix("use_"))
            else:
                interesting.append(f"{k.removeprefix('use_')}={v}")
    label = config_name
    if seed is not None:
        label += f" (seed {seed})"
    if interesting:
        label += " — flags: " + ", ".join(interesting)
    return label


def main():
    rows = []
    for mp in sorted(glob.glob(os.path.join(HERE, "*", "metrics.json"))):
        run = os.path.basename(os.path.dirname(mp))
        try:
            vl, steps, gated, commit, config_name, seed, flags, run_name = load(mp)
        except Exception as e:
            vl, steps, gated, commit = f"ERR {e}", "", "", ""
            config_name = seed = flags = run_name = None
        # If run_name is set in the JSON, prefer it (it should match the
        # dir name, but this catches moved JSONs).
        effective_run = run_name or run
        ref, desc = DESC.get(run, ("—", None))
        if desc is None:
            desc = auto_label(config_name, seed, flags)
        rows.append((effective_run, vl, steps, gated, commit, ref, desc))

    rows.sort(key=lambda r: (r[1] if isinstance(r[1], (int, float)) else 1e9))

    lines = [
        "# Evidence index — `runs/*/metrics.json`",
        "",
        "Auto-generated by `runs/make_evidence_index.py`. **Do not edit by hand** —",
        "re-run the script after committing a new run's `metrics.json`.",
        "",
        "`runs/` is gitignored (checkpoints + logs stay local/remote-only). Only the small",
        "`metrics.json` files are force-committed so every `LEADERBOARD.md` Evidence path",
        "resolves. `metrics.json` carries no run name — this table is the run-dir mapping.",
        "",
        "Sorted by val_loss (best first). All numbers read live from the JSON.",
        "",
        "| Run dir | Val loss | Steps | Gated | Commit | LB ref | What it is |",
        "|---|---|---|---|---|---|---|",
    ]
    for run, vl, steps, gated, commit, ref, desc in rows:
        vls = f"{vl:.4f}" if isinstance(vl, (int, float)) else str(vl)
        lines.append(f"| `{run}` | {vls} | {steps} | {gated} | `{commit}` | {ref} | {desc} |")
    lines.append("")
    lines.append(f"_{len(rows)} runs._")

    out = os.path.join(HERE, "EVIDENCE_INDEX.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", out, f"({len(rows)} runs)")


if __name__ == "__main__":
    main()
