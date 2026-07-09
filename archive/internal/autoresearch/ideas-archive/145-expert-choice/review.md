# Review — 145 expert-choice

## r2 — 2026-06-14 — verdict: approve

Source check: arXiv:2202.09368 (Zhou, Lei, et al. 2022, Google, "Mixture-of-Experts with Expert Choice Routing"). Real, well-cited (GLaM-family lineage). ✓

Mechanism: routing direction inversion — each expert picks its top-k tokens (rather than each token picking its top-k experts). This is a *structural* architectural choice, not an HP knob, and the two routing schemes are not reachable from each other via any loss-coefficient sweep. Step-0 identity holds (router zero-init → uniform scores → all experts pick the same k tokens → average of N identical FFNs ≈ single FFN). ✓

Scope: tiny1m3m only, seed 42. ✓

Closed-list check: NOT a mathematical duplicate of 117-soft-moe (soft slot routing), 118-MoD (per-token skip), or 146-sparse-ffn (token-choice top-1 hard). Expert-choice is a genuinely different routing direction. The closed.md entry for 146 says the MoE axis is closed "regardless of routing style," which is suggestive but does not formally subsume expert-choice — the inductive bias (load-balanced by construction, no auxiliary balancing loss) is structurally different. Approving is not "re-running a closed lever" — it is letting the *fourth* data point decide whether the closure is robust to the one remaining routing variant.

- Noted concern: 146 just nulled (Δ=+0.0057 inside |Δ|<0.01 null band; 2026-06-14) and the MoE axis has now been nulled three times at 0.94M. Prior probability of a win is low. Flag for the runner: this run is most likely a fourth null that ratifies the axis closure; the runner should treat that as the expected outcome and write evidence.md accordingly.

LoC budget: `models/expert_choice_moe.py` = 153 lines, plus wiring edits in `models/layers.py`, `models/llm.py`, `configs/llm_config.py`. Total < 200 LoC. ✓

Falsifiable bar: trt vs ctrl val loss at tiny1m3m seed 42. The plan does not state a tight Δ; recommend the runner use the project's two-ctrl rule (PASS ≤ −0.01 vs both controls; NULL band |Δ| < 0.01; DRIFT > +0.01). Note: with the MoE axis already nulled three times, a true win requires beating the bracket by more than the gate noise — surface the bracket (cached baseline 6.4394±0.04) explicitly in evidence.md.

Transfer-risk: med, justified (Google 50B+ MoE; 0.94M is well below the validated range, but the lever has now been nulled three times in this tier regardless of source-paper scale). ✓

Implementation state: `models/expert_choice_moe.py` (153 LoC), `Tiny1M3MExpertChoiceConfig` (configs/llm_config.py:4905), and `_arq_145-expert-choice.py` all present in the local working tree (uncommitted). Verified wiring: 12/12 blocks use `ExpertChoiceMoE` when flag is on; flag-off path is bit-identical. The runner's prior preflight failure was a box-pull issue, not a code defect. The next runner will need to git pull (or the implementer needs to commit+push) to refresh the box.

Round reset to 1 so the code gate gets a fresh budget. Implementing is already complete; the next flip target is `needs-plan`.