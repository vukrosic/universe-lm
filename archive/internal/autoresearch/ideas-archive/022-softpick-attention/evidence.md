# Evidence — 022 Softpick (rectified-softmax attention)

## Verdict: FAIL — NaN, needs-recode
- tier: tiny1m3m, seed 42, box: vast-81.45.65.189 (V100-PCIE-32GB)
- control val: n/a (shared fire-ctrl was buggy — see "Wiring bug" below)
- treatment val: **NaN** (both orig 2026-06-10T08:19Z and rerun 2026-06-10T11:19Z)
- pass/fail bar (plan.md): WIN trt < ctrl − 0.005 / NULL |Δ| < 0.01 / FAIL trt > ctrl + 0.01
- bpb: n/a (NaN)
- box check: shared fire-ctrl had wiring bug (see 020 evidence.md §Wiring bug)
- raw: remote-results/2026-06-10-vast-tiny1m3m/{022-soft.log, 022-soft-r.log}
- date: 2026-06-10

## Wiring bug (cross-cutting)
Same flag-drop in the shared-ctrl subclass pattern as 020-025. The trt config
`Tiny1M3MSoftpickOnFireConfig` is pre-baked and correct (`use_fire_pe=True`,
`use_softpick=True`), but the shared baseline ran without fire.

## NaN mode
- Both runs reached step ~400/732 and stayed at `loss=nan, acc=0.000` for the rest.
- Final Train Loss: nan · Final Val Loss: nan.
- Softpick replaces softmax with `(softmax(x))² / sum((softmax(x))²)` (sink-free
  rectification). At init the softmax is uniform so the squared-renormalised
  distribution is also uniform — no NaN. NaN appears mid-training, suggesting
  a per-row mass collapse: when one row's softmax becomes very peaky, the
  square concentrates the mass further, and the renorm can divide by a small
  positive that hits fp32 underflow → +inf → NaN on subsequent matmuls.
- The implementation likely needs (a) clamp the denominator, (b) softmax in
  fp32, or (c) a small `+ε` in the denominator.

## Transfer note
The mechanism (arXiv:2504.20966) is reported as a drop-in for softmax with no
size-dependent numerical tricks. The NaN is therefore an impl bug, not a
mechanism failure — fix the clamp, re-test. The lever is plausible at 135M
once the numerical guard is in. Recommend: needs-recode → code-implementer adds
fp32 row-renorm safety + denom clamp; no spec change.
