"""Verify every U-Net skip ablation instantiates and behaves as expected.

Run from repo root:
    python3 tests/test_unet_ablations.py

For each planned ablation:
  - The model instantiates without error.
  - The gate tensor has the expected shape and init value.
  - The effective gate (raw, or sigmoid-wrapped) matches the expected value.
  - Toggling use_unet_skips on/off on the same model isolates the U-Net
    contribution: zero-effective-gate ablations contribute exactly 0;
    nonzero gates contribute >0 at step 0.

Also asserts that out-of-bounds skip_count and invalid gate_type both raise.
"""
import sys
import torch
from configs.llm_config import Tiny1M3MConfig
from models.llm import MinimalLLM


ABLATIONS = [
    # (name, config-overrides, expected-properties)
    ("tiny_unet_ctrl",        dict(use_unet_skips=False),
        dict(has_gates=False, eff=None)),
    ("tiny_unet_raw0",        dict(use_unet_skips=True, unet_gate_type="raw",     unet_gate_init=0.0),
        dict(has_gates=True, shape=(6, 64), raw=0.0,   eff=0.0)),
    ("tiny_unet_raw018",      dict(use_unet_skips=True, unet_gate_type="raw",     unet_gate_init=0.18),
        dict(has_gates=True, shape=(6, 64), raw=0.18,  eff=0.18)),
    ("tiny_unet_sigmoid_m15", dict(use_unet_skips=True, unet_gate_type="sigmoid", unet_gate_init=-1.5),
        dict(has_gates=True, shape=(6, 64), raw=-1.5,  eff=0.1824)),
    ("tiny_unet_sigmoid_m30", dict(use_unet_skips=True, unet_gate_type="sigmoid", unet_gate_init=-3.0),
        dict(has_gates=True, shape=(6, 64), raw=-3.0,  eff=0.0474)),
    ("tiny_unet_raw0_k2",     dict(use_unet_skips=True, unet_skip_count=2),
        dict(has_gates=True, shape=(2, 64), raw=0.0,   eff=0.0)),
    ("tiny_unet_raw0_k4",     dict(use_unet_skips=True, unet_skip_count=4),
        dict(has_gates=True, shape=(4, 64), raw=0.0,   eff=0.0)),
    ("tiny_unet_raw0_k6",     dict(use_unet_skips=True, unet_skip_count=6),
        dict(has_gates=True, shape=(6, 64), raw=0.0,   eff=0.0)),
]


def main() -> int:
    torch.manual_seed(0)
    input_ids = torch.randint(0, 49152, (1, 32))

    print(f"{'name':<26} {'OK':<4} {'gate_shape':<10} {'raw':<7} {'eff':<7} {'contrib':<10} {'check'}")
    print("-" * 95)

    all_pass = True
    for name, kwargs, exp in ABLATIONS:
        cfg = Tiny1M3MConfig()
        for k, v in kwargs.items():
            setattr(cfg, k, v)
        try:
            torch.manual_seed(42)
            m = MinimalLLM(cfg)
            m.eval()
        except Exception as e:
            print(f"{name:<26} FAIL  instantiation: {e}")
            all_pass = False
            continue

        checks = []
        has_gates = hasattr(m, "unet_skip_gates")
        if has_gates != exp["has_gates"]:
            checks.append(f"has_gates {has_gates}!={exp['has_gates']}")

        if has_gates:
            g = m.unet_skip_gates
            if g.shape != exp["shape"]:
                checks.append(f"shape {tuple(g.shape)}!={exp['shape']}")
            if not torch.allclose(g, torch.full_like(g, exp["raw"]), atol=1e-6):
                checks.append("raw init mismatch")
            if m.unet_gate_type == "sigmoid":
                eff_actual = torch.sigmoid(g.flatten()[0]).item()
            else:
                eff_actual = g.flatten()[0].item()
            if abs(eff_actual - exp["eff"]) > 1e-3:
                checks.append(f"effective {eff_actual:.4f}!={exp['eff']:.4f}")

        # Isolate U-Net contribution: toggle on/off on SAME model
        with torch.no_grad():
            if has_gates:
                m.use_unet_skips = True
                out_with = m(input_ids)
                m.use_unet_skips = False
                out_without = m(input_ids)
                m.use_unet_skips = True
                contrib = (out_with - out_without).abs().max().item()
            else:
                m(input_ids)
                contrib = 0.0

        if has_gates:
            if exp["eff"] == 0.0 and contrib > 1e-6:
                checks.append(f"zero-gate should contribute 0, got {contrib:.2e}")
            if exp["eff"] > 0.0 and contrib < 1e-4:
                checks.append(f"nonzero gate should contribute >0, got {contrib:.2e}")

        ok = "PASS" if not checks else "FAIL"
        if checks:
            all_pass = False
        shape_s = str(tuple(g.shape)) if has_gates else "-"
        raw_s = f"{exp['raw']:.3f}" if has_gates else "-"
        eff_s = f"{exp['eff']:.4f}" if has_gates else "-"
        contrib_s = f"{contrib:.2e}"
        check_s = "; ".join(checks) if checks else "ok"
        print(f"{name:<26} {ok:<4}  {shape_s:<10} {raw_s:<7} {eff_s:<7} {contrib_s:<10} {check_s}")

    # Bounds check
    try:
        cfg_bad = Tiny1M3MConfig()
        cfg_bad.use_unet_skips = True
        cfg_bad.unet_skip_count = 99
        MinimalLLM(cfg_bad)
        print(f"{'bounds_check_k99':<26} FAIL  should have raised")
        all_pass = False
    except ValueError:
        print(f"{'bounds_check_k99':<26} PASS  ok (rejects k>n_layers//2)")

    # Gate-type validation
    try:
        cfg_bad = Tiny1M3MConfig()
        cfg_bad.use_unet_skips = True
        cfg_bad.unet_gate_type = "tanh"
        MinimalLLM(cfg_bad)
        print(f"{'gate_type_validation':<26} FAIL  should have raised")
        all_pass = False
    except ValueError:
        print(f"{'gate_type_validation':<26} PASS  ok (rejects gate_type=tanh)")

    print()
    print("=" * 60)
    print("RESULT:", "ALL PASS - safe to launch on Kaggle" if all_pass else "SOME FAILED - DO NOT LAUNCH")
    print("=" * 60)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
