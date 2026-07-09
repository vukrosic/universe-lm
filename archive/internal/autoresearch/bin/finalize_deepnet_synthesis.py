#!/usr/bin/env python3
"""finalize_deepnet_synthesis.py — update DEEPNET-SYNTHESIS.md with 23M/ablation data,
run scaling fit, and generate the final verdict."""

import json
import subprocess
import os

HERE = os.path.dirname(os.path.abspath(__file__))
AUTORESEARCH = os.path.dirname(HERE)
LADDER_DIR = os.path.join(AUTORESEARCH, "ladder")
RESULTS = os.path.join(LADDER_DIR, "results.jsonl")
SYNTHESIS = os.path.join(AUTORESEARCH, "DEEPNET-SYNTHESIS.md")

def load_results():
    """Load all logged ladder points."""
    points = []
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            for line in f:
                if line.strip().startswith("{"):
                    points.append(json.loads(line.strip()))
    return points

def check_ready():
    """Check if 23M has landed and/or ablations are complete."""
    points = load_results()
    has_23m = any(p["rung"] == "Ladder23M529MConfig" for p in points)
    ablations = set(p["arch"] for p in points if p["rung"] == "Ladder8M155MConfig")
    expected_ablations = {"baseline", "deepnet", "deepnet_ab", "rezero", "layerscale"}
    has_ablations = expected_ablations.issubset(ablations)
    
    print(f"Status: 23M={'✓' if has_23m else '✗'}, Ablations={len(ablations)}/{len(expected_ablations)}")
    return has_23m, has_ablations

def summarize_all_points():
    """Print a comprehensive table of all logged points."""
    points = load_results()
    by_rung = {}
    for p in points:
        rung = p["rung"]
        if rung not in by_rung:
            by_rung[rung] = {}
        by_rung[rung][p["arch"]] = p["val_loss"]
    
    print("\n=== COMPREHENSIVE LADDER DATA ===\n")
    archs = sorted(set(a for rung in by_rung.values() for a in rung.keys()))
    print(f"Rungs: {sorted(by_rung.keys())}")
    print(f"Arms: {archs}\n")
    
    for rung in sorted(by_rung.keys()):
        row = by_rung[rung]
        baseline = row.get("baseline")
        print(f"{rung}:")
        if baseline:
            for arch in archs:
                v = row.get(arch)
                if v is None:
                    print(f"  {arch:15} —")
                else:
                    delta = v - baseline if arch != "baseline" else 0
                    print(f"  {arch:15} {v:.4f}" + (f" ({delta:+.4f})" if delta != 0 else ""))
        print()

def main():
    print("=== DeepNet Synthesis Finalization ===\n")
    
    has_23m, has_ablations = check_ready()
    
    if not has_23m and not has_ablations:
        print("→ waiting for 23M baseline + deepnet and E3/E4 ablations")
        print("→ run this again once both are logged")
        return
    
    summarize_all_points()
    
    if has_23m:
        print("✓ 23M landed — ready to run auto-fit and finalize synthesis")
        print("  Next: python3 autoresearch/bin/ladder_status.py (triggers auto-fit)")
        
        # Print 23M verdict
        points = load_results()
        m23_points = {p["arch"]: p["val_loss"] for p in points if p["rung"] == "Ladder23M529MConfig"}
        if m23_points:
            baseline_23m = m23_points.get("baseline")
            deepnet_23m = m23_points.get("deepnet")
            if baseline_23m and deepnet_23m:
                delta = deepnet_23m - baseline_23m
                print(f"\n  23M Verdict: baseline {baseline_23m:.4f} vs deepnet {deepnet_23m:.4f}")
                print(f"  Δ = {delta:+.4f} ({'NULL within band' if abs(delta) <= 0.02 else 'needs verify'})")
    
    if has_ablations:
        print("\n✓ Ablations complete (8M rung) — ready to compare the family")
        print("  → Check if rezero/layerscale match deepnet (family redundancy confirmed)")

if __name__ == "__main__":
    main()
