#!/usr/bin/env python3
"""scaling_fit.py — fit L(N) per architecture across the release ladder and
extrapolate to the 135M release point.

The ladder (see autoresearch/LADDER.md) trains the SAME architecture at several
sizes spanning ~1.9 decades of non-embedding parameters N, all at the Chinchilla
20x-tokens budget. For each architecture we fit the power law

    L(N) = E + A * N**(-alpha)

on NON-EMBEDDING N (the 49k-vocab embedding dominates total params at small N and
does not follow the transformer's scaling — standard Kaplan/Chinchilla
convention), then extrapolate to the target's non-embedding N to PREDICT the
135M release loss before spending the 2.7B-token run. The architecture whose
curve sits lowest at the target N (ideally a steeper `alpha`, i.e. an advantage
that GROWS with scale) is the one worth the full run.

Input: JSONL records, one run per line, from --results (default
autoresearch/ladder/results.jsonl) or stdin (--stdin). Each record:

    {"arch": "champion", "N": 1450000, "tokens": 155000000, "val_loss": 4.91, "seed": 42}

`N` is non-embedding parameters. Multiple seeds per (arch,N) are averaged. An
architecture needs >= 3 distinct N to fit the 3-parameter law (>= 4 for a
residual-checked CI).

Usage:
    python3 autoresearch/bin/scaling_fit.py                 # fit from default results file
    python3 autoresearch/bin/scaling_fit.py --baseline champion
    cat runs.jsonl | python3 autoresearch/bin/scaling_fit.py --stdin
    python3 autoresearch/bin/scaling_fit.py --selftest      # verify recovery on synthetic data
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.optimize import curve_fit

# Non-embedding N of the 135M release target (Full135M2700MConfig), verified by
# build. Update if the target architecture changes.
TARGET_N = 106_830_000
TARGET_LABEL = "135M release (Full135M2700MConfig)"
DEFAULT_RESULTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ladder", "results.jsonl"
)


def power_law(N, E, A, alpha):
    """L(N) = E + A * N**(-alpha). E = irreducible loss floor."""
    return E + A * np.power(N, -alpha)


def fit_arch(Ns, Ls):
    """Fit L = E + A*N^-alpha. Returns dict with params, R^2, and a predictor."""
    Ns = np.asarray(Ns, float)
    Ls = np.asarray(Ls, float)
    order = np.argsort(Ns)
    Ns, Ls = Ns[order], Ls[order]

    # Reasonable bounds: E in [0, min(L)], A >= 0, alpha in (0, 1].
    lo = [0.0, 0.0, 1e-3]
    hi = [float(Ls.min()), np.inf, 1.0]
    # Seed from a log-log line through the endpoints (assume E ~ 0 initially).
    a0 = max(1e-3, (math.log(Ls[0]) - math.log(Ls[-1])) / (math.log(Ns[-1]) - math.log(Ns[0])))
    p0 = [min(Ls.min() * 0.5, hi[0]), Ls[-1] * (Ns[-1] ** a0), a0]
    try:
        popt, _ = curve_fit(power_law, Ns, Ls, p0=p0, bounds=(lo, hi), maxfev=200000)
    except Exception:
        # Fallback: pure power law (E=0) via log-log least squares.
        b, loga = np.polyfit(np.log(Ns), np.log(Ls), 1)
        popt = [0.0, math.exp(loga), -b]
    E, A, alpha = popt
    pred = power_law(Ns, *popt)
    ss_res = float(np.sum((Ls - pred) ** 2))
    ss_tot = float(np.sum((Ls - Ls.mean()) ** 2)) or 1e-12
    r2 = 1.0 - ss_res / ss_tot
    rmse = math.sqrt(ss_res / len(Ls))
    return {"E": E, "A": A, "alpha": alpha, "r2": r2, "rmse": rmse,
            "predict": lambda N: float(power_law(np.asarray(N, float), E, A, alpha))}


def bootstrap_ci(Ns, Ls, target_N, n_boot=2000, seed=0):
    """Residual-bootstrap CI on the target-N extrapolation. Needs >= 4 points."""
    Ns = np.asarray(Ns, float)
    Ls = np.asarray(Ls, float)
    if len(Ns) < 4:
        return None
    base = fit_arch(Ns, Ls)
    resid = Ls - np.array([base["predict"](n) for n in Ns])
    rng = np.random.default_rng(seed)
    preds = []
    for _ in range(n_boot):
        boot_L = np.array([base["predict"](n) for n in Ns]) + rng.choice(resid, size=len(resid), replace=True)
        try:
            f = fit_arch(Ns, boot_L)
            preds.append(f["predict"](target_N))
        except Exception:
            continue
    if not preds:
        return None
    preds = np.array(preds)
    return float(np.percentile(preds, 2.5)), float(np.percentile(preds, 97.5))


def load_records(args):
    if args.stdin:
        lines = sys.stdin.read().splitlines()
    else:
        if not os.path.exists(args.results):
            sys.exit(f"no results file at {args.results} (run ladder rungs first, or use --stdin / --selftest)")
        with open(args.results) as f:
            lines = f.read().splitlines()
    recs = []
    for ln in lines:
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        recs.append(json.loads(ln))
    return recs


def aggregate(recs):
    """arch -> {N -> [losses]} averaged over seeds."""
    by = defaultdict(lambda: defaultdict(list))
    for r in recs:
        if r.get("val_loss") is None or r.get("N") is None:
            continue
        by[r["arch"]][int(r["N"])].append(float(r["val_loss"]))
    out = {}
    for arch, dN in by.items():
        Ns = sorted(dN)
        Ls = [float(np.mean(dN[n])) for n in Ns]
        out[arch] = (Ns, Ls)
    return out


def run(args):
    recs = load_records(args)
    data = aggregate(recs)
    if not data:
        sys.exit("no usable (arch, N, val_loss) records found")

    target_N = args.target_n
    rows = []
    for arch, (Ns, Ls) in sorted(data.items()):
        if len(Ns) < 3:
            print(f"[skip] {arch}: only {len(Ns)} distinct N (need >= 3 to fit)", file=sys.stderr)
            continue
        fit = fit_arch(Ns, Ls)
        pred = fit["predict"](target_N)
        ci = bootstrap_ci(Ns, Ls, target_N) if len(Ns) >= 4 else None
        rows.append({"arch": arch, "n_pts": len(Ns), "span_decades":
                     math.log10(max(Ns) / min(Ns)), "alpha": fit["alpha"],
                     "E": fit["E"], "r2": fit["r2"], "rmse": fit["rmse"],
                     "pred": pred, "ci": ci})

    rows.sort(key=lambda r: r["pred"])
    print(f"\n=== Scaling-law extrapolation to {TARGET_LABEL}  (N={target_N/1e6:.1f}M non-embed) ===\n")
    hdr = f"{'arch':22s} {'pts':>3} {'decades':>7} {'alpha':>6} {'E':>6} {'R^2':>6} {'pred@target':>12} {'95% CI':>20}"
    print(hdr); print("-" * len(hdr))
    base_pred = None
    for r in rows:
        if r["arch"] == args.baseline:
            base_pred = r["pred"]
    for r in rows:
        ci = f"[{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]" if r["ci"] else "(need >=4 pts)"
        flag = ""
        if base_pred is not None and r["arch"] != args.baseline:
            d = r["pred"] - base_pred
            flag = f"  Δ{d:+.3f} vs {args.baseline}" + (" WIN" if d < 0 else "")
        print(f"{r['arch']:22s} {r['n_pts']:>3} {r['span_decades']:>7.2f} "
              f"{r['alpha']:>6.3f} {r['E']:>6.3f} {r['r2']:>6.3f} {r['pred']:>12.4f} {ci:>20}{flag}")
    print()
    if rows:
        best = rows[0]
        print(f"-> lowest predicted target loss: {best['arch']} = {best['pred']:.4f} "
              f"(alpha={best['alpha']:.3f}). A STEEPER alpha means the advantage GROWS with scale.\n")


def selftest():
    """Generate synthetic ladder data from a known law + noise; verify recovery."""
    rng = np.random.default_rng(42)
    Ns = [1_450_000, 3_170_000, 10_890_000, 33_240_000]
    truth = {"baseline": (1.8, 90.0, 0.34), "candidate": (1.8, 130.0, 0.40)}
    recs = []
    for arch, (E, A, al) in truth.items():
        for N in Ns:
            for seed in (42, 123):
                L = E + A * N ** (-al) + rng.normal(0, 0.01)
                recs.append({"arch": arch, "N": N, "tokens": int(20 * N), "val_loss": round(L, 4), "seed": seed})
    data = aggregate(recs)
    print("=== SELFTEST: recover known scaling laws ===")
    print(f"target N = {TARGET_N/1e6:.1f}M non-embed\n")
    for arch, (Ns_, Ls_) in data.items():
        fit = fit_arch(Ns_, Ls_)
        E, A, al = truth[arch]
        print(f"{arch:10s} true(E={E},A={A},alpha={al})  ->  "
              f"fit(E={fit['E']:.2f},A={fit['A']:.1f},alpha={fit['alpha']:.3f})  R^2={fit['r2']:.4f}  "
              f"pred@target={fit['predict'](TARGET_N):.4f} (true={E + A*TARGET_N**(-al):.4f})")
    print("\nIf alpha and pred@target are recovered within noise, the fitter works.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=DEFAULT_RESULTS, help="JSONL results file")
    ap.add_argument("--stdin", action="store_true", help="read JSONL records from stdin")
    ap.add_argument("--baseline", default="champion", help="arch to compute Δ against")
    ap.add_argument("--target-n", type=float, default=TARGET_N, help="non-embed N to extrapolate to")
    ap.add_argument("--selftest", action="store_true", help="verify the fitter on synthetic data")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    run(args)


if __name__ == "__main__":
    main()
