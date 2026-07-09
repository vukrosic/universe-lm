#!/usr/bin/env python3
"""ladder_status.py — harvest the box's ladder results to local, summarize the
per-rung arm deltas, and run the scaling fit when there's enough data.

Closes the ladder loop: training runs log points to results.jsonl ON THE BOX;
this pulls them to the local repo, prints a baseline-vs-arm delta table per rung,
and (when any arch has >= 3 distinct N) calls scaling_fit.py for the extrapolation.

  python3 autoresearch/bin/ladder_status.py            # harvest + summarize + fit
  python3 autoresearch/bin/ladder_status.py --no-pull  # summarize the local file only
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
AUTORESEARCH = os.path.dirname(HERE)
REPO = os.path.dirname(AUTORESEARCH)
LOCAL = os.path.join(AUTORESEARCH, "ladder", "results.jsonl")
BOX_JSON = os.path.join(AUTORESEARCH, "remote-box.json")
REMOTE_RESULTS = "/root/universe-lm/autoresearch/ladder/results.jsonl"


def ssh_cmd():
    """Build the ssh argv from remote-box.json (host/port change per instance)."""
    with open(BOX_JSON) as f:
        box = json.load(f)
    return ["ssh", "-p", str(box["port"]), "-o", "ConnectTimeout=20",
            "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes",
            f"{box['user']}@{box['host']}"]


def harvest():
    try:
        out = subprocess.run(ssh_cmd() + [f"cat {REMOTE_RESULTS} 2>/dev/null"],
                             capture_output=True, text=True, timeout=40).stdout
    except Exception as e:
        print(f"[harvest] ssh failed ({e}); using local file if present.")
        return
    lines = [l for l in out.splitlines() if l.strip().startswith("{")]
    if not lines:
        print("[harvest] box returned no points; using local file if present.")
        return
    os.makedirs(os.path.dirname(LOCAL), exist_ok=True)
    with open(LOCAL, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[harvest] {len(lines)} points -> {os.path.relpath(LOCAL, REPO)}")


def load():
    if not os.path.exists(LOCAL):
        return []
    recs = []
    with open(LOCAL) as f:
        for ln in f:
            ln = ln.strip()
            if ln.startswith("{"):
                recs.append(json.loads(ln))
    return recs


def summarize(recs):
    # (N, rung) -> {arch -> val_loss}
    by_rung = defaultdict(dict)
    archs = set()
    for r in recs:
        by_rung[(r["N"], r.get("rung", "?"))][r["arch"]] = r["val_loss"]
        archs.add(r["arch"])
    other = sorted(a for a in archs if a != "baseline")
    print(f"\n=== Ladder status: {len(recs)} points, arms = {', '.join(sorted(archs))} ===")
    hdr = f"{'N (non-embed)':>14} {'rung':>22} {'baseline':>9}" + "".join(f"{a:>18}" for a in other)
    print(hdr); print("-" * len(hdr))
    for (N, rung) in sorted(by_rung):
        row = by_rung[(N, rung)]
        base = row.get("baseline")
        cells = f"{N:>14,} {rung:>22} {('%.4f'%base) if base is not None else '—':>9}"
        for a in other:
            v = row.get(a)
            if v is None:
                cells += f"{'—':>18}"
            elif base is None:
                cells += f"{('%.4f'%v):>18}"
            else:
                d = v - base
                cells += f"{('%.4f (%+.4f)'%(v,d)):>18}"
        print(cells)
    # distinct-N count per arch (need >=3 to fit)
    nN = defaultdict(set)
    for r in recs:
        nN[r["arch"]].add(r["N"])
    print("\ndistinct N per arm: " + ", ".join(f"{a}:{len(nN[a])}" for a in sorted(archs)))
    return max((len(s) for s in nN.values()), default=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-pull", action="store_true", help="don't ssh the box; summarize local file")
    args = ap.parse_args()
    if not args.no_pull:
        harvest()
    recs = load()
    if not recs:
        print("no ladder points yet.")
        return
    max_n = summarize(recs)
    if max_n >= 3:
        print("\n>= 3 rungs for an arm — running scaling_fit.py:\n")
        subprocess.run([sys.executable, os.path.join(HERE, "scaling_fit.py"), "--baseline", "baseline"])
    else:
        print(f"\n(need >= 3 distinct N per arm to fit L(N); best so far = {max_n}.)")


if __name__ == "__main__":
    main()
