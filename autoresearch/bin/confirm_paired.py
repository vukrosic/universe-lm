#!/usr/bin/env python3
"""confirm_paired.py — the paired 3-seed CONFIRM stage that gates promotion.

A single-seed screen WIN never becomes champion directly (that is how 180-leak
and 209-false-win poisoned the baseline). The daemon parks a screen-WIN in
`needs-confirm`; this tool runs the confirm and promotes only if it holds.

What it does (mirrors the validated 208 confirm, see
autoresearch/ideas/208-value-residual-alibi/confirm-3seed.md):
  * control arm = the current champion (champion.json config_class)
  * treatment arm = champion + the idea's flag(s)
  * 3 seeds (42/123/7) per arm, back-to-back in ONE session on ONE box
    => only within-session noise is in play (no cross-box drift)
  * paired verdict: CONFIRM iff trt 3-seed mean beats champion by > band
    AND the paired mean delta is negative. band default 0.018 ~= 2*SEM of a
    3-seed mean at this tier (within-session sigma ~0.015).

Usage:
  confirm_paired.py <idea-slug> <flag1[,flag2,...]> [--band 0.018] [--promote]
  confirm_paired.py <idea-slug> --collect          # parse an already-finished run

Box/champion config come from autoresearch/{remote-box.json,champion.json}.
This is deliberately a standalone operator tool, not in the daemon hot loop:
a confirm is ~30 min of GPU and only fires for the rare screen-WIN.
"""
import argparse
import json
import os
import statistics as st
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BOX_JSON = os.path.join(ROOT, "autoresearch", "remote-box.json")
CHAMPION_JSON = os.path.join(ROOT, "autoresearch", "champion.json")
SEEDS = ["42", "123", "7"]


def load_json(p):
    with open(p) as f:
        return json.load(f)


def box_ssh_base():
    b = load_json(BOX_JSON)
    host, port = b["host"], str(b["port"])
    user = b.get("user", "root")
    repo = b.get("remote_repo", "/root/universe-lm")
    venv = b.get("remote_venv", "/venv/main")
    ctl = f"/tmp/lab-arq-ctl-{user}-{host}-{port}"
    ssh = ["ssh", "-o", f"ControlPath={ctl}", "-o", "ConnectTimeout=20", "-p", port, f"{user}@{host}"]
    scp = ["scp", "-o", f"ControlPath={ctl}", "-P", port]
    return ssh, scp, f"{user}@{host}", repo, venv


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def make_stub(idea, champion_class, flags):
    """champion (Ctrl) vs champion+flags (Trt). @dataclass is mandatory or the
    inherited dataclass __init__ resets the new flag to its parent default."""
    mod, _, cls = champion_class.rpartition(".")
    flag_lines = "\n".join(f"    {fl.strip()}: bool = True" for fl in flags)
    return f'''#!/usr/bin/env python
"""Auto-generated paired confirm runner for {idea}: champion + [{", ".join(flags)}]."""
from dataclasses import dataclass
from {mod} import {cls}


@dataclass
class Ctrl({cls}):
    pass


@dataclass
class Trt({cls}):
{flag_lines}


C = Trt

if __name__ == "__main__":
    import sys, train_llm
    arm = sys.argv[1]; seed = sys.argv[2]
    sys.modules["__main__"].RunCfg = {{"ctrl": Ctrl, "trt": Trt}}[arm]
    sys.argv = ["train_llm.py", "--config_class", "__main__.RunCfg",
                "--seed", seed, "--dataset_path", "processed_data/pretrain_1B",
                "--warmup", "false"]
    train_llm.main()
'''


def remote_dir(idea):
    return f"/root/arq/confirm_{idea}"


def launch(idea, flags, ssh, scp, repo, venv):
    champ = load_json(CHAMPION_JSON)
    stub_name = f"_arq_confirm_{idea}.py"
    stub_local = os.path.join(ROOT, stub_name)
    with open(stub_local, "w") as f:
        f.write(make_stub(idea, champ["config_class"], flags))

    r = sh(scp + [stub_local, f"{repo}/"])
    if r.returncode != 0:
        sys.exit(f"scp failed: {r.stderr.strip()}")

    # build-smoke both arms; warn loudly if param counts match (flag not wired)
    smoke = (f"cd {repo} && {venv}/bin/python - <<'PY'\n"
             "import importlib.util\n"
             f"s=importlib.util.spec_from_file_location('s','{stub_name}')\n"
             "m=importlib.util.module_from_spec(s); s.loader.exec_module(m)\n"
             "from models.llm import MinimalLLM\n"
             "pc={n:sum(p.numel() for p in MinimalLLM(c()).parameters()) for n,c in [('ctrl',m.Ctrl),('trt',m.Trt)]}\n"
             "print('PARAMS', pc['ctrl'], pc['trt'], 'WIRED' if pc['trt']!=pc['ctrl'] else 'NOT-WIRED')\n"
             "PY")
    r = sh(ssh + [smoke])
    out = (r.stdout + r.stderr).strip().splitlines()
    pline = next((l for l in out if l.startswith("PARAMS")), "")
    print(f"smoke: {pline or out[-3:]}")
    if "NOT-WIRED" in pline:
        sys.exit(f"ABORT: treatment param count == control — flag(s) {flags} not wired into the model. "
                 f"This idea is not actually implemented; nothing to confirm.")
    if "WIRED" not in pline:
        sys.exit(f"ABORT: build-smoke did not pass:\n{r.stdout}\n{r.stderr}")

    rd = remote_dir(idea)
    driver = (f"#!/bin/bash\ncd {repo}\nmkdir -p {rd}\nrm -f {rd}/DONE\nPY={venv}/bin/python\n"
              f"for seed in {' '.join(SEEDS)}; do for arm in ctrl trt; do\n"
              f"  echo \"[$(date -u +%H:%M:%S)] START $arm seed=$seed\"\n"
              f"  $PY {stub_name} $arm $seed > {rd}/${{arm}}_${{seed}}.log 2>&1\n"
              f"  echo \"[$(date -u +%H:%M:%S)] END $arm seed=$seed -> $(grep 'Final Val Loss' {rd}/${{arm}}_${{seed}}.log|tail -1)\"\n"
              f"done; done\necho DONE > {rd}/DONE\n")
    setup = (f"mkdir -p {rd}; cat > {rd}/run.sh <<'EOF'\n{driver}EOF\n"
             f"chmod +x {rd}/run.sh; tmux kill-session -t confirm_{idea} 2>/dev/null; "
             f"tmux new-session -d -s confirm_{idea} \"bash {rd}/run.sh > {rd}/driver.log 2>&1\"; "
             f"sleep 2; tmux ls 2>/dev/null | grep confirm_{idea}")
    r = sh(ssh + [setup])
    print(f"launched: {r.stdout.strip() or r.stderr.strip()}")


def collect(idea, ssh, repo, band, promote):
    rd = remote_dir(idea)
    grab = (f"for s in {' '.join(SEEDS)}; do for a in ctrl trt; do "
            f"f={rd}/${{a}}_${{s}}.log; [ -f \"$f\" ] && "
            f"printf '%s_%s %s\\n' $a $s \"$(grep 'Final Val Loss' $f|tail -1|grep -oE '[0-9]+\\.[0-9]+')\"; "
            f"done; done; cat {rd}/DONE 2>/dev/null||echo notyet")
    r = sh(ssh + [grab])
    vals = {}
    done = False
    for line in r.stdout.splitlines():
        line = line.strip()
        if line == "DONE":
            done = True
        elif line and line[0] in "ct":
            k, _, v = line.partition(" ")
            if v:
                vals[k] = float(v)
    return vals, done


def judge(idea, vals, band, promote):
    missing = [f"{a}_{s}" for s in SEEDS for a in ("ctrl", "trt") if f"{a}_{s}" not in vals]
    if missing:
        return f"incomplete — missing {missing}", False
    ctrl = [vals[f"ctrl_{s}"] for s in SEEDS]
    trt = [vals[f"trt_{s}"] for s in SEEDS]
    deltas = [vals[f"trt_{s}"] - vals[f"ctrl_{s}"] for s in SEEDS]
    cm, tm = st.mean(ctrl), st.mean(trt)
    dm, dsem = st.mean(deltas), st.stdev(deltas) / (3 ** 0.5)
    confirmed = (dm < 0) and (tm < cm - band)
    lines = [
        f"# Paired 3-seed CONFIRM — {idea}",
        "",
        "| seed | ctrl | trt | Δ |",
        "|---|---|---|---|",
        *[f"| {s} | {vals[f'ctrl_{s}']:.4f} | {vals[f'trt_{s}']:.4f} | {vals[f'trt_{s}']-vals[f'ctrl_{s}']:+.4f} |" for s in SEEDS],
        "",
        f"- ctrl 3-seed mean **{cm:.4f}** (median {st.median(ctrl):.4f})",
        f"- trt  3-seed mean **{tm:.4f}** (median {st.median(trt):.4f})",
        f"- paired Δ mean **{dm:+.4f} ± {dsem:.4f} SEM**, band {band}",
        "",
        f"## Verdict: {'CONFIRMED — promote' if confirmed else 'NOT CONFIRMED — stays null, champion unchanged'}",
    ]
    ev_path = os.path.join(ROOT, "autoresearch", "ideas", idea, "confirm-paired.md")
    if os.path.isdir(os.path.dirname(ev_path)):
        with open(ev_path, "w") as f:
            f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    if confirmed and promote:
        promote_champion(idea, tm, ctrl)
    return ("CONFIRMED" if confirmed else "NOT-CONFIRMED"), confirmed


def promote_champion(idea, trt_mean, ctrl_runs):
    print(f"\n(--promote not auto-wiring config_class here; record + re-pin is a deliberate human step)\n"
          f"CONFIRMED: {idea} trt 3-seed mean {trt_mean:.4f} beats champion. "
          f"To promote: update champion.json config_class to the champion+flag class and val to {trt_mean:.4f}.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("idea")
    ap.add_argument("flags", nargs="?", help="comma-separated flag name(s) to add to champion")
    ap.add_argument("--band", type=float, default=0.018)
    ap.add_argument("--collect", action="store_true", help="just parse a finished run")
    ap.add_argument("--promote", action="store_true")
    a = ap.parse_args()
    ssh, scp, _, repo, venv = box_ssh_base()

    if a.collect:
        vals, done = collect(a.idea, ssh, repo, a.band, a.promote)
        if not done and len(vals) < 6:
            print(f"not finished yet: have {len(vals)}/6 runs", file=sys.stderr)
            sys.exit(2)
        judge(a.idea, vals, a.band, a.promote)
        return

    if not a.flags:
        sys.exit("need flags-csv to launch (e.g. use_value_residual)")
    launch(a.idea, a.flags.split(","), ssh, scp, repo, venv)
    print(f"\nrunning ~30 min. collect with:\n  autoresearch/bin/confirm_paired.py {a.idea} --collect"
          + (" --promote" if a.promote else ""))


if __name__ == "__main__":
    main()
