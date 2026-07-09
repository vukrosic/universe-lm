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
    AND the paired mean delta is negative AND all 3 seeds individually favor
    treatment (sign-consistency noise floor). band default 0.001 (operator policy
    2026-06-17: promote on any real 3-seed-mean improvement; the paired same-box
    same-session design + 3/3 sign agreement — not a wide band — guards noise).

Usage:
  confirm_paired.py <idea-slug> <flag1[,flag2,...]> [--band 0.001] [--promote]
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


def make_stub(idea, champion_class, base_flags, flags):
    """Paired Ctrl (the REAL champion) vs Trt (champion + the new lever).

    base_flags = the champion's OWN defining flags (e.g. use_deepnet_alpha). The
    champion's config_class usually does NOT encode them — the live champion is a
    base config + flags set in its run stub — so BOTH arms must apply base_flags,
    or the control silently collapses to the bare base config and the confirm
    credits the treatment with the champion's own gain. That is the 267 bug:
    ctrl ran plain Tiny1M3MAlibiConfig (6.2539) instead of alibi+deepnet (6.2367),
    so poly-alibi was credited with deepnet's −0.017 on top of its own marginal.

    `flags` = the NEW lever(s) under test, applied to the treatment arm only.
    @dataclass is mandatory or the inherited dataclass __init__ resets the flags."""
    mod, _, cls = champion_class.rpartition(".")
    base = [f.strip() for f in base_flags if f and f.strip()]
    new = [f.strip() for f in flags if f and f.strip()]
    ctrl_lines = "\n".join(f"    {fl}: bool = True" for fl in base) or "    pass"
    trt_lines = "\n".join(f"    {fl}: bool = True" for fl in (base + new)) or "    pass"
    return f'''#!/usr/bin/env python
"""Auto-generated paired confirm runner for {idea}.
Ctrl = champion [{", ".join(base) or "base"}]; Trt = champion + [{", ".join(new)}]."""
from dataclasses import dataclass
from {mod} import {cls}


@dataclass
class Ctrl({cls}):
{ctrl_lines}


@dataclass
class Trt({cls}):
{trt_lines}


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


def launch(idea, flags, base_flags, ssh, scp, repo, venv):
    champ = load_json(CHAMPION_JSON)
    stub_name = f"_arq_confirm_{idea}.py"
    stub_local = os.path.join(ROOT, stub_name)
    with open(stub_local, "w") as f:
        f.write(make_stub(idea, champ["config_class"], base_flags, flags))

    r = sh(scp + [stub_local, f"{ssh[-1]}:{repo}/"])
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
    # GPU-free guard before each run: the queue-daemon drains the same box 1-by-1
    # (it refuses to launch while any compute proc holds the GPU). A tiny1m3m run
    # is ~6.6 GB of 12 GB, so two concurrent processes OOM. We mirror the daemon's
    # guard here — wait until no other compute proc holds the GPU, THEN run — so a
    # confirm coexists with a live daemon without OOM and without stopping it
    # (a stopped daemon = idle GPU, the #1 lab failure). Each $PY is foreground, so
    # the next iteration only proceeds once this run finished and the GPU freed.
    wait_gpu = ("  while [ \"$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null "
                "| grep -c .)\" -gt 0 ]; do sleep 5; done\n")
    # Clear DONE *and* any stale per-run logs from a prior confirm of this idea.
    # Without this, a re-run leaves old {arm}_{seed}.log files and --collect reads
    # a stale (or fresh+stale mix) 6/6 set, emitting a bogus verdict — exactly how
    # the buggy plain-alibi 267 result kept resurfacing. `*_*.log` matches the arm
    # logs (ctrl_42.log …) but not driver.log (no underscore).
    driver = (f"#!/bin/bash\ncd {repo}\nmkdir -p {rd}\nrm -f {rd}/DONE {rd}/*_*.log\nPY={venv}/bin/python\n"
              f"for seed in {' '.join(SEEDS)}; do for arm in ctrl trt; do\n"
              f"{wait_gpu}"
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
    # Promote on ANY real 3-seed-mean improvement past a tiny epsilon band (0.001,
    # operator policy 2026-06-17). The paired design makes this safe: `cm` is the
    # champion RE-RUN fresh at the same 3 seeds in the same session, so the bar is
    # drift-free (the pinned champion val only gates the cheap 1-seed screen, never
    # this promote). The noise floor is sign-consistency, NOT a wide band: a 0.001
    # win must hold across ALL 3 seeds (trt beats champion at each seed). For a true
    # null the mean alone false-promotes ~37% (dm < -0.001 ~ 0.3 SEM); requiring
    # 3/3 right-sign drops that to (0.5)^3 = 12.5% while still passing any genuine
    # small gain. Drop `all_negative` to promote on the bare 3-seed average.
    all_negative = all(d < 0 for d in deltas)
    confirmed = (dm < 0) and (tm < cm - band) and all_negative
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
        f"- sign-consistency: {sum(d < 0 for d in deltas)}/3 seeds favor treatment"
        + ("" if all_negative else " — FAILS the all-3-seeds-agree guard"),
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
    ap.add_argument("flags", nargs="?", help="comma-separated NEW flag name(s) to add to the champion (treatment arm)")
    ap.add_argument("--base-flags", default=None,
                    help="comma-separated flag(s) that DEFINE the champion control arm "
                         "(e.g. use_deepnet_alpha). Both arms get these; only `flags` differ. "
                         "Defaults to champion.json's `flags` field. Pass this to test a lever's "
                         "MARGINAL gain over a flag-defined champion (avoids the bare-base control bug).")
    ap.add_argument("--band", type=float, default=0.001)
    ap.add_argument("--collect", action="store_true", help="just parse a finished run")
    ap.add_argument("--promote", action="store_true")
    a = ap.parse_args()
    ssh, scp, _, repo, venv = box_ssh_base()

    # Control-arm flags: explicit --base-flags wins, else champion.json's `flags`
    # (the champion's own defining levers), else none (bare config_class). Without
    # this the control collapses to the base config and over-credits the treatment.
    if a.base_flags is not None:
        base_flags = [f for f in a.base_flags.split(",") if f.strip()]
    else:
        base_flags = load_json(CHAMPION_JSON).get("flags", []) or []

    if a.collect:
        vals, done = collect(a.idea, ssh, repo, a.band, a.promote)
        # Require the DONE marker — never judge on log presence alone. A re-run's
        # stale {arm}_{seed}.log files can read as a full 6/6 set BEFORE the fresh
        # run finishes (that is how the buggy plain-alibi 267 verdict resurfaced).
        # DONE is written only after all 6 fresh runs complete, so it is the only
        # safe completion signal.
        if not done:
            print(f"not finished yet: have {len(vals)}/6 runs, DONE marker absent", file=sys.stderr)
            sys.exit(2)
        judge(a.idea, vals, a.band, a.promote)
        return

    if not a.flags:
        sys.exit("need flags-csv to launch (e.g. use_value_residual)")
    print(f"control arm = champion + {base_flags or '[bare config]'}; "
          f"treatment arm = + {a.flags.split(',')}")
    launch(a.idea, a.flags.split(","), base_flags, ssh, scp, repo, venv)
    print(f"\nrunning ~30 min. collect with:\n  autoresearch/bin/confirm_paired.py {a.idea} --collect"
          + (" --promote" if a.promote else ""))


if __name__ == "__main__":
    main()
