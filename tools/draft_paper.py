#!/usr/bin/env python3
"""Seed compiler: one done idea folder -> draft.md (markdown paper draft).

Reads autoresearch/ideas/<slug>/{idea.md, evidence.md, log.jsonl} and writes
draft.md alongside. Stdlib only.

Sections emitted:
  Title, Abstract, 1 Introduction, 2 Method, 3 Experimental setup,
  4 Results (table), 5 Discussion, References.

Usage:
  python3 tools/draft_paper.py <slug>          # e.g. 001-cautious-muon
  python3 tools/draft_paper.py --all           # every done/ idea
"""
import os, re, sys, json, glob, argparse, datetime as dt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "autoresearch", "ideas"))


def _read(p):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""


def parse_idea(md):
    """Split idea.md into frontmatter dict + {section-title: body} map."""
    fm = {}
    body = md
    m = re.match(r"\A---\n(.*?)\n---\n(.*)\Z", md, re.S)
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                fm[k.strip()] = v.strip()
        body = m.group(2)
    title = ""
    mt = re.search(r"^#\s+(.+)$", body, re.M)
    if mt:
        title = mt.group(1).strip()
    sections = {}
    cur = None
    buf = []
    for line in body.splitlines():
        h = re.match(r"^##\s+(.+)$", line)
        if h:
            if cur:
                sections[cur] = "\n".join(buf).strip()
            cur = h.group(1).strip()
            buf = []
        elif cur:
            buf.append(line)
    if cur:
        sections[cur] = "\n".join(buf).strip()
    return fm, title, sections


VERDICT_RE = re.compile(r"^##?\s*Verdict[:\s]+([A-Z]+)", re.M)
NUM_RE = re.compile(r"(-?\d+\.\d+)")


def parse_evidence(md):
    """Pull verdict + bullet body from evidence.md. Returns dict."""
    out = {"verdict": "UNKNOWN", "bullets": [], "raw": md.strip()}
    mv = VERDICT_RE.search(md)
    if mv:
        out["verdict"] = mv.group(1).strip()
    for line in md.splitlines():
        s = line.strip()
        if s.startswith("- "):
            out["bullets"].append(s[2:].strip())
    return out


def extract_numbers(evid):
    """Best-effort: pull treatment, ctrl, delta numbers from bullet text.
    Strips parentheticals so `(mean 6.40)` doesn't double-count as a sample."""
    nums = {"treatment": [], "ctrl": [], "delta": None}
    paren = re.compile(r"\([^)]*\)")
    for b in evid["bullets"]:
        low = b.lower()
        clean = paren.sub("", b)
        if low.startswith("treatment"):
            nums["treatment"] = [float(x) for x in NUM_RE.findall(clean)]
        elif low.startswith("control") or low.startswith("ctrl"):
            nums["ctrl"] = [float(x) for x in NUM_RE.findall(clean)]
        elif "Δ" in b or "delta" in low:
            ms = NUM_RE.findall(clean)
            if ms:
                nums["delta"] = float(ms[0])
    return nums


def parse_log_jsonl(p):
    """Return list of status-event dicts; tolerate empty/missing."""
    out = []
    if not os.path.exists(p):
        return out
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def results_table(nums):
    """Markdown table; gracefully handles missing values."""
    rows = []
    rows.append("| Arm | Val loss (per run) | Mean |")
    rows.append("|---|---|---|")

    def fmt(xs):
        if not xs:
            return ("—", "—")
        mean = sum(xs) / len(xs)
        return (", ".join(f"{x:.4f}" for x in xs), f"{mean:.4f}")

    t_each, t_mean = fmt(nums["treatment"])
    c_each, c_mean = fmt(nums["ctrl"])
    rows.append(f"| Control (ctrl + ctrl2) | {c_each} | {c_mean} |")
    rows.append(f"| Treatment | {t_each} | {t_mean} |")
    return "\n".join(rows)


def discussion(fm, evid, nums):
    v = evid["verdict"]
    d = nums.get("delta")
    if v == "WIN":
        return ("The treatment beats both controls beyond the ctrl-to-ctrl gap "
                "(two-ctrl rule satisfied). Δ = {} vs mean control. "
                "Effect survives at this scale; next step is a wider-tier "
                "replication.".format(d if d is not None else "see table"))
    if v == "NULL":
        return ("Treatment lands inside the ctrl-to-ctrl noise band; the "
                "two-ctrl bracket is not cleared. Δ = {}. Reporting as NULL "
                "and closing the idea — no further runs on additional seeds "
                "(single-seed rule).".format(d if d is not None else "n/a"))
    if v == "REJECT":
        return "Idea rejected in review prior to compute spend; see review.md."
    return "Verdict not yet recorded; this draft is preliminary."


def abstract(title, fm, ideas_sec, evid, nums):
    mech = ideas_sec.get("Mechanism", "").strip().split("\n\n")[0]
    if not mech:
        mech = "We evaluate the proposed change on a tiny-scale LM training run."
    v = evid["verdict"]
    d = nums.get("delta")
    if v == "WIN":
        outcome = (f"We observe a WIN with Δ = {d} vs mean control under a "
                   f"two-ctrl bracket.")
    elif v == "NULL":
        outcome = (f"We report a NULL: treatment lies within the ctrl-to-ctrl "
                   f"noise band (Δ = {d}).")
    else:
        outcome = f"Verdict: {v}."
    tier = fm.get("tier") or "tiny1m3m"
    return f"{mech} We test on {tier} (seed 42). {outcome}"


def references(ideas_sec):
    src = ideas_sec.get("Source", "").strip()
    if not src:
        return "_None._"
    return f"1. {src}"


def build_draft(slug):
    folder = os.path.join(ROOT, slug)
    idea_md = _read(os.path.join(folder, "idea.md"))
    evid_md = _read(os.path.join(folder, "evidence.md"))
    if not idea_md:
        raise FileNotFoundError(f"no idea.md in {folder}")
    fm, title, ideas_sec = parse_idea(idea_md)
    evid = parse_evidence(evid_md) if evid_md else {"verdict": "PENDING", "bullets": [], "raw": ""}
    nums = extract_numbers(evid)
    log = parse_log_jsonl(os.path.join(folder, "log.jsonl"))
    done_ts = next((e["ts"] for e in reversed(log) if e.get("to") == "done"), fm.get("updated", "—"))

    title = title or slug
    parts = []
    parts.append(f"# {title}\n")
    parts.append(f"_Auto-drafted {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d')} from `autoresearch/ideas/{slug}/`._\n")
    parts.append("\n## Abstract\n")
    parts.append(abstract(title, fm, ideas_sec, evid, nums) + "\n")

    parts.append("\n## 1 Introduction\n")
    src = ideas_sec.get("Source", "").strip()
    if src:
        parts.append(f"This work re-implements and stress-tests the mechanism from {src}.\n")
    parts.append("We integrate the change into our standard tiny-scale training "
                 "harness (MinimalLLM, seed 42) and evaluate against a two-control "
                 "bracket to separate signal from kernel-level nondeterminism.\n")

    parts.append("\n## 2 Method\n")
    mech = ideas_sec.get("Mechanism", "").strip()
    if mech:
        parts.append(mech + "\n")
    lr = ideas_sec.get("LR compensation (project-specific, not from the paper)", "").strip()
    if lr:
        parts.append("\n**LR compensation.** " + lr + "\n")

    parts.append("\n## 3 Experimental setup\n")
    runn = ideas_sec.get("Run notes", "").strip()
    if runn:
        parts.append(runn + "\n")
    else:
        parts.append("Single seed (42); tiny1m3m tier; two control replicates "
                     "vs one treatment.\n")
    bar = ideas_sec.get("Pass / fail bar", "").strip()
    if bar:
        parts.append("\n**Pass/fail bar.**\n" + bar + "\n")

    parts.append("\n## 4 Results\n")
    parts.append(results_table(nums) + "\n")
    if evid["raw"]:
        parts.append("\n<details><summary>raw evidence.md</summary>\n\n" + evid["raw"] + "\n\n</details>\n")

    parts.append("\n## 5 Discussion\n")
    parts.append(discussion(fm, evid, nums) + "\n")

    parts.append("\n## References\n")
    parts.append(references(ideas_sec) + "\n")

    parts.append(f"\n---\n_Status_: **{fm.get('status', '?')}** · _Verdict_: **{evid['verdict']}** · _Closed_: {done_ts}\n")

    return "".join(parts)


def write_draft(slug):
    md = build_draft(slug)
    out = os.path.join(ROOT, slug, "draft.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)
    return out


def done_slugs():
    out = []
    for p in sorted(glob.glob(os.path.join(ROOT, "*", "idea.md"))):
        body = _read(p)
        m = re.search(r"^status:\s*done\b", body, re.M)
        if m:
            out.append(os.path.basename(os.path.dirname(p)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slug", nargs="?", help="idea slug, e.g. 001-cautious-muon")
    ap.add_argument("--all", action="store_true", help="draft every done/ idea")
    args = ap.parse_args()
    if args.all:
        slugs = done_slugs()
    elif args.slug:
        slugs = [args.slug]
    else:
        ap.error("pass a slug or --all")
    for s in slugs:
        path = write_draft(s)
        print(path)


if __name__ == "__main__":
    main()
