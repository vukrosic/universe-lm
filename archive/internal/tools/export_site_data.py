#!/usr/bin/env python3
"""Export lab state as static JSON for the public website.

Walks token2science/papers/*/paper.json (+ optional figures/curves.json and
manuscript/related-work markdown) and writes one summary file the website
renders at /lab. The website is a static GitHub Pages export, so this is the
entire data pipeline: run this, commit the JSON in the website repo, deploy.

Run:  python3 tools/export_site_data.py [--site-dir PATH]
Default site dir: ../open-superintelligence-lab-github-io
"""
import argparse, glob, json, os, re, shutil, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PAPERS = os.path.join(ROOT, "token2science", "papers")
IDEAS = os.path.join(ROOT, "autoresearch", "ideas")
PIPELINE = os.path.join(ROOT, "autoresearch", "PIPELINE.md")
LEADERBOARD = os.path.join(ROOT, "LEADERBOARD.md")


def load(path):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return None


def strip_frontmatter(text):
    if not text:
        return ""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return text
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return "\n".join(lines[i + 1 :])
    return text


def body_text(text):
    text = strip_frontmatter(text or "")
    lines = text.splitlines()
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i < len(lines) and re.match(r"^\s*#\s+\S", lines[i]):
        i += 1
    while i < len(lines) and not lines[i].strip():
        i += 1
    return "\n".join(lines[i:]).strip()


def clean_md(text):
    text = text or ""
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", text)
    text = re.sub(r"\*([^*\n]+)\*", r"\1", text)
    text = re.sub(r"^\s{0,3}>\s?", "", text, flags=re.M)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.M)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.M)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.M)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def first_h1(text, fallback):
    text = strip_frontmatter(text or "")
    m = re.search(r"^\s*#\s+(.+?)\s*$", text, re.M)
    return clean_md(m.group(1)) if m else fallback


def snippet(text, limit=300):
    txt = clean_md(body_text(text))
    return txt[:limit] if txt else ""


def read_status(path):
    try:
        with open(path) as fh:
            for line in fh:
                m = re.match(r"\s*status:\s*(.+)", line, re.I)
                if m:
                    return m.group(1).strip().strip('"').split()[0]
    except OSError:
        pass
    return "?"


def phase_for(status):
    s = (status or "").lower()
    if s in {"needs-taste", "tasting", "needs-repitch", "repitching"}:
        return "taste"
    if s in {"needs-review", "reviewing", "needs-revision", "revising"}:
        return "definition"
    if s in {"needs-plan", "planning", "needs-codereview", "codereviewing", "needs-recode", "recoding"}:
        return "code"
    if s in {"needs-run", "running"}:
        return "run"
    if s in {"done"}:
        return "done"
    return "unknown"


def parse_float(text):
    if text is None:
        return None
    m = re.search(r"[-+]?\d+(?:\.\d+)?", clean_md(str(text)))
    return float(m.group(0)) if m else None


def parse_table_row(line):
    cells = [clean_md(c) for c in line.strip().strip("|").split("|")]
    if len(cells) < 2:
        return None
    return cells


def parse_leaderboard():
    txt = load(LEADERBOARD) or ""
    lines = txt.splitlines()
    tiers = []
    i = 0
    while i < len(lines):
        m = re.match(r"^##\s+(.+)$", lines[i])
        if not m:
            i += 1
            continue
        name = clean_md(m.group(1))
        i += 1
        desc_lines = []
        while i < len(lines):
            line = lines[i]
            if line.startswith("## "):
                break
            if line.startswith("|"):
                break
            desc_lines.append(line)
            i += 1
        description = clean_md("\n".join(desc_lines))
        while i < len(lines) and not lines[i].strip():
            i += 1
        if i >= len(lines) or not lines[i].startswith("|"):
            continue
        header = parse_table_row(lines[i])
        i += 1
        if i < len(lines) and re.match(r"^\|\s*[-:\s|]+\|$", lines[i]):
            i += 1
        header_map = {}
        if header:
            for idx, col in enumerate(header):
                key = re.sub(r"[^a-z0-9]+", "", col.lower())
                header_map[key] = idx
        rows = []
        while i < len(lines):
            line = lines[i]
            if line.startswith("## "):
                break
            if not line.strip():
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                if j < len(lines) and lines[j].startswith("|"):
                    i = j
                    continue
                break
            if not line.startswith("|"):
                break
            cells = parse_table_row(line)
            i += 1
            if not cells:
                continue
            rank_idx = header_map.get("#", header_map.get("rank", 0))
            val_idx = header_map.get("valloss", header_map.get("val", 1))
            delta_idx = header_map.get("dvsctrl", header_map.get("delta", None))
            run_idx = header_map.get("run", 3 if len(cells) > 3 else 0)
            summary_idx = header_map.get("summary", 4 if len(cells) > 4 else len(cells) - 1)
            date_idx = header_map.get("date", 5 if len(cells) > 5 else len(cells) - 1)
            if rank_idx is None or rank_idx >= len(cells):
                continue
            rank = parse_float(cells[rank_idx])
            val_loss = parse_float(cells[val_idx]) if val_idx is not None and val_idx < len(cells) else None
            if rank is None or val_loss is None:
                continue
            row = {
                "rank": int(rank),
                "val_loss": val_loss,
                "delta": parse_float(cells[delta_idx]) if delta_idx is not None and delta_idx < len(cells) else None,
                "run": cells[run_idx] if run_idx < len(cells) else "",
                "summary": cells[summary_idx] if summary_idx < len(cells) else "",
                "date": cells[date_idx] if date_idx < len(cells) else "",
            }
            rows.append(row)
        baseline = None
        for row in rows:
            run = (row.get("run") or "").lower()
            if run in {"control", "ctrl", "baseline"} or "control" in run or "baseline" in run:
                baseline = row["val_loss"]
                break
        if baseline is None and rows:
            baseline = rows[0]["val_loss"]
        if baseline is not None:
            for row in rows:
                if row["delta"] is None:
                    row["delta"] = row["val_loss"] - baseline
        tiers.append({"name": name, "description": description, "rows": rows})
    return tiers


def export_ideas():
    ideas = []
    for f in sorted(glob.glob(os.path.join(IDEAS, "*", "idea.md"))):
        d = os.path.dirname(f)
        slug = os.path.basename(d)
        txt = load(f) or ""
        status = read_status(f)
        verdict_path = os.path.join(d, "evidence.md")
        ideas.append({
            "id": slug,
            "name": first_h1(txt, slug),
            "phase": phase_for(status),
            "status": status,
            "summary": snippet(txt),
            "verdict": snippet(load(verdict_path)) if os.path.isfile(verdict_path) else None,
        })
    return ideas


def export(site_dir):
    out_dir = os.path.join(site_dir, "public", "data", "lab")
    os.makedirs(out_dir, exist_ok=True)

    papers = []
    for pj in sorted(glob.glob(os.path.join(PAPERS, "*", "paper.json"))):
        d = os.path.dirname(pj)
        try:
            meta = json.loads(load(pj) or "{}")
        except json.JSONDecodeError:
            continue
        curves = None
        curves_raw = load(os.path.join(d, "figures", "curves.json"))
        if curves_raw:
            try:
                curves = json.loads(curves_raw)
            except json.JSONDecodeError:
                pass
        pid = meta.get("paper_id", os.path.basename(d))
        # Copy build artifacts (PDF from tools/build_paper.py, LaTeX source,
        # figure PNGs) into the site so the research tab can link/embed them.
        assets_dir = os.path.join(out_dir, "papers", pid)
        assets = {"pdf": None, "tex": None, "images": []}
        for src, key in ((os.path.join(d, "paper.pdf"), "pdf"),
                         (os.path.join(d, "latex", "main.tex"), "tex")):
            if os.path.isfile(src):
                os.makedirs(assets_dir, exist_ok=True)
                dst = os.path.join(assets_dir, os.path.basename(src))
                shutil.copyfile(src, dst)
                assets[key] = f"/data/lab/papers/{pid}/{os.path.basename(src)}"
        for src in sorted(glob.glob(os.path.join(d, "figures", "*.png"))):
            os.makedirs(assets_dir, exist_ok=True)
            dst = os.path.join(assets_dir, os.path.basename(src))
            shutil.copyfile(src, dst)
            assets["images"].append(f"/data/lab/papers/{pid}/{os.path.basename(src)}")
        papers.append({
            "id": pid,
            "title": meta.get("title", ""),
            "status": meta.get("status", ""),
            "authors": meta.get("authors", []),
            "created": meta.get("created", ""),
            "experiments": meta.get("experiments", []),
            "manuscript_md": load(os.path.join(d, "manuscript.md")),
            "related_work_md": load(os.path.join(d, "related-work.md")),
            "experiment_plan_md": load(os.path.join(d, "experiment-plan.md")),
            "curves": curves,
            "assets": assets,
        })

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "papers": papers,
        "n_experiments": sum(len(p["experiments"]) for p in papers),
    }
    out = os.path.join(out_dir, "summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=1)
    ideas = {
        "generated_at": summary["generated_at"],
        "ideas": export_ideas(),
    }
    ideas_out = os.path.join(out_dir, "ideas.json")
    with open(ideas_out, "w") as f:
        json.dump(ideas, f, indent=1)
    leaderboard = {
        "generated_at": summary["generated_at"],
        "tiers": parse_leaderboard(),
    }
    leaderboard_out = os.path.join(out_dir, "leaderboard.json")
    with open(leaderboard_out, "w") as f:
        json.dump(leaderboard, f, indent=1)
    for name in ("goals", "problems", "personal"):
        raw = load(os.path.join(ROOT, "lab", f"{name}.json"))
        if raw is None:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            print(f"skip lab/{name}.json: invalid JSON")
            continue
        data["generated_at"] = summary["generated_at"]
        with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
            json.dump(data, f, indent=1)
        print(f"wrote {name}.json")
    print(f"wrote {out}: {len(papers)} papers, {summary['n_experiments']} experiments")
    print(f"wrote {ideas_out}: {len(ideas['ideas'])} ideas")
    print(f"wrote {leaderboard_out}: {sum(len(t['rows']) for t in leaderboard['tiers'])} leaderboard rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--site-dir", default=os.path.join(ROOT, "..", "open-superintelligence-lab-github-io"))
    args = ap.parse_args()
    export(os.path.abspath(args.site_dir))
