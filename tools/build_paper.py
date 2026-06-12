#!/usr/bin/env python3
"""Build paper artifacts for token2science papers.

For each paper dir containing paper.json:
  - render figures/loss-curves.png and figures/loss-curves-zoom.png from
    figures/curves.json when present
  - generate latex/main.tex from manuscript.md + related-work.md
  - run pdflatex twice and copy latex/main.pdf -> paper.pdf

Usage:
  python3 tools/build_paper.py                 # build all papers
  python3 tools/build_paper.py <paper-dir>     # build one paper dir
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
PAPERS_ROOT = ROOT / "token2science" / "papers"
PDFLATEX = Path("/opt/homebrew/bin/pdflatex")

UNICODE_MAP = {
    "—": "---",
    "−": "$-$",
    "±": "$\\pm$",
    "×": "$\\times$",
    "⊙": "$\\odot$",
    "←": "$\\leftarrow$",
    "·": "$\\cdot$",
    "→": "$\\rightarrow$",
}

LATEX_ESCAPE = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
}


@dataclass
class BuildResult:
    paper_id: str
    paper_dir: Path
    ok: bool = True
    steps: List[str] = None
    errors: List[str] = None
    pdf_path: Optional[Path] = None
    pdf_size: Optional[int] = None
    page_count: Optional[int] = None

    def __post_init__(self) -> None:
        if self.steps is None:
            self.steps = []
        if self.errors is None:
            self.errors = []


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def find_paper_dirs(root: Path) -> List[Path]:
    dirs = []
    for paper_json in root.rglob("paper.json"):
        dirs.append(paper_json.parent)
    return sorted(dirs)


def humanize_label(name: str) -> str:
    return name.replace("_", " ").strip().title()


def latex_escape_text(text: str) -> str:
    out: List[str] = []
    for ch in text:
        if ch in UNICODE_MAP:
            out.append(UNICODE_MAP[ch])
        elif ch in LATEX_ESCAPE:
            out.append(LATEX_ESCAPE[ch])
        else:
            out.append(ch)
    return "".join(out)


def inline_code_to_latex(code: str) -> str:
    # Use detokenize to keep code spans robust without adding extra packages.
    return r"\texttt{\detokenize{" + code + "}}"


_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)")


def convert_inline(text: str) -> str:
    parts = re.split(r"(`[^`]*`)", text)
    converted: List[str] = []
    for part in parts:
        if not part:
            continue
        if part.startswith("`") and part.endswith("`"):
            converted.append(inline_code_to_latex(part[1:-1]))
            continue
        escaped = latex_escape_text(part)
        escaped = _BOLD_RE.sub(lambda m: r"\textbf{" + m.group(1) + "}", escaped)
        escaped = _ITALIC_RE.sub(lambda m: r"\emph{" + m.group(1) + "}", escaped)
        converted.append(escaped)
    return "".join(converted)


def strip_top_heading(md: str) -> str:
    lines = md.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and re.match(r"^#\s+", lines[0]):
        lines = lines[1:]
    while lines and not lines[0].strip():
        lines.pop(0)
    return "\n".join(lines)


def parse_author(md: str) -> str:
    m = re.search(r"^\*\*Author:\*\*\s*(.+)$", md, re.M)
    if m:
        return m.group(1).strip()
    return ""


def parse_title(md: str, fallback: str) -> str:
    m = re.search(r"^\s*#\s+(.+?)\s*$", md, re.M)
    return m.group(1).strip() if m else fallback


def is_table_separator(line: str) -> bool:
    stripped = line.strip()
    if not stripped.startswith("|"):
        return False
    stripped = stripped.strip("|").strip()
    if not stripped:
        return False
    parts = [p.strip() for p in stripped.split("|")]
    return all(re.fullmatch(r":?-{3,}:?", p or "") for p in parts)


def split_table_row(line: str) -> List[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def table_to_latex(lines: Sequence[str]) -> str:
    if len(lines) < 2:
        return ""
    header = split_table_row(lines[0])
    body = [split_table_row(row) for row in lines[2:] if row.strip()]
    ncols = max(len(header), *(len(r) for r in body)) if body else len(header)
    header = header + [""] * (ncols - len(header))
    body = [row + [""] * (ncols - len(row)) for row in body]
    colspec = "l" * ncols
    out = [rf"\begin{{tabular}}{{{colspec}}}", r"\toprule"]
    out.append(" & ".join(convert_inline(cell) for cell in header) + r" \\")
    out.append(r"\midrule")
    for row in body:
        out.append(" & ".join(convert_inline(cell) for cell in row) + r" \\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    return "\n".join(out)


def list_block_to_latex(lines: Sequence[str]) -> str:
    out = [r"\begin{itemize}"]
    for line in lines:
        m = re.match(r"^\s*[-*+]\s+(.*)$", line)
        if not m:
            continue
        out.append(r"\item " + convert_inline(m.group(1).strip()))
    out.append(r"\end{itemize}")
    return "\n".join(out)


def code_block_to_latex(lines: Sequence[str]) -> str:
    return "\n".join([r"\begin{verbatim}"] + list(lines) + [r"\end{verbatim}"])


def paragraph_to_latex(lines: Sequence[str]) -> str:
    text = " ".join(s.strip() for s in lines if s.strip())
    return convert_inline(text)


def markdown_to_latex_body(md: str, *, paper_dir: Path, inject_results_figure: bool) -> str:
    lines = strip_top_heading(md).splitlines()
    out: List[str] = []
    i = 0
    abstract_open = False

    def close_abstract() -> None:
        nonlocal abstract_open
        if abstract_open:
            out.append(r"\end{abstract}")
            abstract_open = False

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue

        if stripped.startswith("```"):
            close_abstract()
            code_lines: List[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            if i < len(lines):
                i += 1
            out.append(code_block_to_latex(code_lines))
            continue

        heading = re.match(r"^(#{2,3})\s+(.+?)\s*$", line)
        if heading:
            close_abstract()
            level = len(heading.group(1))
            title = heading.group(2).strip()
            title_lower = title.lower()
            if level == 2 and title_lower == "abstract":
                out.append(r"\begin{abstract}")
                abstract_open = True
                i += 1
                continue
            if level == 2:
                out.append(r"\section{" + convert_inline(title) + "}")
                if inject_results_figure and title_lower == "results":
                    fig = paper_dir / "latex" / "figures" / "loss-curves.png"
                    if fig.is_file():
                        out.append(r"\begin{figure}[htbp]")
                        out.append(r"\centering")
                        out.append(r"\includegraphics[width=\linewidth]{figures/loss-curves.png}")
                        out.append(r"\end{figure}")
            else:
                out.append(r"\subsection{" + convert_inline(title) + "}")
            i += 1
            continue

        if re.match(r"^\s*[-*+]\s+", line):
            close_abstract()
            block: List[str] = [line]
            i += 1
            while i < len(lines) and re.match(r"^\s*[-*+]\s+", lines[i]):
                block.append(lines[i])
                i += 1
            out.append(list_block_to_latex(block))
            continue

        if (
            "|" in line
            and i + 1 < len(lines)
            and is_table_separator(lines[i + 1])
        ):
            close_abstract()
            block = [line, lines[i + 1]]
            i += 2
            while i < len(lines) and lines[i].strip() and "|" in lines[i]:
                block.append(lines[i])
                i += 1
            out.append(table_to_latex(block))
            continue

        close_abstract()
        para: List[str] = [line]
        i += 1
        while i < len(lines):
            nxt = lines[i]
            nxt_stripped = nxt.strip()
            if not nxt_stripped:
                break
            if nxt_stripped.startswith("```"):
                break
            if re.match(r"^(#{2,3})\s+", nxt):
                break
            if re.match(r"^\s*[-*+]\s+", nxt):
                break
            if "|" in nxt and i + 1 < len(lines) and is_table_separator(lines[i + 1]):
                break
            para.append(nxt)
            i += 1
        out.append(paragraph_to_latex(para))

    close_abstract()
    return "\n\n".join(block for block in out if block is not None)


def build_latex_document(paper_dir: Path, paper_meta: dict) -> Path:
    manuscript_path = paper_dir / "manuscript.md"
    related_path = paper_dir / "related-work.md"
    manuscript = read_text(manuscript_path)
    related = read_text(related_path)

    title = parse_title(manuscript, str(paper_meta.get("title", paper_dir.name)))
    author = parse_author(manuscript)
    if not author:
        authors = paper_meta.get("authors", [])
        if isinstance(authors, list):
            author = " \\and ".join(str(a) for a in authors if str(a).strip())
        else:
            author = str(authors)
    author_parts = [part.strip() for part in author.split(r"\and") if part.strip()]
    if len(author_parts) > 1:
        author_latex = " \\and ".join(convert_inline(part) for part in author_parts)
    else:
        author_latex = convert_inline(author)

    latex_dir = paper_dir / "latex"
    figures_dir = latex_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    body_parts: List[str] = []
    if manuscript.strip():
        body_parts.append(
            markdown_to_latex_body(manuscript, paper_dir=paper_dir, inject_results_figure=True)
        )
    if related.strip():
        related_body = strip_top_heading(related)
        if related_body.strip():
            body_parts.append(r"\section{Related Work}")
            body_parts.append(markdown_to_latex_body(related_body, paper_dir=paper_dir, inject_results_figure=False))

    tex = rf"""\documentclass[11pt,a4paper]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{hyperref}}
\usepackage{{xcolor}}
\hypersetup{{colorlinks=true,linkcolor=blue,urlcolor=blue,citecolor=blue}}
\title{{{convert_inline(title)}}}
\author{{{author_latex}}}
\date{{}}
\begin{{document}}
\maketitle

{chr(10).join(body_parts)}

\end{{document}}
"""
    tex_path = latex_dir / "main.tex"
    write_text(tex_path, tex)
    return tex_path


def render_curve_figures(paper_dir: Path) -> List[Path]:
    curves_path = paper_dir / "figures" / "curves.json"
    if not curves_path.is_file():
        return []

    try:
        curves = load_json(curves_path)
    except Exception as exc:
        raise RuntimeError(f"invalid curves.json: {exc}") from exc

    steps = curves.get("steps") or []
    series = curves.get("series") or {}
    finals = curves.get("finals") or {}
    noise_band = curves.get("noise_band", None)
    if not isinstance(steps, list) or not isinstance(series, dict) or not steps:
        raise RuntimeError("curves.json missing steps or series")

    fig_dir = paper_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    def draw(out_path: Path, xlim_start: Optional[float] = None) -> None:
        plt.rcParams.update(
            {
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                "savefig.facecolor": "white",
                "font.size": 10,
            }
        )
        fig, ax = plt.subplots(figsize=(6.8, 4.4), dpi=150)
        plotted = []
        names = list(series.keys())
        names.sort(key=lambda n: (0 if n == "baseline" else 1, n))
        for name in names:
            ys = series.get(name) or []
            if not isinstance(ys, list) or not ys:
                continue
            n = min(len(steps), len(ys))
            xs = steps[:n]
            ys = ys[:n]
            label = humanize_label(name)
            final = finals.get(name)
            if final is not None:
                try:
                    label = f"{label} ({float(final):.4f})"
                except Exception:
                    pass
            ax.plot(xs, ys, linewidth=2.0 if name == "baseline" else 1.8, label=label)
            plotted.append(name)
        if not plotted:
            plt.close(fig)
            raise RuntimeError("no plottable series found")
        ax.set_xlabel("training step")
        ax.set_ylabel("val loss")
        title = str(curves.get("tier", "")).strip()
        if title:
            ax.set_title(title)
        ax.grid(True, alpha=0.22)
        ax.legend(frameon=False)
        if noise_band is not None:
            try:
                nb = float(noise_band)
            except Exception:
                nb = None
            if nb is not None and "baseline" in series and isinstance(series["baseline"], list):
                base = series["baseline"]
                n = min(len(steps), len(base))
                if n:
                    ax.fill_between(
                        steps[:n],
                        [float(v) - nb for v in base[:n]],
                        [float(v) + nb for v in base[:n]],
                        alpha=0.08,
                        color="black",
                        linewidth=0,
                    )
        if xlim_start is not None:
            ax.set_xlim(left=xlim_start)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    full_path = fig_dir / "loss-curves.png"
    zoom_path = fig_dir / "loss-curves-zoom.png"
    draw(full_path)
    if len(steps) >= 4:
        draw(zoom_path, xlim_start=steps[3])
    elif full_path.exists():
        shutil.copy2(full_path, zoom_path)
    return [full_path, zoom_path] if zoom_path.exists() else [full_path]


def copy_figures_to_latex(paper_dir: Path, rendered: Sequence[Path]) -> None:
    latex_figures = paper_dir / "latex" / "figures"
    latex_figures.mkdir(parents=True, exist_ok=True)
    for src in rendered:
        if src.is_file():
            shutil.copy2(src, latex_figures / src.name)


def run_pdflatex(latex_dir: Path) -> Tuple[bool, str]:
    if not PDFLATEX.is_file():
        return False, f"pdflatex not found at {PDFLATEX}"
    cmd = [str(PDFLATEX), "-interaction=nonstopmode", "main.tex"]
    last_output = ""
    for _ in range(2):
        proc = subprocess.run(
            cmd,
            cwd=latex_dir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        last_output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    pdf_path = latex_dir / "main.pdf"
    return pdf_path.is_file(), last_output


def pdf_page_count(pdf_path: Path) -> Optional[int]:
    pdfinfo = shutil.which("pdfinfo")
    if not pdfinfo:
        return None
    proc = subprocess.run([pdfinfo, str(pdf_path)], capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    for line in proc.stdout.splitlines():
        if line.startswith("Pages:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except Exception:
                return None
    return None


def copy_artifacts(paper_dir: Path) -> Optional[Path]:
    src = paper_dir / "latex" / "main.pdf"
    if not src.is_file():
        return None
    dst = paper_dir / "paper.pdf"
    shutil.copy2(src, dst)
    return dst


def build_one(paper_dir: Path) -> BuildResult:
    paper_json = paper_dir / "paper.json"
    paper_id = paper_dir.name
    result = BuildResult(paper_id=paper_id, paper_dir=paper_dir)

    try:
        paper_meta = load_json(paper_json)
    except Exception as exc:
        result.ok = False
        result.errors.append(f"paper.json: {exc}")
        return result

    try:
        rendered = render_curve_figures(paper_dir)
        if rendered:
            copy_figures_to_latex(paper_dir, rendered)
            result.steps.append(f"figures={len(rendered)}")
    except Exception as exc:
        result.ok = False
        result.errors.append(f"figures: {exc}")

    try:
        tex_path = build_latex_document(paper_dir, paper_meta)
        result.steps.append(f"tex={tex_path.name}")
    except Exception as exc:
        result.ok = False
        result.errors.append(f"latex: {exc}")
        return result

    try:
        ok, output = run_pdflatex(paper_dir / "latex")
        if not ok:
            result.ok = False
            result.errors.append("pdflatex failed")
            tail = "\n".join(output.splitlines()[-20:])
            if tail.strip():
                eprint(f"[{paper_id}] pdflatex tail:\n{tail}")
        else:
            result.steps.append("pdflatex=ok")
    except Exception as exc:
        result.ok = False
        result.errors.append(f"pdflatex: {exc}")

    try:
        pdf_path = copy_artifacts(paper_dir)
        if pdf_path and pdf_path.is_file():
            result.pdf_path = pdf_path
            result.pdf_size = pdf_path.stat().st_size
            result.page_count = pdf_page_count(pdf_path)
            result.steps.append(f"pdf={result.pdf_size}B")
        else:
            result.ok = False
            result.errors.append("pdf copy missing")
    except Exception as exc:
        result.ok = False
        result.errors.append(f"pdf copy: {exc}")

    return result


def format_summary(result: BuildResult) -> str:
    status = "ok" if result.ok else "fail"
    size = f"{result.pdf_size}B" if result.pdf_size is not None else "n/a"
    pages = f"{result.page_count}p" if result.page_count is not None else "np"
    errs = f" errors={len(result.errors)}" if result.errors else ""
    return f"{result.paper_id}: {status} pdf={size} pages={pages}{errs}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paper_dir", nargs="?", help="paper dir or omit to build all")
    args = parser.parse_args(argv)

    if args.paper_dir:
        candidate = Path(args.paper_dir).resolve()
        if candidate.is_file() and candidate.name == "paper.json":
            paper_dirs = [candidate.parent]
        else:
            paper_dirs = [candidate]
    else:
        paper_dirs = find_paper_dirs(PAPERS_ROOT)

    results: List[BuildResult] = []
    for paper_dir in paper_dirs:
        if not (paper_dir / "paper.json").is_file():
            print(f"{paper_dir.name}: skip (no paper.json)")
            continue
        result = build_one(paper_dir)
        results.append(result)
        print(format_summary(result))
        if result.errors:
            for err in result.errors:
                eprint(f"[{result.paper_id}] {err}")

    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
