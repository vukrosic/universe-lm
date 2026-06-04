"""Render the Q-gain/K-gain tutorial (README.md + images/) to a styled PDF.

markdown -> HTML (+CSS) -> PDF via the weasyprint CLI. Images resolve because
the intermediate HTML is written next to README.md (relative `images/...`).

Run from repo root:  python docs/tutorials/qk_gain/make_pdf.py
Output:              docs/tutorials/qk_gain/qk_gain.pdf
"""
import os
import subprocess

import markdown

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get("QK_GAIN_SRC", os.path.join(HERE, "README.md"))
PDF = os.environ.get("QK_GAIN_PDF", os.path.join(HERE, "qk_gain.pdf"))
PDF_BASENAME = os.path.splitext(os.path.basename(PDF))[0]
HTML = os.path.join(HERE, f"_{PDF_BASENAME}.html")

CSS = """
@page { size: A4; margin: 1.8cm 2cm; }
@page { @bottom-center { content: counter(page) " / " counter(pages);
        font-family: -apple-system, sans-serif; font-size: 9px; color: #999; } }
* { box-sizing: border-box; }
body { font-family: "PingFang SC", "Hiragino Sans GB", "Noto Sans CJK SC",
       "Microsoft YaHei", -apple-system, "Helvetica Neue", Arial, sans-serif;
       font-size: 11.5px; line-height: 1.55; color: #1a1a1a; }
h1 { font-size: 25px; line-height: 1.2; margin: 0 0 .2em; color: #111;
     border-bottom: 3px solid #2b6cb0; padding-bottom: .3em; }
h2 { font-size: 17px; margin: 1.4em 0 .5em; color: #2b6cb0;
     page-break-after: avoid; border-bottom: 1px solid #e2e8f0; padding-bottom: .2em; }
h3 { font-size: 13.5px; margin: 1em 0 .4em; color: #2c5282; page-break-after: avoid; }
p { margin: .55em 0; }
a { color: #2b6cb0; text-decoration: none; }
strong { color: #111; }
img { max-width: 100%; display: block; margin: 1em auto; border: 1px solid #e2e8f0;
      border-radius: 6px; page-break-inside: avoid; }
pre { background: #f7fafc; border: 1px solid #e2e8f0; border-left: 3px solid #2b6cb0;
      border-radius: 5px; padding: .7em .9em; margin: .8em 0; overflow-x: hidden;
      white-space: pre-wrap; word-wrap: break-word; page-break-inside: avoid; }
pre code { font-family: "SF Mono", "Menlo", monospace; font-size: 10px; color: #2d3748;
           background: none; padding: 0; }
p code, li code { font-family: "SF Mono", "Menlo", monospace; font-size: 10px;
       background: #edf2f7; padding: .1em .35em; border-radius: 3px; color: #c53030; }
table { border-collapse: collapse; width: 100%; margin: .9em 0; font-size: 10.5px;
        page-break-inside: avoid; }
th, td { border: 1px solid #cbd5e0; padding: .4em .6em; text-align: left; }
th { background: #2b6cb0; color: #fff; }
tr:nth-child(even) td { background: #f7fafc; }
hr { border: none; border-top: 1px solid #e2e8f0; margin: 1.6em 0; }
ul, ol { margin: .5em 0 .5em 1.2em; }
li { margin: .25em 0; }
.byline { color: #718096; font-size: 11px; margin: .2em 0 1.4em; }
"""

with open(SRC, encoding="utf-8") as f:
    md_text = f.read()

body = markdown.markdown(
    md_text,
    extensions=["fenced_code", "tables", "sane_lists", "attr_list"],
)

html = (f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<style>{CSS}</style></head><body>{body}</body></html>")

with open(HTML, "w", encoding="utf-8") as f:
    f.write(html)

subprocess.run(["weasyprint", HTML, PDF], check=True)
os.remove(HTML)
print(f"wrote {PDF} ({os.path.getsize(PDF) // 1024} KB)")
