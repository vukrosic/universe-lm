#!/usr/bin/env python3
"""Build styled HTML from markdown files for FIRE post.

Embeds local images as base64 data URIs so weasyprint / headless Chrome
can resolve them in a single pass without filesystem path issues.
"""
import base64
import re
from pathlib import Path
import markdown

POSTS = Path("/Users/vukrosic/my-life/llm-research-kit-scaling/posts")
EXPORT = POSTS / "export"
EXPORT.mkdir(parents=True, exist_ok=True)

CSS = """
@page { size: A4; margin: 18mm 16mm; }
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial,
               "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
  font-size: 15px;
  line-height: 1.6;
  color: #1a1a1a;
  max-width: 720px;
  margin: 0 auto;
  padding: 24px 16px;
  word-wrap: break-word;
}
h1 {
  font-size: 26px;
  line-height: 1.25;
  margin: 0 0 8px 0;
  font-weight: 700;
}
h2 {
  font-size: 19px;
  line-height: 1.3;
  margin: 28px 0 10px 0;
  font-weight: 700;
}
h3 {
  font-size: 16px;
  margin: 18px 0 8px 0;
  font-weight: 600;
}
p { margin: 0 0 12px 0; }
.author { color: #555; margin: 0 0 16px 0; font-size: 14px; }
img {
  max-width: 100%;
  height: auto;
  display: block;
  margin: 12px auto;
  border-radius: 4px;
}
code {
  font-family: "SF Mono", Menlo, Consolas, "Courier New", monospace;
  font-size: 13px;
  background: #f4f4f4;
  padding: 1px 4px;
  border-radius: 3px;
}
pre {
  background: #f6f8fa;
  border: 1px solid #e1e4e8;
  border-radius: 6px;
  padding: 12px 14px;
  overflow-x: auto;
  font-size: 12.5px;
  line-height: 1.5;
}
pre code { background: transparent; padding: 0; font-size: 12.5px; }
hr {
  border: none;
  border-top: 1px solid #e1e4e8;
  margin: 24px 0;
}
a { color: #0969da; text-decoration: none; }
a:hover { text-decoration: underline; }
.cta-block {
  margin-top: 18px;
  padding: 0;
}
.cta-link {
  display: inline-block;
  margin-top: 6px;
  font-weight: 500;
}
"""


def image_to_data_uri(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".svg": "image/svg+xml",
    }.get(suffix, "image/png")
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def inline_local_images(html: str, base_dir: Path) -> str:
    # find <img src="..."> where src is a relative path
    pattern = re.compile(r'(<img\s+[^>]*src=")([^"]+)("[^>]*>)', re.IGNORECASE)

    def repl(m):
        prefix, src, suffix = m.group(1), m.group(2), m.group(3)
        if src.startswith(("data:", "http://", "https://", "file://")):
            return m.group(0)
        img_path = (base_dir / src).resolve()
        if not img_path.exists():
            print(f"  WARN: image not found: {img_path}")
            return m.group(0)
        data_uri = image_to_data_uri(img_path)
        return f"{prefix}{data_uri}{suffix}"

    return pattern.sub(repl, html)


def build_html(md_path: Path, html_path: Path) -> None:
    md_text = md_path.read_text(encoding="utf-8")
    md = markdown.Markdown(
        extensions=["extra", "sane_lists", "smarty"],
        output_format="html5",
    )
    body = md.convert(md_text)
    body = inline_local_images(body, base_dir=md_path.parent)
    full = (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        f"<title>{md_path.stem}</title>\n"
        f"<style>{CSS}</style>\n"
        "</head>\n<body>\n"
        f"{body}\n"
        "</body>\n</html>\n"
    )
    html_path.write_text(full, encoding="utf-8")
    print(f"wrote {html_path} ({len(full)} bytes)")


def main():
    targets = [
        (POSTS / "2026-06-10-fire-positional-encoding.md",
         EXPORT / "fire-en.html"),
        (POSTS / "2026-06-10-fire-positional-encoding.zh.md",
         EXPORT / "fire-zh.html"),
    ]
    for md_path, html_path in targets:
        if not md_path.exists():
            raise SystemExit(f"missing: {md_path}")
        build_html(md_path, html_path)


if __name__ == "__main__":
    main()
