#!/usr/bin/env python3
"""Verify knowledge-point-visualization outputs (spec 2026-08-08).

Usage:
    python scripts/verify_outputs.py <paper.md> <out_dir> \
        [--formulas 1,2,17] [--browser-check] [--browser PATH]

Checks:
  1. Required output files exist and JSON parses with required fields.
  2. derivation.html MathJax setup: UTF-8 charset, window.MathJax config,
     relative mathjax/tex-svg.js script, no $...$ delimiters, balanced
     \\(...\\) and \\[...\\] delimiters.
  3. Formula coverage: paper \\tag{N} set == derivation \\tag{N} set.
  4. No skip phrases without a nearby step/formula reference.
  5. Every data-demo id exists in interactive-components.js registry.
  6. --browser-check: headless Edge/Chrome opens derivation.html; page must
     write data-mathjax="ok" and data-kpv-ready="ok".
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REQUIRED_FILES = [
    "meta.json", "all.json", "learning-cost.json", "user-state.json",
    "learning-path.json",
    "knowledge-graph.html", "cost-editor.html", "derivation.html",
    "mathjax/tex-svg.js", "interactive-components.js",
]
JSON_MIN_FIELDS = {
    "all.json": ["nodes", "edges", "topo", "proof", "defaults"],
    "learning-path.json": ["target", "path", "stats"],
}
TAG_RE = re.compile(r"\\tag\{(\d+)\}")
DOLLAR_RE = re.compile(r"(?<!\\)\$(?:\$)?[^$\n]+(?<!\\)\$")
SKIP_PHRASES = ["完整推导见扩展版", "推导略", "详见附录", "同理可得", "推导从略"]
REF_RE = re.compile(r"第\s*\d+\s*步|式\s*[（(]?\s*\d+|\\tag\{\d+\}")
DEMO_RE = re.compile(r"""data-demo=["']([^"']+)["']""")
REGISTRY_RE = re.compile(r'register\(\s*"([^"]+)"\s*,')

DEFAULT_BROWSERS = [
    os.environ.get("KPV_BROWSER", ""),
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/msedge",
]


def find_browser(explicit: str | None) -> str | None:
    if explicit:
        # An explicitly requested path is authoritative: fail rather than
        # silently falling back to a default browser.
        return explicit if Path(explicit).is_file() else None
    for cand in DEFAULT_BROWSERS:
        if cand and Path(cand).is_file():
            return cand
    return None


def read(path: Path) -> str:
    # utf-8-sig transparently strips a leading BOM if present, and is a
    # no-op for BOM-less UTF-8 files.
    return path.read_text(encoding="utf-8-sig")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paper", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--formulas", default=None, help="comma list, e.g. 1,2,17")
    ap.add_argument("--browser-check", action="store_true")
    ap.add_argument("--browser", default=None)
    ap.add_argument("--no-skip-refs", action="store_true")
    args = ap.parse_args()

    errors: list[str] = []
    out = args.out_dir
    paper_ok = args.paper.is_file()
    if not paper_ok:
        errors.append(f"paper not found: {args.paper}")

    # Check 1: required files + JSON fields.
    for rel in REQUIRED_FILES:
        p = out / rel
        if not p.is_file():
            errors.append(f"missing required file: {rel}")
            continue
        if p.suffix == ".json":
            try:
                data = json.loads(read(p))
            except Exception as e:
                errors.append(f"{rel}: invalid JSON ({e})")
                continue
            for field in JSON_MIN_FIELDS.get(rel, []):
                if field not in data:
                    errors.append(f"{rel}: missing field {field!r}")

    derivation = out / "derivation.html"
    if derivation.is_file():
        html = read(derivation)
        # Check 2: MathJax setup.
        if not re.search(r'charset=["\']UTF-8["\']', html, re.IGNORECASE):
            errors.append("derivation.html: missing UTF-8 charset")
        if "window.MathJax" not in html:
            errors.append("derivation.html: missing window.MathJax config")
        if not re.search(r'src=["\']mathjax/tex-svg\.js["\']', html):
            errors.append("derivation.html: missing relative MathJax script")
        if not re.search(r'src=["\']interactive-components\.js["\']', html):
            errors.append("derivation.html: missing interactive-components.js script")
        for dollar in DOLLAR_RE.findall(html):
            errors.append(f"derivation.html: dollar-delimited math found: {dollar[:40]!r}")
        for open_d, close_d in (("\\(", "\\)"), ("\\[", "\\]")):
            if html.count(open_d) != html.count(close_d):
                errors.append(
                    f"derivation.html: unbalanced {open_d!r}/{close_d!r} "
                    f"({html.count(open_d)} vs {html.count(close_d)})"
                )
        # Check 3: formula coverage.
        paper_tags: set[str] = set()
        if paper_ok:
            paper_tags = set(TAG_RE.findall(read(args.paper)))
        if args.formulas:
            paper_tags = {t.strip() for t in args.formulas.split(",")}
        html_tags = set(TAG_RE.findall(html))
        missing = paper_tags - html_tags
        if missing:
            errors.append(f"derivation.html: missing formulas {sorted(missing, key=int)}")
        # Check 4: skip phrases.
        for phrase in SKIP_PHRASES:
            for m in re.finditer(re.escape(phrase), html):
                if args.no_skip_refs:
                    errors.append(f"derivation.html: banned phrase {phrase!r}")
                    continue
                window_text = html[max(0, m.start() - 40): m.end() + 80]
                if not REF_RE.search(window_text):
                    errors.append(f"derivation.html: skip phrase {phrase!r} without reference")
        # Check 5: demo registry.
        js_file = out / "interactive-components.js"
        if js_file.is_file():
            registered = set(REGISTRY_RE.findall(read(js_file)))
            for did in DEMO_RE.findall(html):
                if did not in registered:
                    errors.append(f"derivation.html: unknown demo {did!r}")
    else:
        errors.append("derivation.html: missing (skipped content checks)")

    # Check 6: browser-level render verification.
    if args.browser_check and not errors and derivation.is_file():
        browser = find_browser(args.browser)
        if browser is None:
            errors.append("browser-check requested but no Edge/Chrome found")
        else:
            with tempfile.TemporaryDirectory(prefix="kpv-browser-") as td:
                cmd = [
                    browser, "--headless=new", "--disable-gpu",
                    "--no-first-run", "--no-default-browser-check",
                    f"--user-data-dir={td}",
                    "--virtual-time-budget=15000",
                    "--dump-dom", derivation.resolve().as_uri(),
                ]
                try:
                    dom = subprocess.run(cmd, capture_output=True, text=True,
                                         timeout=90).stdout
                except Exception as e:
                    dom = ""
                    errors.append(f"browser check crashed: {e}")
                if 'data-mathjax="ok"' not in dom:
                    errors.append("browser-check: MathJax did not finish typesetting")
                if 'data-kpv-ready="ok"' not in dom:
                    errors.append("browser-check: demos did not initialize cleanly")

    if errors:
        print("VERIFY FAILED")
        for e in errors:
            print("  - " + e)
        return 1
    print("VERIFY PASSED: files, mathjax, formulas, steps, demos"
          + (", browser-render" if args.browser_check else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
