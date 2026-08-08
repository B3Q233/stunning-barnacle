import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "verify_outputs.py"

PAPER = """# Demo Paper

\\[ a = b. \\tag{1} \\]
\\[ c = d. \\tag{2} \\]
"""

JS_FIXTURE = """(function () {
  var KPV = window.KPV = window.KPV || {};
  var demos = {};
  function register(id, factory) { demos[id] = factory; }
  register("gradient-descent", function () {});
  KPV.register = register; KPV.demos = demos;
})();
"""


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def minimal_json(name):
    if name == "all.json":
        return {"nodes": [], "edges": [], "topo": [], "proof": {"initial": [], "states": {}}, "defaults": {}}
    if name == "learning-path.json":
        return {"target": "x", "path": [], "stats": {}}
    return {"ok": True}


def make_dir(tmp_path, derivation_extra="", js=JS_FIXTURE, tags=(1, 2)):
    out = tmp_path / "out"
    paper = tmp_path / "paper.md"
    write(paper, PAPER)
    for name in ["meta.json", "all.json", "learning-cost.json", "user-state.json", "learning-path.json"]:
        write(out / name, json.dumps(minimal_json(name), ensure_ascii=False))
    for name in ["knowledge-graph.html", "cost-editor.html"]:
        write(out / name, "<!DOCTYPE html><html><head><meta charset='UTF-8'></head><body></body></html>")
    write(out / "mathjax" / "tex-svg.js", "/* mathjax stub */")
    write(out / "interactive-components.js", js)
    tag_text = " ".join("\\[ x_{}. \\tag{{{}}} \\]".format(i, i) for i in tags)
    html = """<!DOCTYPE html>
<html><head><meta charset='UTF-8'>
<script>window.MathJax = {{}};</script>
<script src='mathjax/tex-svg.js'></script>
<script src='interactive-components.js'></script>
</head><body>
{tag_text}
<div data-demo='gradient-descent'></div>
{derivation_extra}
</body></html>""".format(tag_text=tag_text, derivation_extra=derivation_extra)
    write(out / "derivation.html", html)
    return paper, out


def run(paper, out, *args):
    return subprocess.run([sys.executable, str(SCRIPT), str(paper), str(out), *args],
                          capture_output=True, text=True, stdin=subprocess.DEVNULL,
                          timeout=60)


def test_all_pass(tmp_path):
    paper, out = make_dir(tmp_path)
    r = run(paper, out)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "VERIFY PASSED" in r.stdout


def test_missing_file(tmp_path):
    paper, out = make_dir(tmp_path)
    (out / "all.json").unlink()
    r = run(paper, out)
    assert r.returncode == 1
    assert "missing required file: all.json" in r.stdout


def test_invalid_json(tmp_path):
    paper, out = make_dir(tmp_path)
    write(out / "all.json", "{bad")
    r = run(paper, out)
    assert r.returncode == 1
    assert "all.json: invalid JSON" in r.stdout


def test_missing_mathjax_script(tmp_path):
    paper, out = make_dir(tmp_path)
    html = (out / "derivation.html").read_text(encoding="utf-8").replace("mathjax/tex-svg.js", "cdn.js")
    write(out / "derivation.html", html)
    r = run(paper, out)
    assert r.returncode == 1
    assert "missing relative MathJax script" in r.stdout


def test_dollar_delimiter(tmp_path):
    paper, out = make_dir(tmp_path)
    html = (out / "derivation.html").read_text(encoding="utf-8") + "<p>$x^2$</p>"
    write(out / "derivation.html", html)
    r = run(paper, out)
    assert r.returncode == 1
    assert "dollar-delimited" in r.stdout


def test_unbalanced_delims(tmp_path):
    paper, out = make_dir(tmp_path)
    html = (out / "derivation.html").read_text(encoding="utf-8") + "<p>\\(x</p>"
    write(out / "derivation.html", html)
    r = run(paper, out)
    assert r.returncode == 1
    assert "unbalanced" in r.stdout


def test_formula_coverage_missing(tmp_path):
    paper, out = make_dir(tmp_path, tags=(1,))
    r = run(paper, out)
    assert r.returncode == 1
    assert "missing formulas" in r.stdout


def test_formula_override(tmp_path):
    paper, out = make_dir(tmp_path, tags=(1,))
    r = run(paper, out, "--formulas", "1")
    assert r.returncode == 0


def test_skip_phrase_without_ref(tmp_path):
    paper, out = make_dir(tmp_path, derivation_extra="<p>同理可得</p>")
    r = run(paper, out)
    assert r.returncode == 1
    assert "skip phrase" in r.stdout


def test_skip_phrase_with_ref_ok(tmp_path):
    paper, out = make_dir(tmp_path, derivation_extra="<p>同理可得（与第 3 步相同，见式(5)）</p>")
    r = run(paper, out)
    assert r.returncode == 0


def test_unknown_demo(tmp_path):
    paper, out = make_dir(tmp_path, derivation_extra="<div data-demo='nope'></div>")
    r = run(paper, out)
    assert r.returncode == 1
    assert "unknown demo" in r.stdout


def test_browser_check_missing_browser(tmp_path):
    paper, out = make_dir(tmp_path)
    r = run(paper, out, "--browser-check", "--browser", str(tmp_path / "no-such-browser.exe"))
    assert r.returncode == 1
    assert "no Edge/Chrome found" in r.stdout
