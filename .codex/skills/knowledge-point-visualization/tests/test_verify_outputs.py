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
    eqs = []
    for i in tags:
        eqs.append("<div class=\"eq\">\\[ x_{}. \\tag{{{}}} \\]</div>".format(i, i))
        eqs.append("<p>解释第 {} 式为什么成立。</p>".format(i))
    tag_text = "\n".join(eqs)
    html = """<!DOCTYPE html>
<html><head><meta charset='UTF-8'>
<script>window.MathJax = {{}};</script>
<script src='mathjax/tex-svg.js'></script>
<script src='interactive-components.js'></script>
</head><body>
<section id="evolution"><h2>0 研究背景</h2><table><thead><tr><th>阶段</th><th>内容</th></tr></thead><tbody><tr><td>① 方法</td><td>以前怎么做</td></tr><tr><td>② 缺陷</td><td>为什么不够</td></tr><tr><td>③ 创新</td><td>本文如何解决</td></tr></tbody></table></section>
<section id="toolbox"><h2>1 基础工具箱</h2><table><thead><tr><th>概念</th><th>解释</th></tr></thead><tbody><tr><td>a</td><td>b</td></tr><tr><td>c</td><td>d</td></tr><tr><td>e</td><td>f</td></tr><tr><td>g</td><td>h</td></tr><tr><td>i</td><td>j</td></tr></tbody></table></section>
<section id="symbols"><h2>2 符号总表</h2><a href="#symbols">符号表</a><table><thead><tr><th>符号</th><th>意义</th><th>为什么需要</th></tr></thead><tbody><tr><td>x</td><td>a</td><td>b</td></tr><tr><td>y</td><td>c</td><td>d</td></tr><tr><td>z</td><td>e</td><td>f</td></tr><tr><td>w</td><td>g</td><td>h</td></tr><tr><td>v</td><td>i</td><td>j</td></tr></tbody></table></section>
<section id="core"><h2>3 核心公式</h2><div class="why-card"><p>为什么这样优化</p><p>为什么不是 SGD</p><p>为什么梯度置零成立</p><p>代价是什么</p></div><div class="viz" data-formula="tag1"><p class="viz-caption">这张图帮助理解公式 (1)</p></div>
<div class="step"><span class="num">1</span><h3>公式推导<span class="src-badge" data-src="optimization">Optimization</span></h3><div class="numeric-example">微型例子：1 2 3</div>
{tag_text}
<div data-demo='gradient-descent'></div>
{derivation_extra}
</div></section>
<section id="why"><h2>为什么卡片</h2><p>为什么这样优化 为什么不是 SGD 为什么梯度置零成立 代价是什么</p></section>
<section id="experiments"><h2>4 实验验证</h2><table><thead><tr><th>对应公式</th><th>验证假设</th><th>实验结果</th><th>结论</th></tr></thead><tbody><tr><td>a</td><td>b</td><td>c</td><td>d</td></tr><tr><td>e</td><td>f</td><td>g</td><td>h</td></tr></tbody></table></section>
<section id="appendix"><h2>附录</h2><a href="knowledge-graph.html">知识图谱</a><a href="learning-path.json">学习路径</a></section>
<section id="formulas"><h2>公式索引</h2><table><thead><tr><th>编号</th><th>含义</th></tr></thead><tbody></tbody></table></section>
<section id="check"><h2>覆盖自检</h2><table><thead><tr><th>公式</th><th>状态</th></tr></thead><tbody><tr><td>(1)</td><td>✅</td></tr><tr><td>(2)</td><td>✅</td></tr></tbody></table></section>
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


def test_missing_tutor_section(tmp_path):
    paper, out = make_dir(tmp_path)
    html = (out / "derivation.html").read_text(encoding="utf-8").replace('id="evolution"', 'id="not-evolution"')
    write(out / "derivation.html", html)
    r = run(paper, out)
    assert r.returncode == 1
    assert "missing teaching section: evolution" in r.stdout


def test_consecutive_equations_without_explanation(tmp_path):
    paper, out = make_dir(tmp_path, derivation_extra="<p>\\[a=b\\]</p><p>\\[a=c\\]</p>")
    r = run(paper, out)
    assert r.returncode == 1
    assert "consecutive equations" in r.stdout


def test_consecutive_equations_with_explanation_ok(tmp_path):
    paper, out = make_dir(
        tmp_path,
        derivation_extra="<p>\\[a=b\\]</p><p>这里解释从第一步到第二步的代数变换。</p><p>\\[a=c\\]</p>",
    )
    r = run(paper, out)
    assert r.returncode == 0


def test_missing_why_card(tmp_path):
    paper, out = make_dir(tmp_path)
    html = (out / "derivation.html").read_text(encoding="utf-8").replace("why-card", "plain-card")
    write(out / "derivation.html", html)
    r = run(paper, out)
    assert r.returncode == 1
    assert "why-card" in r.stdout


def test_why_card_missing_questions(tmp_path):
    paper, out = make_dir(tmp_path)
    html = (out / "derivation.html").read_text(encoding="utf-8").replace("梯度置零", "梯度清零")
    write(out / "derivation.html", html)
    r = run(paper, out)
    assert r.returncode == 1
    assert "why-card missing questions" in r.stdout


def test_missing_numeric_example(tmp_path):
    paper, out = make_dir(tmp_path)
    html = (out / "derivation.html").read_text(encoding="utf-8").replace("numeric-example", "text-block")
    write(out / "derivation.html", html)
    r = run(paper, out)
    assert r.returncode == 1
    assert "numeric-example" in r.stdout


def test_tagged_step_missing_numeric_example(tmp_path):
    paper, out = make_dir(tmp_path)
    html = (out / "derivation.html").read_text(encoding="utf-8")
    html = html.replace('<div class="numeric-example">微型例子：1 2 3</div>', "", 1)
    html = html.replace(
        '<section id="experiments">',
        '<section id="experiments"><div class="numeric-example">1 2 3</div>',
        1,
    )
    write(out / "derivation.html", html)
    r = run(paper, out)
    assert r.returncode == 1
    assert "tagged step missing numeric-example" in r.stdout


def test_viz_without_caption(tmp_path):
    paper, out = make_dir(tmp_path)
    html = (out / "derivation.html").read_text(encoding="utf-8").replace("这张图帮助理解公式 (1)", "一张示意图")
    write(out / "derivation.html", html)
    r = run(paper, out)
    assert r.returncode == 1
    assert "viz missing caption" in r.stdout


def test_viz_more_than_one_per_section(tmp_path):
    paper, out = make_dir(
        tmp_path,
        derivation_extra="<div class='viz' data-formula='tag2'><p class='viz-caption'>第二张图帮助理解公式 (2)</p></div>",
    )
    r = run(paper, out)
    assert r.returncode == 1
    assert "more than 1 visualization" in r.stdout


def test_coverage_table_missing_tag(tmp_path):
    paper, out = make_dir(tmp_path)
    html = (out / "derivation.html").read_text(encoding="utf-8").replace("<tr><td>(2)</td>", "<tr><td>(99)</td>")
    write(out / "derivation.html", html)
    r = run(paper, out)
    assert r.returncode == 1
    assert "coverage table" in r.stdout


def test_missing_src_badge(tmp_path):
    paper, out = make_dir(tmp_path)
    html = (out / "derivation.html").read_text(encoding="utf-8").replace('data-src="optimization"', "")
    write(out / "derivation.html", html)
    r = run(paper, out)
    assert r.returncode == 1
    assert "missing formula source badges" in r.stdout


def test_definition_badge_without_question(tmp_path):
    paper, out = make_dir(tmp_path)
    html = (out / "derivation.html").read_text(encoding="utf-8").replace(
        "</h3>",
        '<span class="src-badge" data-src="definition">Definition</span></h3>',
        1,
    )
    write(out / "derivation.html", html)
    r = run(paper, out)
    assert r.returncode == 1
    assert "why-card missing questions for definition" in r.stdout


def test_definition_question_present_ok(tmp_path):
    paper, out = make_dir(tmp_path)
    html = (out / "derivation.html").read_text(encoding="utf-8")
    html = html.replace(
        "<h3>公式推导",
        '<h3>公式推导<span class="src-badge" data-src="definition">Definition</span>',
        1,
    )
    html = html.replace(
        '<div class="why-card">',
        '<div class="why-card"><p>为什么这样定义，它描述什么对象</p>',
        1,
    )
    write(out / "derivation.html", html)
    r = run(paper, out)
    assert r.returncode == 0


def test_toolbox_too_many_rows(tmp_path):
    paper, out = make_dir(tmp_path)
    html = (out / "derivation.html").read_text(encoding="utf-8")
    extra = "".join("<tr><td>x{}</td><td>y</td></tr>".format(i) for i in range(4))
    html = html.replace("<tr><td>i</td><td>j</td></tr>", "<tr><td>i</td><td>j</td></tr>" + extra, 1)
    write(out / "derivation.html", html)
    r = run(paper, out)
    assert r.returncode == 1
    assert "toolbox rows" in r.stdout


def test_experiments_missing_theory_link(tmp_path):
    paper, out = make_dir(tmp_path)
    html = (out / "derivation.html").read_text(encoding="utf-8").replace("<th>对应公式</th>", "<th>指标</th>")
    write(out / "derivation.html", html)
    r = run(paper, out)
    assert r.returncode == 1
    assert "experiments missing theory link" in r.stdout


def test_matrix_example_missing_explicit_representation(tmp_path):
    paper, out = make_dir(
        tmp_path,
        derivation_extra='<div class="eq">\\[ Y^\\top C Y. \\tag{3} \\]</div><p>解释。</p>',
    )
    r = run(paper, out)
    assert r.returncode == 1
    assert "matrix example missing explicit representation" in r.stdout


def test_matrix_example_explicit_ok(tmp_path):
    paper, out = make_dir(
        tmp_path,
        derivation_extra='<div class="eq">\\[ Y^\\top C Y. \\tag{3} \\]</div><p>解释。</p>',
    )
    html = (out / "derivation.html").read_text(encoding="utf-8")
    html = html.replace(
        '<div class="numeric-example">微型例子：1 2 3</div>',
        '<div class="numeric-example">Y∈R^{2×1}=[1; 0.5]；第一步：CY=[4; 0.5]；计算 Yᵀ(CY)=1×4+0.5×0.5=4.25</div>',
        1,
    )
    write(out / "derivation.html", html)
    r = run(paper, out)
    assert r.returncode == 0


def test_matrix_example_latex_ok(tmp_path):
    paper, out = make_dir(
        tmp_path,
        derivation_extra='<div class="eq">\\[ Y^\\top C Y. \\tag{3} \\]</div><p>解释。</p>',
    )
    html = (out / "derivation.html").read_text(encoding="utf-8")
    html = html.replace(
        '<div class="numeric-example">微型例子：1 2 3</div>',
        '<div class="numeric-example">\\(Y\\in\\mathbb{R}^{2\\times1}=\\begin{bmatrix}1\\\\0.5\\end{bmatrix}\\)；第一步：\\(CY=\\begin{bmatrix}4\\\\0.5\\end{bmatrix}\\)；计算 \\(Y^\\top(CY)=1\\times4+0.5\\times0.5=4.25\\)</div>',
        1,
    )
    write(out / "derivation.html", html)
    r = run(paper, out)
    assert r.returncode == 0
