# KPV 可视化技能优化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 knowledge-point-visualization 技能每次运行都能稳定产出“公式全覆盖、逐步推导、通俗解释、可交互演示、MathJax 真实渲染”的推导页与完整交付物，无需用户反复迭代。

**Architecture:** 在保留现有 JSON 管线（meta/all/learning-cost/learning-path）不动的前提下，新增三样东西：一个机器可验证的交付门 `scripts/verify_outputs.py`；一个离线交互演示库 `assets/interactive-components.js`；一个推导页标准骨架 `assets/derivation-template.html`。SKILL.md 增加第 5.5 步（生成推导页）与第 7 步（校验门），并把“面向高中生通俗解释”与“离线约束”固化为硬性规范。

**Tech Stack:** Python 3（校验脚本 + pytest 测试）、原生 JavaScript（交互演示，无任何外部依赖）、HTML/CSS（模板）、无头 Edge/Chrome（浏览器级渲染验证）。

## Global Constraints

- 交付物（论文目录内，缺一即失败）：`meta.json`、`all.json`、`learning-cost.json`、`user-state.json`、`learning-path.json`、`knowledge-graph.html`、`cost-editor.html`、`derivation.html`、`mathjax/tex-svg.js`、`interactive-components.js`。
- 校验 6 项：产物齐全 / MathJax 生效 / 公式全覆盖（`\tag{N}` 集合与论文一致）/ 无跳步词（“完整推导见扩展版、推导略、详见附录、同理可得”无引用时失败）/ demo 注册完整 / 浏览器级渲染（`data-mathjax="ok"` 且 `data-kpv-ready="ok"`）。
- 推导页读者定位：高中生。每个概念 = “一句话 + 生活例子 + 最小公式 + 论文位置”，禁止默认读者有大学数学背景。
- 离线约束：页面只允许相对路径本地引用（`mathjax/tex-svg.js`、`interactive-components.js`），禁止 CDN、fetch、外部字体。
- 不改 `scripts/build_graph.py`、`scripts/plan_path.py`、JSON schema；`knowledge-graph.html`、`cost-editor.html` 保持从 assets 拷贝。
- 代码文件一律 UTF-8；HTML 必须有 `<meta charset="UTF-8">`。

---

## File Structure

| 文件 | 职责 |
|---|---|
| `scripts/verify_outputs.py` | 交付门：6 项机器校验（含无头浏览器渲染验证） |
| `tests/test_verify_outputs.py` | verify_outputs.py 的 pytest 测试 |
| `assets/interactive-components.js` | 离线交互演示库：注册表 + 5 个内置 demo + 就绪标记 |
| `tests/interactive-components.test.js` | 组件库的 Node 冒烟测试（注册表完整性 + initAll 空场景） |
| `assets/derivation-template.html` | 推导页标准骨架：MathJax、样式、导航、步骤卡、公式索引、覆盖自检、就绪脚本 |
| `SKILL.md`（修改） | 输出契约、第 5.5 步、第 7 步、基础概念强化、离线约束 |
| `agents/openai.yaml`（修改） | 默认提示词包含新产物与校验门 |

接口约定（任务间依赖）：
- Task 2 产出 `register("<demo-id>", factory)` 注册表；Task 3 模板使用 `data-demo="gradient-descent"`。
- Task 1 产出 `verify_outputs.py <paper.md> <out_dir> [--formulas ...] [--browser-check] [--browser PATH]`；Task 5 使用它验收。
- 就绪标记约定（Task 2/3 共同遵守）：页面在 `<html>` 上写 `data-mathjax`、`data-demos-total`、`data-demos-ok`、`data-demos-failed`、`data-kpv-ready`。

---

### Task 1: 交付门 `scripts/verify_outputs.py`

**Files:**
- Create: `G:\Idea\.codex\skills\knowledge-point-visualization\scripts\verify_outputs.py`
- Test: `G:\Idea\.codex\skills\knowledge-point-visualization\tests\test_verify_outputs.py`

**Interfaces:**
- Consumes: 论文 Markdown 路径、输出目录路径（命令行参数）。
- Produces: `main() -> int`；CLI：`python scripts/verify_outputs.py <paper.md> <out_dir> [--formulas 1,2,17] [--browser-check] [--browser PATH]`；退出码 0 = 全绿，1 = 失败并打印 `FAIL` 明细。

- [ ] **Step 1: 写失败测试**

创建 `tests/test_verify_outputs.py`（内容如下，pytest 可直接运行）：

```python
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
<script>window.MathJax = {};</script>
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
                          capture_output=True, text=True)


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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_verify_outputs.py -q`（在 `G:\Idea\.codex\skills\knowledge-point-visualization` 下）
Expected: FAIL，`ModuleNotFoundError` / `No module named 'verify_outputs'`（脚本尚不存在）。

- [ ] **Step 3: 实现 `scripts/verify_outputs.py`**

创建文件，内容如下：

```python
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
DEMO_RE = re.compile(r'data-demo="([^"]+)"')
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
    for cand in ([explicit] if explicit else []) + DEFAULT_BROWSERS:
        if cand and Path(cand).is_file():
            return cand
    return None


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


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
        if 'charset="UTF-8"' not in html and "charset=utf-8" not in html.lower():
            errors.append("derivation.html: missing UTF-8 charset")
        if "window.MathJax" not in html:
            errors.append("derivation.html: missing window.MathJax config")
        if 'src="mathjax/tex-svg.js"' not in html:
            errors.append("derivation.html: missing relative MathJax script")
        if 'src="interactive-components.js"' not in html:
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_verify_outputs.py -q`
Expected: 12 passed。

- [ ] **Step 5: 提交**

```bash
git add .codex/skills/knowledge-point-visualization/scripts/verify_outputs.py .codex/skills/knowledge-point-visualization/tests/test_verify_outputs.py
git commit -m "feat(kpv): 新增 verify_outputs.py 交付门（6 项校验 + 浏览器渲染验证）"
```

---

### Task 2: 交互演示库 `assets/interactive-components.js`

**Files:**
- Create: `G:\Idea\.codex\skills\knowledge-point-visualization\assets\interactive-components.js`
- Test: `G:\Idea\.codex\skills\knowledge-point-visualization\tests\interactive-components.test.js`

**Interfaces:**
- Consumes: 无（独立库）。
- Produces: `KPV.register(id, factory)`、`KPV.initAll(root) -> {total, ok, failed}`、`KPV.demos`、`KPV.helpers`；内置 demo id：`gradient-descent`、`projection-box`、`implicit-function`、`sgld-sampling`、`matrix-decomposition`；`window.__KPV_DEMOS__`。

- [ ] **Step 1: 写失败测试**

创建 `tests/interactive-components.test.js`：

```js
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const src = fs.readFileSync(path.join(__dirname, "..", "assets", "interactive-components.js"), "utf8");
const sandbox = {
  window: {},
  document: { querySelectorAll: () => [] },
  performance: { now: () => 0 },
  requestAnimationFrame: () => 0,
  cancelAnimationFrame: () => {},
};
sandbox.window.window = sandbox.window;
vm.createContext(sandbox);
vm.runInContext(src, sandbox);

const KPV = sandbox.window.KPV;
if (!KPV || typeof KPV.register !== "function" || typeof KPV.initAll !== "function") {
  throw new Error("KPV API missing");
}
const expected = ["gradient-descent", "projection-box", "implicit-function", "sgld-sampling", "matrix-decomposition"];
for (const id of expected) {
  if (!KPV.demos[id]) throw new Error("missing demo: " + id);
}
const res = KPV.initAll({ querySelectorAll: () => [] });
if (res.total !== 0 || res.ok !== 0 || res.failed.length !== 0) {
  throw new Error("initAll empty case failed: " + JSON.stringify(res));
}
console.log("interactive-components OK: " + expected.join(", "));
```

- [ ] **Step 2: 运行测试确认失败**

Run: `node tests/interactive-components.test.js`
Expected: FAIL，`ENOENT`（文件不存在）。

- [ ] **Step 3: 实现 `assets/interactive-components.js`**

创建文件，内容如下（离线、纯原生 JS、无 CDN/fetch）：

```js
/* KPV 交互演示库 —— 离线、纯原生 JS、无 CDN/fetch。
   用法: <div data-demo="gradient-descent" data-config='{"lr":0.25}'></div>
   注册: KPV.register(id, factory(container, cfg, H))
   就绪: KPV.initAll(root) -> {total, ok, failed}；window.__KPV_DEMOS__ */
(function () {
  "use strict";
  var KPV = window.KPV = window.KPV || {};
  var demos = {};
  var NS = "http://www.w3.org/2000/svg";

  function el(tag, attrs, children) {
    var node = document.createElement(tag);
    attrs = attrs || {};
    if (attrs["class"]) { node.className = attrs["class"]; delete attrs["class"]; }
    if (attrs.html) { node.innerHTML = attrs.html; delete attrs.html; }
    for (var k in attrs) node.setAttribute(k, attrs[k]);
    (children || []).forEach(function (c) {
      if (c == null) return;
      node.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    });
    return node;
  }
  function svgEl(tag, attrs) {
    var node = document.createElementNS(NS, tag);
    for (var k in (attrs || {})) node.setAttribute(k, attrs[k]);
    return node;
  }
  function svgBox(w, h) {
    return svgEl("svg", { width: w, height: h, viewBox: "0 0 " + w + " " + h });
  }
  function slider(label, min, max, step, value, onChange) {
    var wrap = el("label", { "class": "kpy-ctl" });
    wrap.appendChild(el("span", { "class": "kpy-ctl-label" }, [label + " "]));
    var input = el("input", { type: "range", min: String(min), max: String(max),
                              step: String(step), value: String(value) });
    var out = el("output", { "class": "kpy-ctl-value" }, [String(value)]);
    input.addEventListener("input", function () {
      out.textContent = input.value;
      onChange(parseFloat(input.value));
    });
    wrap.appendChild(input); wrap.appendChild(out);
    return wrap;
  }
  function playPause(onTick) {
    var running = false, raf = null, last = 0;
    var btn = el("button", { "class": "kpy-btn" }, ["播放"]);
    btn.addEventListener("click", function () {
      running = !running;
      btn.textContent = running ? "暂停" : "播放";
      if (running) { last = performance.now(); loop(); }
      else if (raf) { cancelAnimationFrame(raf); raf = null; }
    });
    function loop() {
      raf = requestAnimationFrame(function (t) {
        if (!running) return;
        var dt = Math.min(64, t - last); last = t;
        onTick(dt / 1000);
        loop();
      });
    }
    return btn;
  }
  function tooltip(container, text) {
    var tip = el("div", { "class": "kpy-tip", html: text });
    tip.style.display = "none";
    container.appendChild(tip);
    container.addEventListener("mouseenter", function () { tip.style.display = "block"; });
    container.addEventListener("mouseleave", function () { tip.style.display = "none"; });
    return tip;
  }
  function info(container, what, how, formula) {
    container.appendChild(el("div", { "class": "kpy-info" }, [
      el("div", {}, [el("b", {}, ["这是什么："]), document.createTextNode(what)]),
      el("div", {}, [el("b", {}, ["怎么玩："]), document.createTextNode(how)]),
      el("div", { html: "对应公式：" + formula }),
    ]));
  }
  function register(id, factory) { demos[id] = factory; }
  function initAll(root) {
    var results = { total: 0, ok: 0, failed: [] };
    var nodes = (root || document).querySelectorAll("[data-demo]");
    Array.prototype.forEach.call(nodes, function (node) {
      results.total++;
      var id = node.getAttribute("data-demo");
      var cfg = {};
      try { cfg = JSON.parse(node.getAttribute("data-config") || "{}"); }
      catch (e) {
        results.failed.push(id + ":bad-config");
        node.setAttribute("data-demo-error", "bad-config");
        return;
      }
      try {
        var factory = demos[id];
        if (!factory) throw new Error("unknown demo: " + id);
        factory(node, cfg, { el: el, svgEl: svgEl, svg: svgBox, slider: slider,
                             playPause: playPause, tooltip: tooltip, info: info });
        node.classList.add("kpy-demo-ok");
        results.ok++;
      } catch (e) {
        node.classList.add("kpy-demo-error");
        node.setAttribute("data-demo-error", String((e && e.message) || e));
        results.failed.push(id + ": " + ((e && e.message) || e));
      }
    });
    window.__KPV_DEMOS__ = results;
    return results;
  }

  /* ---------- 梯度下降 ---------- */
  register("gradient-descent", function (c, cfg) {
    var lr0 = cfg.lr || 0.25, x0 = cfg.x0 || 2.0, y0 = cfg.y0 || 1.6;
    var W = 300, H = 240, S = 42, cx = W / 2, cy = H / 2;
    var svg = svgBox(W, H);
    [0.5, 1, 2, 4, 8].forEach(function (lv) {
      var d = [], r = Math.sqrt(lv);
      for (var i = 0; i <= 64; i++) {
        var a = i * 2 * Math.PI / 64;
        var x = cx + r * S * Math.cos(a), y = cy - r * S * Math.sin(a);
        d.push((i ? "L" : "M") + x.toFixed(1) + " " + y.toFixed(1));
      }
      svg.appendChild(svgEl("path", { d: d.join(" "), fill: "none", stroke: "#c7cde8" }));
    });
    var trail = svgEl("polyline", { fill: "none", stroke: "#4f46e5", "stroke-width": 2 });
    var dot = svgEl("circle", { r: 5, fill: "#dc2626" });
    svg.appendChild(trail); svg.appendChild(dot);
    var x = x0, y = y0, pts = [];
    function draw() {
      trail.setAttribute("points", pts.map(function (p) {
        return (cx + p[0] * S).toFixed(1) + " " + (cy - p[1] * S).toFixed(1);
      }).join(" "));
      dot.setAttribute("cx", (cx + x * S).toFixed(1));
      dot.setAttribute("cy", (cy - y * S).toFixed(1));
    }
    function step() {
      x -= lr0 * 2 * x; y -= lr0 * 2 * y;
      pts.push([x, y]); if (pts.length > 80) pts.shift();
      draw();
    }
    c.appendChild(el("div", { "class": "kpy-demo-title" }, ["梯度下降：沿着最陡的方向下山"]));
    info(c, "函数 f(x,y)=x²+y² 像一个碗，最低点在 (0,0)。梯度指向上升最陡的方向，取负梯度就是下坡最陡的方向。",
      "拖动学习率滑杆改变步长；点“播放”看小球从起点滚到碗底。",
      "\\widetilde{\\mathbf M}^{(t+1)}=\\mathrm{Proj}_{\\mathbb M}(\\widetilde{\\mathbf M}^{(t)}+s_t\\nabla R)");
    c.appendChild(svg);
    var lrCtl = slider("学习率", 0.02, 0.6, 0.01, lr0, function (v) { lr0 = v; });
    var resetBtn = el("button", { "class": "kpy-btn" }, ["重置"]);
    resetBtn.addEventListener("click", function () { x = x0; y = y0; pts = []; draw(); });
    c.appendChild(el("div", { "class": "kpy-ctls" }, [lrCtl, playPause(step), resetBtn]));
    tooltip(c, "红点 = 当前参数；蓝线 = 走过的路；圆圈是等高线，越靠内越低。");
    draw();
  });

  /* ---------- 投影 / clamp ---------- */
  register("projection-box", function (c, cfg) {
    var px0 = cfg.x || 1.5, py0 = cfg.y || 0.8;
    var W = 300, H = 240, S = 60, cx = W / 2, cy = H / 2;
    function X(x) { return cx + x * S; }
    function Y(y) { return cy - y * S; }
    function clamp(v) { return Math.max(-1, Math.min(1, v)); }
    var svg = svgBox(W, H);
    svg.appendChild(svgEl("rect", { x: X(-1), y: Y(1), width: 2 * S, height: 2 * S,
                                    fill: "#eef2ff", stroke: "#4338ca" }));
    var line = svgEl("line", { stroke: "#94a3b8", "stroke-dasharray": "4 3" });
    var pDot = svgEl("circle", { r: 5, fill: "#dc2626" });
    var qDot = svgEl("circle", { r: 5, fill: "#16a34a" });
    svg.appendChild(line); svg.appendChild(pDot); svg.appendChild(qDot);
    function draw() {
      var qx = clamp(px0), qy = clamp(py0);
      line.setAttribute("x1", X(px0)); line.setAttribute("y1", Y(py0));
      line.setAttribute("x2", X(qx)); line.setAttribute("y2", Y(qy));
      pDot.setAttribute("cx", X(px0)); pDot.setAttribute("cy", Y(py0));
      qDot.setAttribute("cx", X(qx)); qDot.setAttribute("cy", Y(qy));
    }
    c.appendChild(el("div", { "class": "kpy-demo-title" }, ["投影 / clamp：把点截回可行域"]));
    info(c, "蓝色方框是可行域（|x|≤1, |y|≤1）。方框外的点不能直接用，投影就是把它拉回方框内最近的位置。",
      "拖动滑杆移动红点，绿点就是投影结果。",
      "\\mathrm{Proj}_{\\mathbb M}(\\widetilde{\\mathbf M})=\\mathrm{clamp}(\\widetilde{\\mathbf M},\\pm\\Lambda)");
    c.appendChild(svg);
    c.appendChild(el("div", { "class": "kpy-ctls" }, [
      slider("x", -2, 2, 0.05, px0, function (v) { px0 = v; draw(); }),
      slider("y", -2, 2, 0.05, py0, function (v) { py0 = v; draw(); }),
    ]));
    tooltip(c, "绿点 = Proj 后的点：x、y 分别被截到 [−1,1] 内。");
    draw();
  });

  /* ---------- 隐函数求导 ---------- */
  register("implicit-function", function (c, cfg) {
    var W = 300, H = 240, S = 80, cx = W / 2, cy = H / 2;
    var ang = 0.6;
    var svg = svgBox(W, H);
    svg.appendChild(svgEl("circle", { r: S, fill: "none", stroke: "#4338ca" }));
    var tangent = svgEl("line", { stroke: "#16a34a", "stroke-width": 2 });
    var dot = svgEl("circle", { r: 5, fill: "#dc2626" });
    svg.appendChild(tangent); svg.appendChild(dot);
    function draw() {
      var x = Math.cos(ang), y = Math.sin(ang);
      var X = cx + x * S, Y = cy - y * S;
      var slope = (Math.abs(y) < 1e-6) ? 1e6 : -x / y;
      var dx = Math.sqrt(1 + slope * slope) || 1;
      var nx = X - dx * 60, ny = Y - slope * dx * 60;
      tangent.setAttribute("x1", nx); tangent.setAttribute("y1", ny);
      tangent.setAttribute("x2", 2 * X - nx); tangent.setAttribute("y2", 2 * Y - ny);
      dot.setAttribute("cx", X); dot.setAttribute("cy", Y);
    }
    c.appendChild(el("div", { "class": "kpy-demo-title" }, ["隐函数求导：曲线上的点怎么跟着动"]));
    info(c, "圆 x²+y²=1 上，y 不是 x 的显式函数，但每一点的切线斜率仍能算：两边对 x 求导得 2x+2yy′=0，所以 y′=−x/y。",
      "拖动滑杆沿圆移动红点，绿切线随点变化。",
      "F(\\Theta,\\widetilde{\\mathbf M})=0 \\Rightarrow \\frac{\\partial\\Theta}{\\partial\\widetilde{\\mathbf M}}=-(\\partial F/\\partial\\Theta)^{-1}(\\partial F/\\partial\\widetilde{\\mathbf M})");
    c.appendChild(svg);
    c.appendChild(el("div", { "class": "kpy-ctls" }, [
      slider("角度", 0, 6.28, 0.02, ang, function (v) { ang = v; draw(); }),
    ]));
    tooltip(c, "绿线是切线，斜率 = −x/y；在 (0,±1) 处切线竖直。");
    draw();
  });

  /* ---------- SGLD 采样 ---------- */
  register("sgld-sampling", function (c, cfg) {
    var noise = cfg.noise || 0.5;
    var W = 320, H = 200, pad = 24;
    var mu = 0.5;
    var svg = svgBox(W, H);
    var dots = svgEl("g", {});
    svg.appendChild(dots);
    function SX(x) { return pad + (x + 1.5) * (W - 2 * pad) / 4; }
    function postY(x) { var s2 = 1 + 1 / (noise * noise + 0.001); return Math.exp(-(x - mu) * (x - mu) / (2 * s2)); }
    var maxY = postY(mu);
    function SY(y) { return H - pad - (y / maxY) * (H - 2 * pad); }
    var curve = svgEl("path", { fill: "none", stroke: "#4f46e5", "stroke-width": 2 });
    svg.appendChild(curve);
    function drawCurve() {
      var d = [];
      for (var i = 0; i <= 60; i++) {
        var x = (i / 60) * 4 - 1.5;
        d.push((i ? "L" : "M") + SX(x).toFixed(1) + " " + SY(postY(x)).toFixed(1));
      }
      curve.setAttribute("d", d.join(" "));
    }
    var theta = mu;
    function step() {
      var g = (theta - mu) / (1 + 1 / (noise * noise + 0.001));
      theta = theta - 0.05 * g + noise * 0.22 * (Math.random() * 2 - 1);
      dots.appendChild(svgEl("circle", { r: 2, fill: "#dc2626",
        cx: SX(theta).toFixed(1), cy: SY(postY(theta)).toFixed(1) }));
      while (dots.childNodes.length > 300) dots.removeChild(dots.firstChild);
    }
    c.appendChild(el("div", { "class": "kpy-demo-title" }, ["SGLD：带噪声的梯度上升 ≈ 从后验采样"]));
    info(c, "贝叶斯后验告诉我们参数大概在哪，但没法直接算。SGLD 的做法：沿着后验梯度往上走（更像后验），同时每步加一点随机噪声（才能在周围散布开）。",
      "调大噪声滑杆看样本散布变大；点“播放”收集样本。",
      "\\widetilde{\\mathbf M}^{(t+1)}=\\widetilde{\\mathbf M}^{(t)}+\\frac{s_t}{2}\\nabla\\log p+\\varepsilon_t");
    c.appendChild(svg);
    c.appendChild(el("div", { "class": "kpy-ctls" }, [
      slider("噪声", 0.1, 1.5, 0.05, noise, function (v) { noise = v; drawCurve(); }),
      playPause(step),
    ]));
    tooltip(c, "蓝线 = 真实后验密度；红点 = SGLD 采出的样本。");
    drawCurve();
  });

  /* ---------- SVD 分解 ---------- */
  register("matrix-decomposition", function (c, cfg) {
    var s2 = 1.0;
    c.appendChild(el("div", { "class": "kpy-demo-title" }, ["SVD：A = UΣVᵀ"]));
    info(c, "任意矩阵 A 都能分解成 UΣVᵀ：U、V 像旋转，Σ 像拉伸，奇异值就是拉伸倍数。",
      "拖动滑杆改变第二个奇异值 σ₂，观察 A 如何变化。",
      "\\mathbf X=\\mathbf U\\boldsymbol\\Sigma\\mathbf V^\\top");
    function cell(v, tip) {
      var td = el("td", {}, [v.toFixed(2)]);
      tooltip(td, tip);
      return td;
    }
    function row(vals, tips) {
      var tr = el("tr", {});
      vals.forEach(function (v, i) { tr.appendChild(cell(v, tips[i])); });
      return tr;
    }
    var aTbl = el("table", { "class": "kpy-grid" });
    var uTbl = el("table", { "class": "kpy-grid" });
    var sTbl = el("table", { "class": "kpy-grid" });
    var vtTbl = el("table", { "class": "kpy-grid" });
    var wrap = el("div", { "class": "kpy-matrix-row" });
    wrap.appendChild(uTbl); wrap.appendChild(sTbl); wrap.appendChild(vtTbl);
    wrap.appendChild(el("span", { "class": "kpy-eq-symbol" }, ["="]));
    wrap.appendChild(aTbl);
    c.appendChild(wrap);
    function render() {
      var av = [0.8 * s2, 0.6, 0.6 * s2, -0.8];
      var tipsA = ["a₁₁ = 0.8σ₂", "a₁₂ = 0.6", "a₂₁ = 0.6σ₂", "a₂₂ = −0.8"];
      aTbl.innerHTML = "";
      aTbl.appendChild(row(av.slice(0, 2), tipsA.slice(0, 2)));
      aTbl.appendChild(row(av.slice(2, 4), tipsA.slice(2, 4)));
      uTbl.innerHTML = ""; uTbl.appendChild(row([0.8, 0.6], ["U 第一列", "U 第二列"]));
      uTbl.appendChild(row([0.6, -0.8], ["U 第二行", "U 第二行"]));
      sTbl.innerHTML = ""; sTbl.appendChild(row([1, 0], ["σ₁=1", "0"]));
      sTbl.appendChild(row([0, s2], ["0", "σ₂ 由滑杆控制"]));
      vtTbl.innerHTML = ""; vtTbl.appendChild(row([0.8, -0.6], ["Vᵀ 第一行", "Vᵀ 第一行"]));
      vtTbl.appendChild(row([0.6, 0.8], ["Vᵀ 第二行", "Vᵀ 第二行"]));
    }
    c.appendChild(el("div", { "class": "kpy-ctls" }, [
      slider("σ₂", 0, 3, 0.05, s2, function (v) { s2 = v; render(); }),
    ]));
    tooltip(c, "A = UΣVᵀ：改变 σ₂ 时 A 的第一列同步缩放。");
    render();
  });

  KPV.register = register; KPV.initAll = initAll; KPV.demos = demos;
  KPV.helpers = { el: el, svgEl: svgEl, svg: svgBox, slider: slider,
                  playPause: playPause, tooltip: tooltip, info: info };
})();
```

- [ ] **Step 4: 运行测试确认通过**

Run: `node tests/interactive-components.test.js`
Expected: `interactive-components OK: gradient-descent, projection-box, implicit-function, sgld-sampling, matrix-decomposition`

- [ ] **Step 5: 提交**

```bash
git add .codex/skills/knowledge-point-visualization/assets/interactive-components.js .codex/skills/knowledge-point-visualization/tests/interactive-components.test.js
git commit -m "feat(kpv): 新增离线交互演示库 interactive-components.js（5 个内置 demo）"
```

---

### Task 3: 推导页标准模板 `assets/derivation-template.html`

**Files:**
- Create: `G:\Idea\.codex\skills\knowledge-point-visualization\assets\derivation-template.html`

**Interfaces:**
- Consumes: Task 2 的 `interactive-components.js`（模板引用 `data-demo="gradient-descent"`）。
- Produces: 一个可直接打开的 HTML 骨架；页面 `<html>` 元素在加载完成后写入 `data-mathjax`、`data-demos-total`、`data-demos-ok`、`data-demos-failed`、`data-kpv-ready`。

- [ ] **Step 1: 创建模板**

创建 `assets/derivation-template.html`，内容如下（AI 生成推导页时以此为骨架填内容，不重写结构/样式）：

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>论文推导可视化（Knowledge Point Visualization）</title>
<script>
window.MathJax = {
  tex: { inlineMath: [['\\(','\\)']], displayMath: [['\\[','\\]']] },
  svg: { fontCache: 'global' }
};
</script>
<script src="mathjax/tex-svg.js"></script>
<script src="interactive-components.js"></script>
<style>
:root{--bg:#f5f6fa;--card:#fff;--ink:#20242f;--muted:#5b6472;--line:#e2e5ee;--accent:#4f46e5;--green:#16a34a;--blue:#0ea5e9}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:"Segoe UI","Microsoft YaHei",system-ui,sans-serif;line-height:1.75}
.wrap{max-width:1060px;margin:0 auto;padding:0 20px}
header{background:linear-gradient(135deg,#312e81,#4f46e5 55%,#0ea5e9);color:#fff;padding:30px 0 24px}
header h1{margin:0 0 8px;font-size:1.5rem}
nav{margin:12px 0 0}
nav a{color:#fff;background:rgba(255,255,255,.16);border:1px solid rgba(255,255,255,.35);border-radius:999px;padding:3px 14px;margin-right:8px;text-decoration:none;font-size:.85rem;display:inline-block;margin-bottom:6px}
section{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:22px 26px;margin:20px 0}
h2{margin:0 0 12px;font-size:1.18rem;border-left:5px solid var(--accent);padding-left:10px}
h3{margin:18px 0 8px;font-size:1rem;color:#312e81}
.eq{background:#fafbff;border:1px dashed #c7cde8;border-radius:10px;padding:10px 16px;margin:10px 0;overflow-x:auto}
.step{background:#fff;border:1px solid var(--line);border-radius:12px;padding:16px 20px;margin:14px 0}
.step .num{display:inline-block;background:var(--accent);color:#fff;border-radius:999px;width:26px;height:26px;line-height:26px;text-align:center;font-size:.85rem;margin-right:8px}
.step h3{display:inline;border:0;padding:0;margin:0}
.goal{background:#eef7ff;border-left:4px solid var(--blue);padding:8px 12px;border-radius:8px;margin:8px 0;font-size:.92rem}
.micro{background:#fbfcfe;border-left:3px solid var(--accent);padding:8px 12px;margin:8px 0;border-radius:6px;font-size:.93rem}
.micro .why{color:var(--muted);font-size:.88rem}
.tag{display:inline-block;background:#eef0fe;color:#4338ca;border:1px solid #d6daf6;border-radius:6px;padding:0 8px;font-size:.76rem;margin:2px 3px 2px 0}
.demo{border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:12px 0;background:#fafbff}
.kpy-demo-title{font-weight:600;margin-bottom:6px}
.kpy-info{font-size:.88rem;color:var(--muted);margin-bottom:8px}
.kpy-ctls{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-top:8px}
.kpy-ctl{display:inline-flex;gap:6px;align-items:center;font-size:.85rem}
.kpy-btn{background:var(--accent);color:#fff;border:0;border-radius:6px;padding:5px 12px;cursor:pointer}
.kpy-tip{position:absolute;z-index:10;background:#20242f;color:#fff;padding:6px 10px;border-radius:6px;font-size:.8rem;max-width:280px}
.kpy-demo-error{outline:2px solid #dc2626}
.kpy-grid{border-collapse:collapse;margin:4px}
.kpy-grid td{border:1px solid var(--line);padding:6px 10px;text-align:center;min-width:44px}
.kpy-matrix-row{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.kpy-eq-symbol{font-size:1.3rem;color:var(--muted)}
.ok{color:var(--green);font-weight:600}
table{width:100%;border-collapse:collapse;font-size:.87rem;margin:10px 0}
th,td{border:1px solid var(--line);padding:7px 10px;text-align:left;vertical-align:top}
th{background:#f1f3f9}
code{background:#eef1f7;border-radius:5px;padding:1px 5px;font-family:Consolas,monospace;font-size:.85em}
footer{color:var(--muted);font-size:.82rem;padding:18px 0 34px;text-align:center}
</style>
</head>
<body>
<header><div class="wrap">
  <h1>论文推导可视化</h1>
  <p>公式全覆盖 · 逐步推导 · 通俗解释 · 交互演示</p>
  <nav>
    <a href="knowledge-graph.html">知识图谱</a>
    <a href="cost-editor.html">学习代价编辑器</a>
    <a href="#toolbox">基础工具箱</a>
    <a href="#chain">推导主线</a>
    <a href="#formulas">公式索引</a>
    <a href="#check">覆盖自检</a>
  </nav>
</div></header>

<main class="wrap">

<section id="toolbox">
  <h2>1.0 基础工具箱（高中生水平）</h2>
  <p class="muted">公式里出现的每个概念都能在这里找到“一句话 + 生活例子 + 最小公式 + 论文位置”。</p>
  <table>
    <thead><tr><th>概念</th><th>通俗解释（生活例子）</th><th>最小公式</th><th>论文哪里用到</th></tr></thead>
    <tbody>
      <tr><td>梯度</td><td>“蒙眼下山”：每一步都往脚底最陡的方向迈，梯度就是那个方向。</td>
          <td>\(\nabla f=(\partial f/\partial x,\partial f/\partial y)\)</td><td>梯度下降、PGA 更新</td></tr>
      <tr><td>投影/clamp</td><td>把跑出操场围栏的点拉回围栏内最近的位置。</td>
          <td>\(\mathrm{clamp}(x,a,b)=\min(b,\max(a,x))\)</td><td>可行域约束</td></tr>
    </tbody>
  </table>
</section>

<section id="chain">
  <h2>2. 推导主线（Step by Step）</h2>
  <div class="step" id="step1">
    <span class="num">1</span><h3>示例步骤：问题设定</h3>
    <div class="goal">目标：用少量观测恢复完整矩阵。</div>
    <div class="micro">
      <div class="eq">\[ \min_{\mathbf X}\|\mathcal R_\Omega(\mathbf M-\mathbf X)\|_F^2,\quad \mathrm{s.t.}\ \mathrm{rank}(\mathbf X)\le k. \tag{1} \]</div>
      <div class="why">为什么：让观测位置的误差最小，同时限制矩阵不太复杂（低秩）。</div>
    </div>
    <p><span class="tag">c_matrix_completion</span></p>
    <div class="demo" data-demo="gradient-descent" data-config='{"lr":0.25,"x0":2.0,"y0":1.6}'></div>
  </div>
</section>

<section id="formulas">
  <h2>3. 公式索引</h2>
  <table>
    <thead><tr><th>编号</th><th>含义</th><th>所在步骤</th></tr></thead>
    <tbody>
      <tr><td>(1)</td><td>矩阵补全目标函数</td><td><a href="#step1">步骤 1</a></td></tr>
    </tbody>
  </table>
</section>

<section id="check">
  <h2>4. 覆盖自检</h2>
  <table>
    <thead><tr><th>公式/概念</th><th>所在步骤</th><th>工具箱位置</th></tr></thead>
    <tbody>
      <tr><td>(1) 矩阵补全</td><td>步骤 1</td><td>工具箱：投影</td></tr>
    </tbody>
  </table>
  <p class="ok">✓ 全部覆盖，无跳步。</p>
</section>

</main>
<footer>由 knowledge-point-visualization 技能生成 · MathJax 离线渲染 · 交互演示离线可用</footer>

<script>
(function () {
  var mj = "missing", finished = false;
  function finish() {
    if (finished) return;
    finished = true;
    document.documentElement.setAttribute("data-mathjax", mj);
    var demos = window.KPV ? KPV.initAll(document) : { total: 0, ok: 0, failed: ["KPV missing"] };
    document.documentElement.setAttribute("data-demos-total", String(demos.total));
    document.documentElement.setAttribute("data-demos-ok", String(demos.ok));
    document.documentElement.setAttribute("data-demos-failed", demos.failed.join("|") || "none");
    document.documentElement.setAttribute("data-kpv-ready",
      (mj === "ok" && demos.failed.length === 0 && demos.total > 0) ? "ok" : "errors");
  }
  function waitForMathJax(tries) {
    if (window.MathJax && MathJax.startup && MathJax.startup.promise) {
      MathJax.startup.promise.then(function () { mj = "ok"; finish(); },
                                   function () { mj = "error"; finish(); });
    } else if (tries > 0) {
      setTimeout(function () { waitForMathJax(tries - 1); }, 200);
    } else {
      finish();
    }
  }
  waitForMathJax(50);
})();
</script>
</body>
</html>
```

- [ ] **Step 2: 模板冒烟测试（内容校验）**

建临时目录 `tmp/kpv-smoke`：把模板复制为 `derivation.html`，把 `assets/interactive-components.js`、`assets/mathjax/tex-svg.js` 复制进去，写一个只含 `\tag{1}` 的最小 `paper.md`，然后运行：

```bash
python scripts/verify_outputs.py tmp/kpv-smoke/paper.md tmp/kpv-smoke --formulas 1
```
Expected: `VERIFY PASSED`。

- [ ] **Step 3: 模板冒烟测试（浏览器渲染）**

Run: `python scripts/verify_outputs.py tmp/kpv-smoke/paper.md tmp/kpv-smoke --formulas 1 --browser-check`
Expected: `VERIFY PASSED`，且输出含 `browser-render`。若本机找不到 Edge/Chrome，先安装或通过 `--browser <路径>` 指定。

- [ ] **Step 4: 提交**

```bash
git add .codex/skills/knowledge-point-visualization/assets/derivation-template.html
git commit -m "feat(kpv): 新增推导页标准模板 derivation-template.html（MathJax+就绪标记+步骤卡）"
```

---

### Task 4: 更新 SKILL.md 与 agents/openai.yaml

**Files:**
- Modify: `G:\Idea\.codex\skills\knowledge-point-visualization\SKILL.md`
- Modify: `G:\Idea\.codex\skills\knowledge-point-visualization\agents\openai.yaml`

**Interfaces:**
- Consumes: Task 1（校验命令）、Task 2/3（模板与组件库路径）。
- Produces: 新的技能规范，供 Task 5 执行。

- [ ] **Step 1: 输出契约增加两行**

在 `SKILL.md` 的 `## Output contract` 小节 `- cost-editor.html...` 行后追加：

```markdown
- `derivation.html` — 推导页：公式全覆盖 + 逐步推导 + 通俗解释 + 交互演示（用 `assets/derivation-template.html` 骨架生成，内容规范见 Step 5.5）。
- `interactive-components.js` — 交互演示库，从 `assets/` 拷贝。
```

- [ ] **Step 2: 新增 Step 5.5（生成推导页）**

在 `## Step 5 — Visualization` 小节之后、`## Step 6 — Learning path` 之前插入：

```markdown
## Step 5.5 — 生成推导页 → `derivation.html`

用 `assets/derivation-template.html` 作为骨架（AI 只填内容，不重写结构/样式）；把 `assets/interactive-components.js`、`assets/mathjax/tex-svg.js` 拷贝到输出目录。按以下规范生成：

1. **公式全覆盖**：论文中每个 `\tag{N}` 编号公式都必须以相同编号出现在推导页，且放在正确的推导上下文；
2. **逐步推导（面向高中生）**：每个微步骤 = 一个等式 + 通俗解释；先用一句话说明“这一步在做什么”，再用比喻/生活例子展开复杂概念；不默认读者有微积分背景；涉及符号必须链到基础工具箱；
3. **跳步禁令**：禁止“同理可得 / 推导略 / 完整推导见扩展版 / 详见附录”等无引用省略；同构步骤必须显式写出式子并引用步骤号；
4. **含义可视化**：每个关键公式（梯度下降、隐函数求导、SGLD 更新等）配 `data-demo` 交互演示；复杂公式步骤卡默认折叠，公式容器 `overflow-x:auto` 防溢出；
5. **覆盖自检**：页面底部“覆盖自检”表逐项打勾，标注“✓ 全部覆盖，无跳步”。
```

- [ ] **Step 3: 新增 Step 7（校验与交付）**

在 `## Step 6 — Learning path` 小节之后、`## Conventions` 之前插入：

```markdown
## Step 7 — 校验与交付（不过不交付）

运行：

```bash
python <skill>/scripts/verify_outputs.py <paper>.md . --browser-check
```

校验 6 项：产物齐全 / MathJax 生效 / 公式全覆盖 / 无跳步词 / demo 注册完整 / 浏览器真实渲染。失败则修复后重跑，直到输出 `VERIFY PASSED` 才能交付。
```

- [ ] **Step 4: 强化基础概念覆盖章节**

在 `## Basic concept coverage` 小节加一条：

```markdown
- 解释必须高中生可懂：每个概念给“一句话 + 生活例子 + 最小公式 + 论文位置”，禁止默认读者有大学数学背景。
```

- [ ] **Step 5: Conventions 增加两条**

```markdown
- 离线约束：页面只允许相对路径本地引用（`mathjax/tex-svg.js`、`interactive-components.js`），禁止 CDN、fetch、外部字体。
- 交付门槛：`scripts/verify_outputs.py --browser-check` 必须输出 `VERIFY PASSED` 才算完成。
```

- [ ] **Step 6: 更新 agents/openai.yaml**

把 `default_prompt` 改为：

```yaml
interface:
  display_name: "Knowledge Point Visualization"
  short_description: "Paper md → knowledge graph JSON → learning-cost editor → HTML (derivation + graph)"
  default_prompt: "Use $knowledge-point-visualization to turn this paper Markdown into meta.json, all.json, learning-cost.json, learning-path.json, knowledge-graph.html, cost-editor.html, and derivation.html (all paper formulas, high-school-level step-by-step explanations, interactive demos). Then run scripts/verify_outputs.py --browser-check and iterate until it passes."
```

- [ ] **Step 7: 验证文档改动存在**

Run: `rg -n "Step 5.5|Step 7|verify_outputs|interactive-components" .codex/skills/knowledge-point-visualization/SKILL.md .codex/skills/knowledge-point-visualization/agents/openai.yaml`
Expected: 每个关键字至少出现一次。

- [ ] **Step 8: 提交**

```bash
git add .codex/skills/knowledge-point-visualization/SKILL.md .codex/skills/knowledge-point-visualization/agents/openai.yaml
git commit -m "docs(kpv): SKILL.md 增加第 5.5 步推导页与第 7 步校验门，强化高中生通俗解释与离线约束"
```

---

### Task 5: 验收 —— 用 PGD.md 完整重跑新流程

**Files:**
- 参考: `G:\Idea\MinerU-Skill\PGD_07b060\PGD.md`（论文源）、同目录现有 JSON 产物。
- 产出: `G:\Idea\MinerU-Skill\PGD_07b060\derivation.html`（按新规范重写）、`interactive-components.js`、`mathjax/tex-svg.js`。

**Interfaces:**
- Consumes: Task 1 的 `verify_outputs.py`、Task 2/3 的资产、Task 4 的 SKILL.md 内容规范。
- Produces: 全绿的验收样例。

- [ ] **Step 1: 按 Step 5.5 规范重写 derivation.html**

以 `assets/derivation-template.html` 为骨架，覆盖 PGD 论文全部 17 个 `\tag{1}`–`\tag{17}` 公式；每个关键公式（如梯度下降 Eq(10)、隐函数求导 Eq(12)、SGLD Eq(17)）配交互 demo；每个微步骤用高中生能懂的“一句话 + 生活例子”解释；禁止无引用跳步词；页面底部覆盖自检表逐项打勾。

- [ ] **Step 2: 拷贝资产到输出目录**

```bash
Copy-Item assets/interactive-components.js G:\Idea\MinerU-Skill\PGD_07b060\
Copy-Item assets/mathjax/tex-svg.js G:\Idea\MinerU-Skill\PGD_07b060\mathjax\tex-svg.js
```

- [ ] **Step 3: 运行完整校验**

Run: `python scripts/verify_outputs.py G:\Idea\MinerU-Skill\PGD_07b060\PGD.md G:\Idea\MinerU-Skill\PGD_07b060 --browser-check`
Expected: `VERIFY PASSED ... browser-render`。

- [ ] **Step 4: 修复循环直到全绿**

若失败，按 `FAIL` 明细逐项修复 derivation.html，重跑 Step 3，直到全绿。

- [ ] **Step 5: 人工抽查**

无头 Edge 打开 `derivation.html`：确认公式渲染为 SVG（无裸 LaTeX）、梯度下降 demo 可播放/拖动、步骤间有明确“上一步/下一步”关系、复杂公式无横向截断。

- [ ] **Step 6: 提交验收结果**

```bash
git add .codex/skills/knowledge-point-visualization
git commit -m "feat(kpv): 推导页模板+交互组件+校验门落地，PGD 验收样例全绿"
```

---

## Self-Review

**1. Spec 覆盖：**
- 交付契约（9+1 文件）→ Task 1 Check 1 + Global Constraints ✓
- MathJax 生效（裸 LaTeX 禁止）→ Task 1 Check 2/6 + Task 3 模板 ✓
- 公式全覆盖 → Task 1 Check 3 + Task 5 ✓
- 无跳步表述 → Task 1 Check 4 + SKILL.md 第 5.5 步 ✓
- 复杂公式可视化可读（溢出/臃肿/关系清晰）→ 模板 `.eq{overflow-x:auto}`、步骤卡 + 上一步/下一步导航、Task 5 Step 5 ✓
- 可交互含义演示（梯度下降等）→ Task 2 五个 demo + 模板嵌入 ✓
- 面向高中生通俗解释 → SKILL.md 基础概念强化 + Step 5.5 规范 + Task 5 ✓
- 每次运行一次达标 → 第 7 步校验门“不过不交付” ✓

**2. 占位符扫描：** 无 TBD/TODO；所有代码步骤均含完整内容。

**3. 类型一致性：** `verify_outputs.py` 参数与 Task 5 命令一致；`data-demo` id 与 `register()` id 一致（`gradient-descent` 等 5 个）；就绪标记 `data-mathjax`/`data-kpv-ready` 在 Task 2（initAll 写 `__KPV_DEMOS__`）与 Task 3（模板 boot 写 html 属性）及 Task 1 Check 6（读取 html 属性）三处一致。
