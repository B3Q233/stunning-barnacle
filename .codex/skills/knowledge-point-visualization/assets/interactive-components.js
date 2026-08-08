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
