# 交互数十等分直方图实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `plot_item_freq.py` 增加 `--decile-dist`：按交互数取值十分位切成 10 档，输出各档去重物品数直方图（PNG+CSV+组合图）。

**Architecture:** 纯函数 `build_decile_item_counts`（TDD）+ 绘图函数 + CLI 接入；复用现有样式工具。

**Tech Stack:** Python 3.12、matplotlib、numpy、unittest。

## Global Constraints

- 只统计 c>0 的物品；空 counts 抛 ValueError。
- 边界 = np.percentile(交互数, [10,…,100])；每件物品归入“首个 ≥ c 的边界”档。
- 输出不改变现有 line/hist/count-dist 行为。
- 全量回归：`python -m unittest discover -s tests -p "test_*.py" -v`（TPA 根）。

---

### Task 1：build_decile_item_counts（TDD）

**Files:** Modify `TPA/visualization/item_freq/plot_item_freq.py`、`TPA/tests/test_item_freq.py`

**Produces:** `build_decile_item_counts(counts) -> (labels, boundaries, item_counts)`

- [ ] 失败测试：counts=Counter({i: i for i in range(1, 101)}) → 10 档各 10 件；
  全等 counts → 全部在第 1 档、其余 0；空输入 ValueError。
- [ ] 实现函数；测试通过；提交 `feat(viz): 交互数十等分统计`

---

### Task 2：绘图与 CLI

**Files:** Modify `TPA/visualization/item_freq/plot_item_freq.py`

- [ ] 新增 `plot_decile_histogram` / `plot_decile_histogram_all`（顶会风格柱状图）。
- [ ] `main()` 增加 `--decile-dist`；输出
  `decile_dist_{dataset}.csv`（bucket,boundary,item_count）、png，多数据集组合图。
- [ ] 冒烟测试；提交 `feat(viz): 十分位直方图输出`

---

### Task 3：README 与全量回归

**Files:** Modify `TPA/visualization/item_freq/README.md`

- [ ] README 增加 `--decile-dist` 说明与产物清单；全量 unittest；提交。

---

## Self-Review

- spec §2–§4 均有 Task；接口命名一致；无占位步骤。
