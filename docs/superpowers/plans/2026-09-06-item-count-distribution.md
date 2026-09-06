# 交互数分布图实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 `plot_item_freq.py` 增加可选 `--count-dist` 输出：x=交互数、y=拥有该交互数的物品数（PNG+CSV，多数据集组合图）。

**Architecture:** 新增纯函数 `build_count_distribution(counts)`（TDD 先行）与绘图函数；`main()` 在 `--count-dist` 开启时追加产物；复用现有样式与坐标工具。

**Tech Stack:** Python 3.12、matplotlib、numpy、unittest。

## Global Constraints

- 不改变现有 line/hist 默认输出行为。
- 只统计 c>0 的物品；缺失交互数补 0；空 counts 抛 ValueError。
- 组合图仅在面板数 >1 时输出。
- 全量回归命令：`python -m unittest discover -s tests -p "test_*.py" -v`（TPA 根，PowerShell）。

---

### Task 1：build_count_distribution（TDD）

**Files:**
- Modify: `TPA/visualization/item_freq/plot_item_freq.py`
- Test: `TPA/tests/test_item_freq.py`

**Interfaces:**
- Produces: `build_count_distribution(counts: Counter) -> tuple[np.ndarray, np.ndarray]`

- [ ] 先加失败测试：

```python
def test_build_count_distribution_basic(self):
    from visualization.item_freq.plot_item_freq import build_count_distribution
    from collections import Counter
    x, y = build_count_distribution(Counter({1: 3, 3: 2, 7: 1}))
    self.assertEqual(x.tolist(), [1, 2, 3, 4, 5, 6, 7])
    self.assertEqual(y.tolist(), [3, 0, 2, 0, 0, 0, 1])

def test_build_count_distribution_empty_raises(self):
    from visualization.item_freq.plot_item_freq import build_count_distribution
    from collections import Counter
    with self.assertRaises(ValueError):
        build_count_distribution(Counter())
```

- [ ] 运行确认失败（函数不存在）。
- [ ] 实现函数；运行测试通过。
- [ ] 提交 `feat(viz): 交互数分布统计函数`

---

### Task 2：绘图与 CLI 接入

**Files:**
- Modify: `TPA/visualization/item_freq/plot_item_freq.py`

- [ ] 新增 `plot_count_distribution` / `plot_count_distribution_all`（复用
  `_apply_publication_style/_apply_scale/_plot_y`）。
- [ ] `main()` 增加 `--count-dist`；为每个数据集写
  `count_dist_{dataset}.csv/.png`；面板 >1 时写 `count_dist_all_datasets.png`。
- [ ] `python tests`（test_item_freq）通过。
- [ ] 提交 `feat(viz): 交互数分布图输出`

---

### Task 3：README 与全量回归

**Files:**
- Modify: `TPA/visualization/item_freq/README.md`

- [ ] README 增加 `--count-dist` 说明与产物清单。
- [ ] 全量 unittest 通过；提交 `docs(viz): count-dist 用法说明`。

---

## Self-Review

- spec §2–§4 均有对应 Task；接口命名一致；无占位步骤。
