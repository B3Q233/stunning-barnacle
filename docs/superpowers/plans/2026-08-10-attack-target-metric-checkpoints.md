# 攻击按目标物品指标选优 Checkpoint + 共享评估层 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让四个攻击（tpa / pgd / bandwagon / random）与技能模板的中毒模型
checkpoint 按**被攻击目标物品**的 NDCG / HR 选优（`target_ndcg@K` 为主指标），
并把攻击评估逻辑收敛到共享模块，以后加指标只改一处。

**Architecture:** 新建共享模块 `TPA/evaluation/attack_eval.py`
（ranking_scores / compute_target_metrics / aggregate_target_metrics /
build_attack_eval_metrics / compare_models / format_report / save_report），
四个攻击与模板的 `evaluate.py` 变纯 re-export 薄壳；各攻击 `fit.py` 训练评估
调用 `build_attack_eval_metrics` 把 `target_*` 指标并入 BestTracker，配置里
target 指标排最前使 `primary_metric == "target_ndcg@10"`，`--skip-train` 与
最终对比报告都加载按目标 NDCG 选出的最优模型。

**Tech Stack:** Python 3.12 + PyTorch 2.5 + numpy；测试用 stdlib unittest
（不新增依赖）；命令在 PowerShell 执行。

## Global Constraints

- 攻击评估 K 以各 config 现有 `@K` 为准：tpa / pgd / bandwagon / random = `@10`；
  技能模板 = `@20`。
- 目标人群过滤统一用**干净训练集** `clean_meta["user_items"]`，保证 clean /
  poisoned 两次评估口径一致。
- `target_hr@K` / `target_ndcg@K` = 多目标**等权均值**；跳过 `n_elig == 0` 的
  目标；所有目标均无合格用户时返回 `0.0`。
- 报告输出文件名保持现状：tpa / random → `attack_comparison.md`；pgd →
  `pgd_comparison.md`；bandwagon → `bandwagon_comparison.md`；模板 →
  `attack_comparison.md`。
- 不改 `training/metrics.py`（BestTracker 已支持任意指标名）；不改
  `models/*` 干净模型训练逻辑。
- 不新增第三方依赖；测试用 stdlib unittest。
- 向后兼容：metrics 配置无 `target_*` 时只计算整体指标，行为与旧版一致；
  `--skip-train` 加载链不变（`{primary}-best-model.pt` → `best.pt` →
  `latest.pt`）。
- 提交按任务分，沿用 `feat` / `docs` 前缀；每个任务结束必须跑测试与
  `py_compile`。

---

### Task 1: 共享攻击评估层 `evaluation/attack_eval.py` + 单测 + evaluate.py 薄壳化

**Files:**
- Create: `TPA/evaluation/attack_eval.py`
- Create: `TPA/tests/test_attack_eval.py`
- Modify: `TPA/attacks/tpa/evaluate.py`、`TPA/attacks/pgd/evaluate.py`、
  `TPA/attacks/bandwagon/evaluate.py`、`TPA/attacks/random/evaluate.py`
  （全部替换为薄壳，内容见 Step 5）

**Interfaces:**
- Produces（Task 2/3 依赖的精确签名）：
  - `aggregate_target_metrics(target_metrics: Dict[int, Dict[str, Any]],
    target_items: List[int], k: int) -> Dict[str, float]`
  - `build_attack_eval_metrics(scores, user_ids, user_items, test_pos,
    clean_user_items, targets, ks, metric_names)
    -> Tuple[Dict[str, float], Dict[int, Dict[str, Any]]]`
  - `save_report(report, out_dir, name="attack", title=None) -> Path`
  - `compute_target_metrics(scores, user_ids, clean_user_items, target_items, k)
    -> Dict[int, Dict[str, Any]]`（返回明细含新增 `n_elig` 字段）

- [ ] **Step 1: 写失败测试 `TPA/tests/test_attack_eval.py`**

```python
"""evaluation.attack_eval 单元测试（unittest，无第三方依赖）"""
import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from evaluation.attack_eval import (
    aggregate_target_metrics,
    build_attack_eval_metrics,
    compute_target_metrics,
    format_report,
    save_report,
)


def make_scores(rows):
    return torch.tensor(rows, dtype=torch.float32)


class ComputeTargetMetricsTest(unittest.TestCase):
    """3 用户 × 5 物品；目标物品 4。"""

    SCORES = make_scores([
        [5.0, 4.0, 3.0, 2.0, 9.0],    # u0：item4 rank1（但训练集已交互，被排除）
        [9.0, 8.0, 6.0, 1.0, 7.0],    # u1：item4 rank3（k=3 命中）
        [1.0, 2.0, 3.0, 9.0, 0.5],    # u2：item4 rank5（未命中）
    ])
    USER_IDS = [0, 1, 2]

    def test_hit_and_eligible_filter(self):
        clean = {0: {4}, 1: set(), 2: set()}  # u0 训练集已交互目标 → 排除
        out = compute_target_metrics(self.SCORES, self.USER_IDS, clean, [4], 3)
        m = out[4]
        self.assertEqual(m["n_elig"], 2)
        self.assertAlmostEqual(m["hr@k"], 0.5)
        self.assertAlmostEqual(m["ndcg@k"], 0.25)      # dcg = 1/log2(4) = 0.5，/2
        self.assertEqual(m["hit_users"], 1)
        self.assertAlmostEqual(m["mean_rank"], 3.0)
        self.assertAlmostEqual(m["mean_rank_all"], 4.0)  # (3 + 5) / 2
        # 旧别名兼容
        self.assertEqual(m["exposure"], m["hr@k"])
        self.assertEqual(m["ndcg"], m["ndcg@k"])

    def test_no_eligible_users(self):
        clean = {0: {4}, 1: {4}, 2: {4}}
        m = compute_target_metrics(self.SCORES, self.USER_IDS, clean, [4], 3)[4]
        self.assertEqual(m["n_elig"], 0)
        self.assertEqual(m["hr@k"], 0.0)
        self.assertEqual(m["ndcg@k"], 0.0)
        self.assertIsNone(m["mean_rank"])
        self.assertIsNone(m["mean_rank_all"])

    def test_all_eligible_hit_at_rank_one(self):
        scores = make_scores([[9.0, 1.0, 0.5], [8.0, 2.0, 0.5]])
        out = compute_target_metrics(scores, [0, 1], {0: set(), 1: set()}, [0], 3)
        self.assertEqual(out[0]["hr@k"], 1.0)
        self.assertAlmostEqual(out[0]["ndcg@k"], 1.0)  # rank1：dcg=1，IDCG=1


class AggregateTargetMetricsTest(unittest.TestCase):
    def test_single_target(self):
        tm = {5: {"hr@k": 0.5, "ndcg@k": 0.25, "n_elig": 2}}
        agg = aggregate_target_metrics(tm, [5], 10)
        self.assertEqual(agg, {"target_hr@10": 0.5, "target_ndcg@10": 0.25})

    def test_multiple_targets_equal_mean(self):
        tm = {
            5: {"hr@k": 0.5, "ndcg@k": 0.25, "n_elig": 2},
            7: {"hr@k": 0.8, "ndcg@k": 0.6, "n_elig": 5},
        }
        agg = aggregate_target_metrics(tm, [5, 7], 10)
        self.assertAlmostEqual(agg["target_hr@10"], 0.65)
        self.assertAlmostEqual(agg["target_ndcg@10"], 0.425)

    def test_skips_target_without_eligible_users(self):
        tm = {
            5: {"hr@k": 0.5, "ndcg@k": 0.25, "n_elig": 2},
            7: {"hr@k": 0.0, "ndcg@k": 0.0, "n_elig": 0},
        }
        agg = aggregate_target_metrics(tm, [5, 7], 10)
        self.assertEqual(agg, {"target_hr@10": 0.5, "target_ndcg@10": 0.25})

    def test_all_no_eligible_returns_zero(self):
        tm = {
            5: {"hr@k": 0.0, "ndcg@k": 0.0, "n_elig": 0},
            7: {"hr@k": 0.0, "ndcg@k": 0.0, "n_elig": 0},
        }
        agg = aggregate_target_metrics(tm, [5, 7], 10)
        self.assertEqual(agg, {"target_hr@10": 0.0, "target_ndcg@10": 0.0})


class BuildAttackEvalMetricsTest(unittest.TestCase):
    """与 ComputeTargetMetricsTest 同夹具；整体指标按测试计划手算。"""

    SCORES = make_scores([
        [5.0, 4.0, 3.0, 2.0, 9.0],
        [9.0, 8.0, 6.0, 1.0, 7.0],
        [1.0, 2.0, 3.0, 9.0, 0.5],
    ])
    USER_IDS = [0, 1, 2]
    USER_ITEMS = {0: {0}, 1: set(), 2: set()}   # 整体指标过滤用（训练集）
    TEST_POS = {0: {1}, 1: {4}, 2: {0}}
    CLEAN = {0: {4}, 1: set(), 2: set()}        # 目标人群过滤用（干净训练集）
    TARGETS = [4]

    def test_with_target_metrics(self):
        names = ["target_ndcg@3", "target_hr@3", "recall@3", "ndcg@3"]
        res, details = build_attack_eval_metrics(
            self.SCORES, self.USER_IDS, self.USER_ITEMS, self.TEST_POS,
            self.CLEAN, self.TARGETS, [3], names,
        )
        self.assertEqual(set(res), set(names))
        self.assertAlmostEqual(res["target_ndcg@3"], 0.25)
        self.assertAlmostEqual(res["target_hr@3"], 0.5)
        self.assertAlmostEqual(res["recall@3"], 2 / 3)
        # u0 ndcg = 1/log2(3)，u1 ndcg = 0.5，u2 = 0 → 均值
        self.assertAlmostEqual(
            res["ndcg@3"], (1.0 / math.log2(3) + 0.5) / 3)
        self.assertEqual(set(details), {4})
        self.assertEqual(details[4]["n_elig"], 2)

    def test_without_target_metrics(self):
        names = ["recall@3", "ndcg@3"]
        res, details = build_attack_eval_metrics(
            self.SCORES, self.USER_IDS, self.USER_ITEMS, self.TEST_POS,
            self.CLEAN, self.TARGETS, [3], names,
        )
        self.assertEqual(set(res), set(names))
        self.assertEqual(details, {})

    def test_empty_targets_no_crash(self):
        names = ["target_ndcg@3", "recall@3"]
        res, details = build_attack_eval_metrics(
            self.SCORES, self.USER_IDS, self.USER_ITEMS, self.TEST_POS,
            self.CLEAN, [], [3], names,
        )
        self.assertEqual(res["target_ndcg@3"], 0.0)
        self.assertEqual(details, {})


class FormatReportTest(unittest.TestCase):
    REPORT = {
        "k": 3,
        "model_utility": {
            "clean": {"recall@3": 0.6, "ndcg@3": 0.4},
            "poisoned": {"recall@3": 0.58, "ndcg@3": 0.39},
        },
        "target_metrics": {
            "clean": {4: {"hr@k": 0.0, "ndcg@k": 0.0, "mean_rank_all": 900.0}},
            "poisoned": {4: {"hr@k": 0.5, "ndcg@k": 0.25, "mean_rank_all": 300.0}},
        },
    }

    def test_title_and_conclusion(self):
        md = format_report(self.REPORT, title="PGD（投影梯度上升投毒）攻击对比报告")
        self.assertIn("# PGD（投影梯度上升投毒）攻击对比报告（Top-3）", md)
        self.assertIn("## 结论", md)
        self.assertIn("投毒显著提升了目标物品曝光", md)
        self.assertIn("recall@3", md)

    def test_missing_recall_does_not_crash(self):
        report = dict(self.REPORT)
        report["model_utility"] = {
            "clean": {"ndcg@3": 0.4},
            "poisoned": {"ndcg@3": 0.39},
        }
        md = format_report(report)
        self.assertIn("## 结论", md)

    def test_no_utility_skips_utility_section(self):
        report = dict(self.REPORT)
        report["model_utility"] = {"clean": None, "poisoned": None}
        md = format_report(report)
        self.assertNotIn("模型效用（测试集", md)
        self.assertIn("## 结论", md)


class SaveReportTest(unittest.TestCase):
    def test_name_mapping_and_json(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            md = save_report(FormatReportTest.REPORT, out, name="pgd")
            self.assertEqual(md.name, "pgd_comparison.md")
            self.assertTrue((out / "pgd_comparison.json").exists())
            self.assertIn(
                "PGD（投影梯度上升投毒）攻击对比报告（Top-3）",
                md.read_text(encoding="utf-8"),
            )
            json.loads((out / "pgd_comparison.json").read_text(encoding="utf-8"))

    def test_tpa_and_template_default_to_attack_name(self):
        for name in ("tpa", "attack_imp_direct_poison"):
            with tempfile.TemporaryDirectory() as d:
                out = Path(d)
                save_report(FormatReportTest.REPORT, out, name=name)
                self.assertTrue((out / "attack_comparison.md").exists())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试确认失败**

```powershell
G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_attack_eval -v
```

Expected: FAIL / ModuleNotFoundError（`evaluation.attack_eval` 不存在）。

- [ ] **Step 3: 实现 `TPA/evaluation/attack_eval.py`**

```python
"""攻击评估共享层 —— 所有攻击模块（tpa/pgd/bandwagon/random）统一使用。

收敛自 attacks/{attack}/evaluate.py 的重复实现：
- ranking_scores / compute_target_metrics / compare_models：纯共用
- aggregate_target_metrics：多目标 hr@K / ndcg@K 等权均值（checkpoint 选优用）
- build_attack_eval_metrics：训练中单次评估 = 整体指标 + 目标指标合并
- format_report / save_report：报告标题与输出文件名由 name 参数控制

两类指标：
1. 模型效用（clean vs poisoned 在测试集上的 recall@K / ndcg@K）——投毒代价检查
2. 攻击效果（目标物品在 Top-K 中的 HR@K / NDCG@K / 命中用户数 / 平均排名）

评估协议与 LightGCN 一致：all-ranking、过滤训练集已交互物品。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from evaluation.metrics import compute_metrics, match_metric_values


# 报告输出文件名与标题：按攻击模块名归一化，保持各攻击现有文件名不变
REPORT_NAMES = {
    "tpa": "attack",
    "random": "attack",
    "pgd": "pgd",
    "bandwagon": "bandwagon",
    "attack_imp_direct_poison": "attack",
}
REPORT_TITLES = {
    "pgd": "PGD（投影梯度上升投毒）攻击对比报告",
    "bandwagon": "Bandwagon（从众）攻击对比报告",
}


def ranking_scores(model, test_pairs: List[Tuple[int, int]]
                   ) -> Tuple[torch.Tensor, List[int], Dict[int, set]]:
    """对测试用户做全量排序评分。

    返回 (scores, test_user_ids, test_pos)。
    scores: (n_test_users, n_items)，行顺序与 test_user_ids 一致。
    """
    test_pos: Dict[int, set] = {}
    for u, i in test_pairs:
        test_pos.setdefault(u, set()).add(i)
    test_users = sorted(test_pos.keys())

    model.set_eval()
    with torch.no_grad():
        user_emb = model.get_user_embeddings()
        item_emb = model.get_item_embeddings()
        ids = torch.LongTensor(test_users).to(user_emb.device)
        scores = user_emb[ids] @ item_emb.T
    return scores, test_users, test_pos


def compute_target_metrics(scores: torch.Tensor, user_ids: List[int],
                           clean_user_items: Dict[int, set],
                           target_items: List[int], k: int) -> Dict[int, Dict[str, Any]]:
    """目标物品的攻击效果指标（HR@K / NDCG@K）。

    只统计"训练集中未交互过该目标物品"的合法用户（攻击的目标人群），
    用干净训练集过滤，保证 clean / poisoned 两次评估口径一致。

    返回 {target: {hr@k, ndcg@k, hit_users, mean_rank, mean_rank_all, n_elig}}
    - hr@k: 目标物品进入 Top-K 的用户比例（= 经典 Hit Rate @K）
    - ndcg@k: 单目标 NDCG（命中用户按排名位置折损，IDCG=1）
    - mean_rank: 命中用户的平均排名（未命中记 None；越小攻击越强）
    - mean_rank_all: 全体合格用户的平均排名（不受 Top-K 截断，更灵敏）
    - n_elig: 合格用户数（聚合时跳过 n_elig == 0 的目标）
    """
    topk = torch.topk(scores, k, dim=1).indices  # (n_users, k)
    ranks_all = torch.argsort(scores, dim=1, descending=True)  # (n_users, n_items)
    out: Dict[int, Dict[str, Any]] = {}
    for t in target_items:
        eligible = [
            r for r, uid in enumerate(user_ids)
            if t not in clean_user_items.get(uid, set())
        ]
        n_elig = len(eligible)
        if n_elig == 0:
            out[t] = {"hr@k": 0.0, "ndcg@k": 0.0, "hit_users": 0,
                      "mean_rank": None, "mean_rank_all": None, "n_elig": 0,
                      "exposure": 0.0, "ndcg": 0.0}
            continue

        hits = 0
        dcg = 0.0
        ranks: List[int] = []
        ranks_all_list: List[int] = []
        for r in eligible:
            pos = (topk[r] == t).nonzero(as_tuple=False)
            if pos.numel():
                rank = int(pos.item()) + 1
                hits += 1
                dcg += 1.0 / np.log2(rank + 1)
                ranks.append(rank)
            pos_all = (ranks_all[r] == t).nonzero(as_tuple=False)
            ranks_all_list.append(int(pos_all.item()) + 1)

        out[t] = {
            "hr@k": hits / n_elig,
            "ndcg@k": dcg / n_elig,
            "hit_users": hits,
            "mean_rank": float(np.mean(ranks)) if ranks else None,
            "mean_rank_all": float(np.mean(ranks_all_list)),
            "n_elig": n_elig,
            # 旧字段别名，兼容已有 JSON 消费者
            "exposure": hits / n_elig,
            "ndcg": dcg / n_elig,
        }
    return out


def aggregate_target_metrics(target_metrics: Dict[int, Dict[str, Any]],
                             target_items: List[int], k: int) -> Dict[str, float]:
    """把每个目标物品的 hr@k / ndcg@k 聚合为 checkpoint 选优指标。

    等权均值：跳过 n_elig == 0 的目标（避免把 0 拉低均值）；
    所有目标均无合格用户时返回 0.0。

    返回 {"target_hr@k": float, "target_ndcg@k": float}
    """
    eligible = [target_metrics[t] for t in target_items
                if t in target_metrics and target_metrics[t].get("n_elig", 0) > 0]
    if not eligible:
        return {"target_hr@k": 0.0, "target_ndcg@k": 0.0}
    return {
        "target_hr@k": float(np.mean([m["hr@k"] for m in eligible])),
        "target_ndcg@k": float(np.mean([m["ndcg@k"] for m in eligible])),
    }


def build_attack_eval_metrics(scores: torch.Tensor, user_ids: List[int],
                              user_items: Dict[int, set],
                              test_pos: Dict[int, set],
                              clean_user_items: Dict[int, set],
                              targets: List[int], ks: List[int],
                              metric_names: List[str]
                              ) -> Tuple[Dict[str, float], Dict[int, Dict[str, Any]]]:
    """训练中单次评估：整体指标 + （可选）目标指标，与配置指标名对齐。

    - 先算目标指标（避免被 compute_metrics 的 -inf 过滤污染排名）；
    - 整体指标：compute_metrics(scores, user_items, test_pos, k=K)；
    - 若 metric_names 含 target_ 前缀指标，对每个 K 追加 target_hr@K /
      target_ndcg@K；
    - 返回 (res, target_details)：
      res 为扁平 {指标名: 值}（BestTracker.update 直接消费）；
      target_details 为最大 K 下的 {target: {...}} 明细（写入 history）。
    """
    target_by_k: Dict[int, Dict[int, Dict[str, Any]]] = {}
    if any(name.startswith("target_") for name in metric_names):
        for K in ks:
            target_by_k[K] = compute_target_metrics(
                scores, user_ids, clean_user_items, targets, K)

    res_by_k: Dict[int, Dict[str, float]] = {
        K: compute_metrics(scores, user_items, test_pos, k=K) for K in ks
    }
    for K in ks:
        if K in target_by_k:
            res_by_k[K].update(aggregate_target_metrics(target_by_k[K], targets, K))

    target_details = target_by_k.get(max(ks), {}) if ks else {}
    return match_metric_values(metric_names, res_by_k), target_details


def compare_models(clean_model, poisoned_model, clean_meta: Dict[str, Any],
                   poisoned_meta: Dict[str, Any], target_items: List[int], k: int,
                   report_utility: bool = True) -> Dict[str, Any]:
    """clean vs poisoned 全量对比。"""
    scores_c, users_c, test_pos_c = ranking_scores(clean_model, clean_meta["test_pairs"])
    scores_p, users_p, test_pos_p = ranking_scores(poisoned_model, poisoned_meta["test_pairs"])

    # 模型效用（各自训练集过滤；真实用户的训练交互在注入前后一致）
    # 用于衡量攻击代价：投毒后推荐质量不应显著下降
    if report_utility:
        clean_util = compute_metrics(scores_c, clean_meta["user_items"], test_pos_c, k)
        poisoned_util = compute_metrics(scores_p, poisoned_meta["user_items"], test_pos_p, k)
    else:
        clean_util = poisoned_util = None

    # 攻击效果（统一用干净训练集过滤目标人群）
    clean_att = compute_target_metrics(scores_c, users_c, clean_meta["user_items"],
                                       target_items, k)
    poisoned_att = compute_target_metrics(scores_p, users_p, clean_meta["user_items"],
                                          target_items, k)

    return {
        "k": k,
        "model_utility": {
            "clean": clean_util,
            "poisoned": poisoned_util,
        },
        "target_metrics": {
            "clean": clean_att,
            "poisoned": poisoned_att,
        },
    }


def format_report(report: Dict[str, Any], title: str = "投毒攻击对比报告") -> str:
    """把对比结果格式化为 Markdown 报告；标题由参数控制。"""
    k = report["k"]
    lines = [
        f"# {title}（Top-{k}）",
        "",
    ]
    cu = report["model_utility"]["clean"]
    pu = report["model_utility"]["poisoned"]
    if cu is not None:
        lines += [
            "## 模型效用（测试集 all-ranking，投毒代价检查）",
            "",
            "| 指标 | Clean | Poisoned | Δ |",
            "|------|-------|----------|---|",
        ]
        for key in cu:
            delta = pu[key] - cu[key]
            lines.append(f"| {key} | {cu[key]:.4f} | {pu[key]:.4f} | {delta:+.4f} |")

    lines += [
        "",
        f"## 目标物品攻击效果（HR@{k} / NDCG@{k}）",
        "",
        "合格用户 = 训练集未交互过该目标物品的用户（统一用干净训练集过滤）",
        "",
        f"| Target | Clean HR@{k} | Poisoned HR@{k} | Clean NDCG@{k} | Poisoned NDCG@{k} | "
        f"Clean 平均排名 | Poisoned 平均排名 |",
        f"|--------|------------|----------------|---------------|------------------|"
        f"---------------|------------------|",
    ]
    ca = report["target_metrics"]["clean"]
    pa = report["target_metrics"]["poisoned"]
    for t in ca:
        cr = ca[t]
        pr = pa[t]
        rank_c = f"{cr['mean_rank_all']:.1f}"
        rank_p = f"{pr['mean_rank_all']:.1f}"
        lines.append(
            f"| {t} | {cr['hr@k']:.4f} | {pr['hr@k']:.4f} | "
            f"{cr['ndcg@k']:.4f} | {pr['ndcg@k']:.4f} | {rank_c} | {rank_p} |"
        )

    # 结论段：攻击增益 + 投毒代价（通用文案，不再写攻击名专属措辞）
    lines += ["", "## 结论", ""]
    for t in ca:
        cr = ca[t]
        pr = pa[t]
        hr_delta = pr["hr@k"] - cr["hr@k"]
        lines.append(
            f"- 目标物品 {t}：HR@{k} {cr['hr@k']:.4f} → {pr['hr@k']:.4f} "
            f"（{hr_delta:+.4f}），平均排名 {cr['mean_rank_all']:.1f} → "
            f"{pr['mean_rank_all']:.1f}；投毒显著提升了目标物品曝光。"
        )
    if cu is not None:
        rec_keys = [key for key in cu if key.startswith("recall@")]
        if rec_keys:
            rec_key = rec_keys[0]
            util_delta = pu[rec_key] - cu[rec_key]
            trend = ("未下降（投毒代价可接受）" if util_delta >= -0.01
                     else "显著下降（投毒代价过大，需调低假用户比例）")
            lines.append(
                f"- 模型效用 {rec_key}：Clean {cu[rec_key]:.4f} → Poisoned "
                f"{pu[rec_key]:.4f}（{util_delta:+.4f}），{trend}。"
            )
    lines.append("")
    return "\n".join(lines)


def save_report(report: Dict[str, Any], out_dir: Path, name: str = "attack",
                title: str | None = None) -> Path:
    """写 {name}_comparison.md 与 {name}_comparison.json，返回 md 路径。

    name 按 REPORT_NAMES 归一化（保持各攻击现有文件名）；title 缺省时按
    REPORT_TITLES 或通用标题。
    """
    file_name = REPORT_NAMES.get(name, "attack")
    if title is None:
        title = REPORT_TITLES.get(name, "投毒攻击对比报告")
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{file_name}_comparison.md"
    md_path.write_text(format_report(report, title=title), encoding="utf-8")
    (out_dir / f"{file_name}_comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return md_path
```

- [ ] **Step 4: 运行测试确认通过**

```powershell
G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_attack_eval -v
```

Expected: 全部 PASS。

- [ ] **Step 5: 四个攻击 evaluate.py 薄壳化（替换整个文件内容）**

以下内容用于 `TPA/attacks/tpa/evaluate.py`，文件第一行 docstring 换成各自的攻击名：

```python
"""TPA 攻击效果评估 —— 共享实现见 evaluation/attack_eval.py"""
from evaluation.attack_eval import (  # noqa: F401
    build_attack_eval_metrics,
    compare_models,
    compute_target_metrics,
    format_report,
    ranking_scores,
    save_report,
)
```

docstring 分别替换为：`PGD 攻击效果评估`、`Bandwagon 攻击效果评估`、
`Random 攻击效果评估`（对应 pgd / bandwagon / random）。

- [ ] **Step 6: 语法检查 + 全量回归**

```powershell
G:\Idea\.venv\Scripts\python.exe -m py_compile G:\Idea\TPA\evaluation\attack_eval.py G:\Idea\TPA\tests\test_attack_eval.py G:\Idea\TPA\attacks\tpa\evaluate.py G:\Idea\TPA\attacks\pgd\evaluate.py G:\Idea\TPA\attacks\bandwagon\evaluate.py G:\Idea\TPA\attacks\random\evaluate.py
G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_training_metrics tests.test_modes tests.test_attack_eval -v
```

Expected: 无语法错误；全部 PASS。

- [ ] **Step 7: Commit**

```bash
git -C G:\Idea add TPA/evaluation/attack_eval.py TPA/tests/test_attack_eval.py TPA/attacks/tpa/evaluate.py TPA/attacks/pgd/evaluate.py TPA/attacks/bandwagon/evaluate.py TPA/attacks/random/evaluate.py
git -C G:\Idea commit -m "feat(eval): 共享攻击评估层 attack_eval.py 与薄壳化 + 单测"
```

---

### Task 2: 四个攻击 fit.py 接入目标物品选优 + config 重排

**Files:**
- Modify: `TPA/attacks/tpa/fit.py`、`TPA/attacks/pgd/fit.py`、
  `TPA/attacks/bandwagon/fit.py`、`TPA/attacks/random/fit.py`
- Modify: `TPA/attacks/tpa/config.yaml`、`TPA/attacks/pgd/config.yaml`、
  `TPA/attacks/bandwagon/config.yaml`、`TPA/attacks/random/config.yaml`

**Interfaces:**
- Consumes: Task 1 的 `build_attack_eval_metrics` 与 `save_report(report, out_dir,
  name=...)`（通过各攻击自身的 evaluate 薄壳导入）。
- Produces: 训练产出 `target_ndcg@10-best-model.pt` / `target_hr@10-best-model.pt`
  等 checkpoint；`history.json` 的 `best` 段含 `target_ndcg@10`，评估 epoch 条目
  含 `targets` 明细。

- [ ] **Step 1: 修改 `TPA/attacks/tpa/fit.py` 的 import 块**

把：

```python
from evaluation.metrics import compute_metrics
```

删除；把：

```python
from attacks.tpa.evaluate import compare_models, ranking_scores, save_report
```

改为：

```python
from attacks.tpa.evaluate import (
    build_attack_eval_metrics,
    compare_models,
    ranking_scores,
    save_report,
)
```

并把 `training.metrics` 导入中的 `match_metric_values` 删除，只保留
`BestTracker, eval_ks_from_metrics, safe_checkpoint_name`。

- [ ] **Step 2: 修改 `train_poisoned_model` 签名**

把：

```python
def train_poisoned_model(cfg: TrainingConfig, poisoned_meta: Dict[str, Any],
                         out_dir: Path, warm_start: bool,
                         warm_ckpt: Path | None, clean_num_users: int | None,
                         model_cls, dataset_cls, metrics_cfg,
                         checkpoint_mode: str = "per_metric"
                         ) -> Tuple[Any, List[Dict[str, Any]]]:
```

改为：

```python
def train_poisoned_model(cfg: TrainingConfig, poisoned_meta: Dict[str, Any],
                         out_dir: Path, warm_start: bool,
                         warm_ckpt: Path | None, clean_num_users: int | None,
                         model_cls, dataset_cls, metrics_cfg,
                         checkpoint_mode: str = "per_metric",
                         targets: List[int] | None = None,
                         clean_user_items: Dict[int, set] | None = None,
                         ) -> Tuple[Any, List[Dict[str, Any]]]:
```

- [ ] **Step 3: 替换评估 epoch 的指标计算块**

把：

```python
        if epoch % eval_every == 0 or epoch == 1:
            scores, users, test_pos_local = ranking_scores(model, poisoned_meta["test_pairs"])
            ks = eval_ks_from_metrics(metrics_cfg, k)
            res_by_k = {
                K: compute_metrics(scores, user_items, test_pos_local, k=K)
                for K in ks
            }
            res = match_metric_values(list(tracker.directions), res_by_k)
            entry.update(res)
```

改为：

```python
        if epoch % eval_every == 0 or epoch == 1:
            scores, users, test_pos_local = ranking_scores(model, poisoned_meta["test_pairs"])
            ks = eval_ks_from_metrics(metrics_cfg, k)
            res, target_details = build_attack_eval_metrics(
                scores, users, user_items, test_pos_local,
                clean_user_items or {}, targets or [], ks,
                list(tracker.directions),
            )
            entry.update(res)
            if target_details:
                entry["targets"] = target_details
```

- [ ] **Step 4: 修改 `main()` 的调用与报告输出**

把：

```python
        model, history = train_poisoned_model(
            cfg, poisoned_meta, out_dir, warm_start, warm_ckpt,
            clean_num_users=clean_meta["num_users"] if warm_start else None,
            model_cls=model_cls,
            dataset_cls=dataset_cls,
            metrics_cfg=metrics_cfg,
            checkpoint_mode=checkpoint_mode,
        )
```

改为：

```python
        model, history = train_poisoned_model(
            cfg, poisoned_meta, out_dir, warm_start, warm_ckpt,
            clean_num_users=clean_meta["num_users"] if warm_start else None,
            model_cls=model_cls,
            dataset_cls=dataset_cls,
            metrics_cfg=metrics_cfg,
            checkpoint_mode=checkpoint_mode,
            targets=targets,
            clean_user_items=clean_meta["user_items"],
        )
```

把：

```python
    md_path = save_report(report, out_dir)
```

改为：

```python
    md_path = save_report(report, out_dir, name=attack_name)
```

- [ ] **Step 5: 把 Step 1-4 的相同改动应用到 pgd / bandwagon / random 的 fit.py**

唯一差异：Step 1 的 evaluate import 行分别为
`from attacks.pgd.evaluate import (...)`、`from attacks.bandwagon.evaluate import (...)`、
`from attacks.random.evaluate import (...)`，其余代码块完全相同。

- [ ] **Step 6: 更新四个 config.yaml 的 `evaluation.metrics`**

把（四个文件相同）：

```yaml
  metrics:                       # 方向标注：upper=越高越好 / lower=越低越好；@K 是评估 K 的唯一权威
  - recall@10: upper
  - ndcg@10: upper
```

改为：

```yaml
  metrics:                       # 方向标注：upper=越高越好 / lower=越低越好；@K 是评估 K 的唯一权威
  - target_ndcg@10: upper        # 攻击主选优指标：目标物品 NDCG@10（多目标取均值）
  - target_hr@10: upper          # 攻击辅助选优指标：目标物品 HR@10（多目标取均值）
  - recall@10: upper             # 模型效用（投毒代价参考）
  - ndcg@10: upper
```

- [ ] **Step 7: 语法检查 + 回归测试**

```powershell
G:\Idea\.venv\Scripts\python.exe -m py_compile G:\Idea\TPA\attacks\tpa\fit.py G:\Idea\TPA\attacks\pgd\fit.py G:\Idea\TPA\attacks\bandwagon\fit.py G:\Idea\TPA\attacks\random\fit.py
G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_training_metrics tests.test_modes tests.test_attack_eval -v
```

Expected: 无语法错误；全部 PASS。

- [ ] **Step 8: Commit**

```bash
git -C G:\Idea add TPA/attacks/tpa/fit.py TPA/attacks/tpa/config.yaml TPA/attacks/pgd/fit.py TPA/attacks/pgd/config.yaml TPA/attacks/bandwagon/fit.py TPA/attacks/bandwagon/config.yaml TPA/attacks/random/fit.py TPA/attacks/random/config.yaml
git -C G:\Idea commit -m "feat(attacks): fit.py 接入目标物品选优（4 攻击 + config）"
```

---

### Task 3: 技能模板同步（paper-code-implementation）

**Files:**
- Modify: `.codex/skills/paper-code-implementation/assets/attack-imp-direct-poison/evaluate.py`
- Modify: `.codex/skills/paper-code-implementation/assets/attack-imp-direct-poison/fit.py`
- Modify: `.codex/skills/paper-code-implementation/assets/attack-imp-direct-poison/config.yaml`
- Modify: `.codex/skills/paper-code-implementation/SKILL.md`
- Modify: `.codex/skills/paper-code-implementation/references/config_template.yaml`
- Modify: `.codex/skills/paper-code-implementation/references/attack_imp_direct_poison.md`

**Interfaces:**
- Consumes: Task 1 的共享模块与 Task 2 的 fit.py 改动模式。
- Produces: 模板生成的攻击项目默认按 `target_ndcg@K` 选优。

- [ ] **Step 1: 模板 evaluate.py 薄壳化**

把 `assets/attack-imp-direct-poison/evaluate.py` 整个内容替换为：

```python
"""攻击模板评估 —— 共享实现见 evaluation/attack_eval.py"""
from evaluation.attack_eval import (  # noqa: F401
    build_attack_eval_metrics,
    compare_models,
    compute_target_metrics,
    format_report,
    ranking_scores,
    save_report,
)
```

- [ ] **Step 2: 模板 fit.py 接入目标物品选优（完整改动）**

2a) import 块：删除 `from evaluation.metrics import compute_metrics`；把：

```python
from attacks.attack_imp_direct_poison.evaluate import compare_models, ranking_scores, save_report
```

改为：

```python
from attacks.attack_imp_direct_poison.evaluate import (
    build_attack_eval_metrics,
    compare_models,
    ranking_scores,
    save_report,
)
```

并把 `training.metrics` 导入中的 `match_metric_values` 删除，只保留
`BestTracker, eval_ks_from_metrics, safe_checkpoint_name`。

2b) `train_poisoned_model` 签名改为：

```python
def train_poisoned_model(cfg: TrainingConfig, poisoned_meta: Dict[str, Any],
                         out_dir: Path, warm_start: bool,
                         warm_ckpt: Path | None, clean_num_users: int | None,
                         model_cls, dataset_cls, metrics_cfg,
                         checkpoint_mode: str = "per_metric",
                         targets: List[int] | None = None,
                         clean_user_items: Dict[int, set] | None = None,
                         ) -> Tuple[Any, List[Dict[str, Any]]]:
```

2c) 评估 epoch 指标计算块改为：

```python
        if epoch % eval_every == 0 or epoch == 1:
            scores, users, test_pos_local = ranking_scores(model, poisoned_meta["test_pairs"])
            ks = eval_ks_from_metrics(metrics_cfg, k)
            res, target_details = build_attack_eval_metrics(
                scores, users, user_items, test_pos_local,
                clean_user_items or {}, targets or [], ks,
                list(tracker.directions),
            )
            entry.update(res)
            if target_details:
                entry["targets"] = target_details
```

2d) `main()` 中 `train_poisoned_model` 调用改为：

```python
        model, history = train_poisoned_model(
            cfg, poisoned_meta, out_dir, warm_start, warm_ckpt,
            clean_num_users=clean_meta["num_users"] if warm_start else None,
            model_cls=model_cls,
            dataset_cls=dataset_cls,
            metrics_cfg=metrics_cfg,
            checkpoint_mode=checkpoint_mode,
            targets=targets,
            clean_user_items=clean_meta["user_items"],
        )
```

2e) 报告输出改为：

```python
    md_path = save_report(report, out_dir, name=attack_name)
```

- [ ] **Step 3: 模板 config.yaml 的 `evaluation.metrics` 重排（K=20）**

把：

```yaml
  metrics:                       # 方向标注：upper=越高越好 / lower=越低越好；@K 是评估 K 的唯一权威
  - recall@20: upper
  - ndcg@20: upper
```

改为：

```yaml
  metrics:                       # 方向标注：upper=越高越好 / lower=越低越好；@K 是评估 K 的唯一权威
  - target_ndcg@20: upper        # 攻击主选优指标：目标物品 NDCG@20（多目标取均值）
  - target_hr@20: upper          # 攻击辅助选优指标：目标物品 HR@20（多目标取均值）
  - recall@20: upper             # 模型效用（投毒代价参考）
  - ndcg@20: upper
```

- [ ] **Step 4: 更新 SKILL.md「多指标最优 checkpoint」段落**

在 `SKILL.md` 中该段落（以 `- \`--skip-train\` 加载顺序：...` 开头的一行）之后
插入一行新 bullet：

```markdown
- 攻击实验选优主指标：`target_ndcg@K`（被攻击目标物品的 NDCG，多目标取均值），
  `target_hr@K` 为辅助；整体 recall/ndcg 仅作投毒代价参考。统一实现在
  `evaluation/attack_eval.py`（build_attack_eval_metrics / aggregate_target_metrics），
  攻击模块 evaluate.py 只是薄壳，禁止各自重写。
```

- [ ] **Step 5: 更新 `references/config_template.yaml` 的 evaluation 段**

把：

```yaml
evaluation:
  metrics:                                          # [paper] 见论文 Table 4
  - recall@20: upper                                # upper=越高越好 / lower=越低越好（显式标注优先）
  - ndcg@20: upper
```

改为：

```yaml
evaluation:
  metrics:                                          # [paper] 见论文 Table 4
  - target_ndcg@20: upper                           # 攻击实验主选优指标（多目标取均值）
  - target_hr@20: upper                             # 攻击辅助选优指标
  - recall@20: upper                                # 模型效用（投毒代价参考）
  - ndcg@20: upper
```

- [ ] **Step 6: 更新 `references/attack_imp_direct_poison.md`**

1. 第 3 节文件职责表 `evaluate.py` 行改为：

```markdown
| `evaluate.py` | HR@K / NDCG@K / 模型效用报告（薄壳，共享实现见 `evaluation/attack_eval.py`） |
```

2. 第 4 节 model 阶段校验项追加一条：

```markdown
- 中毒模型 checkpoint 按 `target_ndcg@K` 选优（`--skip-train` 默认加载
  `target_ndcg@K-best-model.pt`）；整体 recall/ndcg 仅作投毒代价参考
```

3. 第 6 节交付清单"多指标最优 checkpoint"条目末尾追加：
`攻击实验主选优指标为 \`target_ndcg@K\`（见 SKILL.md「多指标最优 checkpoint」）`

- [ ] **Step 7: 语法检查**

```powershell
G:\Idea\.venv\Scripts\python.exe -m py_compile G:\Idea\.codex\skills\paper-code-implementation\assets\attack-imp-direct-poison\evaluate.py G:\Idea\.codex\skills\paper-code-implementation\assets\attack-imp-direct-poison\fit.py
```

Expected: 无语法错误。

- [ ] **Step 8: Commit**

```bash
git -C G:\Idea add -- .codex/skills/paper-code-implementation
git -C G:\Idea commit -m "feat(skill): paper-code-implementation 模板同步目标物品选优"
```

---

### Task 4: 文档更新 + 端到端冒烟验证 + 总体验收

**Files:**
- Modify: `TPA/attacks/tpa/docs/USAGE.md`、`TPA/attacks/pgd/docs/USAGE.md`、
  `TPA/attacks/bandwagon/docs/USAGE.md`、`TPA/attacks/random/docs/USAGE.md`
- Modify: `TPA/attacks/tpa/docs/DESIGN.md`

**Interfaces:**
- Consumes: Task 2 完成后的可运行攻击 fit 流程。
- Produces: 文档说明 + 冒烟产物（临时目录，验证后删除）。

- [ ] **Step 1: 更新四个攻击 USAGE.md 的配置文件详解 evaluation 段**

在每个 `USAGE.md` 的 `evaluation:` 配置块之后插入：

```markdown
> 攻击选优：`evaluation.metrics` 的 `target_ndcg@K` / `target_hr@K` 是中毒模型
> checkpoint 的选优指标（主指标 = `target_ndcg@K`，`--skip-train` 与对比报告
> 都加载按它选出的最优模型）；整体 `recall@K` / `ndcg@K` 仅作投毒代价参考。
> 每个评估 epoch 的目标物品明细（hr/ndcg/命中人数/平均排名）写入 `history.json`。
```

（tpa 的 USAGE.md 中该块位于"## 3. 配置文件详解"的 `output:` 之前；
pgd / bandwagon / random 的 USAGE.md 结构相同，按同名标题定位插入。）

- [ ] **Step 2: 更新 `TPA/attacks/tpa/docs/DESIGN.md` 评估协议段**

在"攻击效果：目标物品 Clean/Poisoned 的 **HR@K** 与 **NDCG@K**，外加平均排名"
一行之后插入：

```markdown
- checkpoint 选优：中毒模型按目标物品 `target_ndcg@K` / `target_hr@K` 选优
  （主指标 `target_ndcg@K`，`--skip-train` 与对比报告加载它），
  整体 recall/ndcg 仅作投毒代价参考
```

- [ ] **Step 3: 端到端冒烟（2 epoch，临时 tag 与临时输出目录）**

1) 复制 tpa config 到临时文件：

```powershell
Copy-Item -LiteralPath 'G:\Idea\TPA\attacks\tpa\config.yaml' -Destination 'G:\Idea\tmp\smoke_tpa_fit.yaml'
```

2) 用 apply_patch 对 `G:\Idea\tmp\smoke_tpa_fit.yaml` 做三处修改：

```yaml
run_tag: smoke-target-metrics
```

```yaml
training:
  epochs: 2
  eval_every: 1
```

```yaml
output:
  dir: tmp/smoke-outputs
```

3) 生成中毒数据 + 训练（白盒模式，路径缓存已存在）：

```powershell
G:\Idea\.venv\Scripts\python.exe G:\Idea\TPA\attacks\tpa\run.py --config G:\Idea\tmp\smoke_tpa_fit.yaml --mode data
G:\Idea\.venv\Scripts\python.exe G:\Idea\TPA\attacks\tpa\run.py --config G:\Idea\tmp\smoke_tpa_fit.yaml --mode model
```

4) 验证产物：

```powershell
Get-ChildItem 'G:\Idea\tmp\smoke-outputs\ml100k\lightgcn\smoke-target-metrics\checkpoints' | Select-Object Name
Get-Content 'G:\Idea\tmp\smoke-outputs\ml100k\lightgcn\smoke-target-metrics\history.json' -Encoding UTF8 | Select-String 'target_ndcg@10|"targets"'
```

Expected:
- checkpoints 含 `target_ndcg@10-best-model.pt` 与 `target_hr@10-best-model.pt`
  （另有 recall@10 / ndcg@10）；
- history.json 的 `best` 段含 `target_ndcg@10`；评估 epoch 条目含 `"targets"`；
- 对比报告文件存在且标题为"投毒攻击对比报告（Top-10）"。

5) `--skip-train` 加载链验证：

```powershell
G:\Idea\.venv\Scripts\python.exe G:\Idea\TPA\attacks\tpa\fit.py --config G:\Idea\tmp\smoke_tpa_fit.yaml --skip-train
```

Expected: 日志显示
`[fit] 从 checkpoint 加载中毒模型: ...target_ndcg@10-best-model.pt`。

6) 清理冒烟产物（先确认路径，再删除）：

```powershell
Remove-Item -LiteralPath 'G:\Idea\tmp\smoke_tpa_fit.yaml' -Force
Remove-Item -LiteralPath 'G:\Idea\tmp\smoke-outputs' -Recurse -Force
Remove-Item -LiteralPath 'G:\Idea\TPA\attacks\tpa\data\poisoned\ml100k\lightgcn\smoke-target-metrics' -Recurse -Force
Remove-Item -LiteralPath 'G:\Idea\TPA\attacks\tpa\data\poisoned\ml100k\lightgcn\latest.json' -Force -ErrorAction SilentlyContinue
```

（白盒 smoke 的 data 阶段会在 `poisoned/ml100k/lightgcn/` 写 `latest.json`
指针；冒烟前该文件不存在，删除以恢复原状。）

- [ ] **Step 4: 总体验收**

```powershell
G:\Idea\.venv\Scripts\python.exe -m py_compile G:\Idea\TPA\attacks\tpa\fit.py G:\Idea\TPA\attacks\pgd\fit.py G:\Idea\TPA\attacks\bandwagon\fit.py G:\Idea\TPA\attacks\random\fit.py G:\Idea\TPA\evaluation\attack_eval.py
G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_training_metrics tests.test_modes tests.test_attack_eval -v
```

Expected: 无语法错误；全部 PASS。

- [ ] **Step 5: Commit**

```bash
git -C G:\Idea add TPA/attacks/tpa/docs/USAGE.md TPA/attacks/tpa/docs/DESIGN.md TPA/attacks/pgd/docs/USAGE.md TPA/attacks/bandwagon/docs/USAGE.md TPA/attacks/random/docs/USAGE.md
git -C G:\Idea commit -m "docs: 更新攻击 USAGE/DESIGN 说明"
```
