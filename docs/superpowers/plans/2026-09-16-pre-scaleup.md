# pre 跨数据集放大实验（Batch A）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: 用 `executing-plans`（或
> `subagent-driven-development`）按任务逐条实施本计划。步骤用 `- [ ]` 复选框跟踪。

**Goal:** 把 ml100k 上的「交互退化 -> 表示位移 -> 投毒攻击的恢复」结论，升级为可跨
ml100k / gowalla / amazon-book 复现、可审计、可断点续跑的 Batch A 实验体系，并冻结
Batch B 的接口边界。

**Architecture:** 三层严格分层——`pre/runners/`（怎么跑一个实验）、`pre/analysis/`
（纯函数口径 + 派生表）、`pre/schedule/`（何时 / 在哪张卡 / 跑哪些 shard）。实验定义先
冻成产物（`pre/targets/<dataset>.json`）与协议号（`experimental_protocol_version`），
再进入矩阵。

**Tech Stack:** Python 3.12、PyTorch 2.5.1+cu121、numpy、scipy.sparse、stdlib unittest、
PyYAML、matplotlib（仅出图）；不新增第三方依赖。

**设计依据:** `docs/superpowers/specs/2026-09-16-pre-scaleup-design.md`（第 1–6 节，已冻结）

## Global Constraints

- 注释中文；来源标注 `[paper]/[ai]/[unreported]/[官方代码]`；配置键四要素（为什么 /
  是什么 / 公式或出处 / 取值举例）。
- 环境 `G:\Idea\.venv\Scripts\python.exe`；测试 stdlib unittest；**不新增第三方依赖**。
- 测试命令（工作目录 `G:\Idea\TPA`）：主测试
  `G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_xxx -v`；pre 层测试
  `G:\Idea\.venv\Scripts\python.exe -m unittest pre.tests.test_xxx -v`。
- 全量回归：`G:\Idea\.venv\Scripts\python.exe -m unittest discover -s tests -t . -v`
  （Task 0 记录基线，交付前必须全绿）。
- 提交规范：Conventional Commits `type(scope): 中文描述`；只用 `git add <明确路径>`，
  禁止 `git add -f` 与 `git add -A`。
- **Protocol Freeze Principle**（spec §0.4）：Gate 0a 开始后，任何改变实验定义 / 数据生成 /
  预算 / 目标集 / 分析主口径 / Gate 判据的修改都必须回到 spec 评审；普通工程修复可通过
  单测继续，但必须记录 provenance。
- 主口径固定：alignment=`procrustes`（拟合掩码**排除目标物品**）、distance=`l2`、K=10
  （K=20 为补充）；`RR_pc` 为机制主指标，`RR` 为原始主指标，两者并列保留。
- 跨数据集表**禁止**出现绝对距离（`Delta_abs` / `d_deg` / `d_ctrl`），由单元测试强制。
- 产物分层：`outputs/<tag>/{raw,analysis/{legacy,corrected},tables,figures}/` + `manifest.json`；
  `raw/checkpoint/log` 不入库，`analysis/tables/figures/plan.json/coverage.*` 入白名单。
- GPU：不硬编码卡号，运行时探测 + `CUDA_VISIBLE_DEVICES` 绑定；保留 1 张 debug 卡。
- 命名约定：`<batch>` = 一次批量实验（如 `batchA`，其下放 `plan.json`/`coverage.*`）；
  `<tag>` = 其下的 shard 级目录（如 `batchA-gowalla-lightgcn-s42-item0016`）。

## 文件结构（本计划新增 / 修改）

新增：

```
TPA/pre/protocol.py                       协议号 + manifest 字段表（单一事实来源）
TPA/pre/analysis/metrics.py               距离 / 恢复量纯函数
TPA/pre/analysis/decomposition.py         径向-切向恒等式
TPA/pre/analysis/placebo.py               d_ctrl / 超噪声位移 E / pc_valid
TPA/pre/analysis/concentration.py         F 的集中度与方向余弦
TPA/pre/analysis/audience.py              固定受众 / 剩余受众
TPA/pre/analysis/schema.py                跨数据集表 schema 校验
TPA/pre/analysis/plots.py                 PDF + SVG 出图
TPA/pre/targets/<dataset>.json            冻结目标集（入库）
TPA/pre/schedule/plan.py                  分片计划器
TPA/pre/schedule/run.py                   调度器（GPU 探测 / 绑定 / resume）
TPA/pre/schedule/verify.py                完整性与覆盖率校验
TPA/pre/docs/SERVER_RUNBOOK.md            运行手册
TPA/pre/docs/ANALYSIS_PROTOCOL.md         分析协议（数学口径 + 版本号）
TPA/pre/tests/test_pre_*.py               新增单测
docs/superpowers/specs/2026-09-16-batchB-attack-porting-design.md   （P10）
```

修改：

```
TPA/pre/run.py                            新增 --seed/--seeds/--refresh-targets/--preflight
TPA/pre/runners/common.py                 路径助手（raw_root/cond_dir）+ 去重断言
TPA/pre/runners/pipeline.py               seed 维度、shared 退化数据、冻结目标
TPA/pre/runners/targets.py                冻结产物写出 / 读取 / 指纹校验
TPA/pre/runners/degrade.py                去重断言、shared 目录、指纹
TPA/pre/runners/train_model.py            路径改走 cond_dir（含 seed 段）
TPA/pre/runners/run_attack.py             路径改走 cond_dir；TPA pre_stage 传退化 meta
TPA/pre/analysis/{align,aggregate,exposure,distances}.py
TPA/attacks/uba/{generate,uplift}.py      处理效应缓存 key + 行身份
TPA/attacks/batch/registry.py             注册 advinject
TPA/attacks/tpa/{path_builder,generate}.py 数据源可重定向
TPA/pre/configs/default.yaml              deletion_seed / 预算双归一化 / 协议号
.gitignore                                论文资产白名单
```

---

## Task 0: 基线与协议号（P0）

**Files:**
- Create: `TPA/pre/protocol.py`
- Create: `TPA/pre/docs/SERVER_RUNBOOK.md`（骨架，Task 25 完稿）
- Test: `TPA/pre/tests/test_pre_protocol.py`

**Interfaces:**
- Consumes: 无
- Produces: `PROTOCOL_VERSION: str`、`PROTOCOL_FIELDS: tuple[str, ...]`、
  `sha256_file(path: Path) -> str`、
  `manifest_fields(cfg, *, dataset, victim_model, target_file_sha256, ratio=None,
  attack=None, model_seed=None, deletion_seed=None, attack_seed=None, k=None,
  budget=None) -> Dict[str, Any]`

- [ ] **Step 1: 记录回归基线**（写入本次会话记录，后续每次提交前对比）

Run: `G:\Idea\.venv\Scripts\python.exe -m unittest discover -s tests -t . -v`
（工作目录 `G:\Idea\TPA`）
Expected: 全部通过（上一轮基线 268 通过；若已有历史失败项，先逐条记录，避免与本次引入的
失败混淆）

- [ ] **Step 2: 写失败测试** —— `TPA/pre/tests/test_pre_protocol.py`

```python
# -*- coding: utf-8 -*-
"""协议号与 manifest 字段表（spec §2.7）。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

TPA_ROOT = Path(__file__).resolve().parents[2]
if str(TPA_ROOT) not in sys.path:
    sys.path.insert(0, str(TPA_ROOT))

from pre.protocol import (PROTOCOL_FIELDS, PROTOCOL_VERSION,  # noqa: E402
                          manifest_fields, sha256_file)


class TestProtocol(unittest.TestCase):
    def test_fields_are_13_and_ordered(self):
        self.assertEqual(len(PROTOCOL_FIELDS), 14)
        self.assertEqual(PROTOCOL_FIELDS[0], "experimental_protocol_version")
        self.assertIn("code_commit", PROTOCOL_FIELDS)

    def test_manifest_fields_covers_every_declared_field(self):
        cfg = {"dataset": "gowalla", "k": 10,
               "pre": {"attack": {"num_fake_users": 884, "filler_size": 9}}}
        m = manifest_fields(cfg, dataset="gowalla", victim_model="lightgcn",
                            target_file_sha256="deadbeef", ratio=0.5, attack="uba",
                            model_seed=42, deletion_seed=42, attack_seed=42,
                            k=10)
        for f in PROTOCOL_FIELDS:
            self.assertIn(f, m, f"manifest 缺字段 {f}")
        self.assertEqual(m["experimental_protocol_version"], PROTOCOL_VERSION)
        self.assertEqual(m["ratio"], 0.5)

    def test_sha256_file_missing_returns_empty(self):
        self.assertEqual(sha256_file(Path("no/such/file.bin")), "")

    def test_sha256_file_detects_change(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            a = Path(d) / "a.bin"; b = Path(d) / "b.bin"
            a.write_bytes(b"hello"); b.write_bytes(b"hello!")
            self.assertNotEqual(sha256_file(a), sha256_file(b))
            self.assertEqual(len(sha256_file(a)), 64)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: 运行确认失败**

Run: `G:\Idea\.venv\Scripts\python.exe -m unittest pre.tests.test_pre_protocol -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'pre.protocol'`）

- [ ] **Step 4: 实现 `TPA/pre/protocol.py`**

```python
# -*- coding: utf-8 -*-
"""Batch A 实验协议号与 manifest 字段表（单一事实来源）。

为什么需要它（设计动机）：
    spec §2.7 规定每个 manifest 记录 14 个字段，用于把论文数字反向追踪到
    `Figure -> analysis output -> raw CSV -> shard -> config -> target.json -> git commit`。
    如果字段名散落在多个写入点，迟早出现"某次跑的 manifest 少一个字段但没人发现"。
    这里把协议号与字段表集中定义，所有写入点都从这里取。

功能是什么：
    PROTOCOL_VERSION  协议版本号；任何改变实验定义/口径的变更都必须递增它。
    PROTOCOL_FIELDS   manifest 必须出现的字段名（顺序即写入顺序）。
    manifest_fields() 由 cfg + 运行上下文组装 manifest 元数据。
    sha256_file()     文件指纹（meta / target 校验用）。

参考出处：spec §2.7（provenance 链）、§1.8（meta fingerprint）、§5.8（git_commit）。

使用举例：
    from pre.protocol import manifest_fields, sha256_file
    m = manifest_fields(cfg, dataset="gowalla", victim_model="lightgcn",
                        target_file_sha256=sha256_file(pre_targets_path("gowalla")),
                        ratio=0.5, attack="uba", model_seed=42)
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

PROTOCOL_VERSION = "batchA-2026-09-16"   # [ai] 与 spec 日期绑定；协议变更必须改这里

PROTOCOL_FIELDS: Sequence[str] = (
    "experimental_protocol_version",  # 协议版本
    "dataset",                        # 数据集名
    "victim_model",                   # 受害模型名（lightgcn / mf）
    "target_file_sha256",             # pre/targets/<dataset>.json 的指纹
    "deletion_seed",                  # 删除排列种子（自变量）
    "model_seed",                     # 模型初始化/负采样种子
    "attack_seed",                    # 攻击生成种子（= model_seed）
    "ratio",                          # 删除比例（None = 与比例无关的产物）
    "rho_A",                          # 假用户占用户数比例
    "rho_P",                          # 每个假用户画像条数 / 平均用户度数
    "rho_E",                          # 注入交互 / 训练交互（实际预算记录）
    "attack",                         # 攻击臂（random/bandwagon/pgd/uba/degradation-only）
    "K",                              # 评估 Top-K
    "code_commit",                    # git HEAD
)


def sha256_file(path) -> str:
    """文件 sha256；文件不存在返回空串（调用方负责判定"缺失即报错"）。

    为什么返回空串而不是抛异常：指纹校验点在热路径上，缺文件的情形由更上层的
    preflight 明确报错；这里只负责"缺失可识别"（空串不可能等于任何真实摘要）。
    """
    p = Path(path)
    if not p.exists():
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _budget_from_cfg(cfg: Dict[str, Any],
                     budget: Optional[Dict[str, float]]) -> Dict[str, float]:
    """预算三元组：显式传入优先，否则从 cfg.pre.attack 读（缺省 0.0）。"""
    if budget:
        return {k: float(budget.get(k, 0.0)) for k in ("rho_A", "rho_P", "rho_E")}
    atk = (cfg.get("pre", {}) or {}).get("attack", {}) or {}
    return {"rho_A": float(atk.get("rho_A", 0.0) or 0.0),
            "rho_P": float(atk.get("rho_P", 0.0) or 0.0),
            "rho_E": float(atk.get("rho_E", 0.0) or 0.0)}


def manifest_fields(cfg: Dict[str, Any], *, dataset: str, victim_model: str,
                    target_file_sha256: str, ratio: Optional[float] = None,
                    attack: Optional[str] = None,
                    model_seed: Optional[int] = None,
                    deletion_seed: Optional[int] = None,
                    attack_seed: Optional[int] = None,
                    k: Optional[int] = None,
                    budget: Optional[Dict[str, float]] = None
                    ) -> Dict[str, Any]:
    """按 PROTOCOL_FIELDS 组装 manifest 元数据（缺项填 None/0，不省略键）。

    矩阵注释（为什么用 dict 而不是 dataclass）：
        manifest 直接 json 落盘，字段随协议演进增减；dict + 集中字段表比
        每加一个字段就改一处 dataclass 更不容易漏写。
    """
    from pre.runners.common import git_commit          # 延迟 import 避免环依赖

    seed = model_seed if model_seed is not None else \
        int(((cfg.get("pre", {}) or {}).get("training", {}) or {}).get("seeds", [42])[0])
    b = _budget_from_cfg(cfg, budget)
    out: Dict[str, Any] = {
        "experimental_protocol_version": PROTOCOL_VERSION,
        "dataset": str(dataset),
        "victim_model": str(victim_model),
        "target_file_sha256": str(target_file_sha256 or ""),
        "deletion_seed": int(deletion_seed) if deletion_seed is not None else None,
        "model_seed": int(seed),
        "attack_seed": int(attack_seed) if attack_seed is not None else int(seed),
        "ratio": float(ratio) if ratio is not None else None,
        "rho_A": b["rho_A"], "rho_P": b["rho_P"], "rho_E": b["rho_E"],
        "attack": attack,
        "K": int(k) if k is not None else int(cfg.get("k", 10)),
        "code_commit": git_commit(),
    }
    return {f: out[f] for f in PROTOCOL_FIELDS}
```

- [ ] **Step 5: 运行确认通过**

Run: `G:\Idea\.venv\Scripts\python.exe -m unittest pre.tests.test_pre_protocol -v`
Expected: PASS（4 个用例）

- [ ] **Step 6: 写 runbook 骨架 `TPA/pre/docs/SERVER_RUNBOOK.md`**

内容至少含 5 个小节标题（正文留到 Task 25 完稿，但标题与占位结构现在就固定）：

```markdown
# Batch A 服务器运行手册

## 0. 前置条件（preflight）
## 1. 环境搭建（独立 venv + requirements.txt）
## 2. 数据与目标集检查
## 3. 分片与 GPU 分配
## 4. 启动 / 监控 / 断点续跑
## 5. 完整性校验与 Gate 判据
## 6. 产物回传与白名单
```

- [ ] **Step 7: 提交**

```bash
git add TPA/pre/protocol.py TPA/pre/tests/test_pre_protocol.py TPA/pre/docs/SERVER_RUNBOOK.md
git commit -m "feat(pre): 新增协议号与 manifest 字段表（Batch A P0）"
```

---

## Task 1: 目标集冻结产物（spec §1.3 / §2.6）

**Files:**
- Modify: `TPA/pre/analysis/exposure.py`（抽出纯函数 `exposure_from_scores`，补 `ndcg@k`）
- Modify: `TPA/pre/runners/targets.py`（冻结产物写出 / 读取 / 指纹校验）
- Modify: `TPA/pre/runners/pipeline.py`（`phase_targets` 落冻结文件）
- Modify: `TPA/pre/run.py`（新增 `--refresh-targets`）
- Create: `TPA/pre/targets/{ml100k,gowalla,amazon-book}.json`（**入库**）
- Test: `TPA/pre/tests/test_pre_targets.py`

**Interfaces:**
- Consumes: `pre.protocol.sha256_file`
- Produces:
  - `exposure_from_scores(scores, users, meta, k) -> Dict[int, Dict[str, float]]`
    （键 `hr@k` / `ndcg@k` / `n_elig` / `hits`）
  - `freeze_payload(targets, meta, *, exposure=None, k=10) -> Dict[str, Any]`
  - `frozen_path(dataset: str) -> Path`（`TPA/pre/targets/<dataset>.json`）
  - `write_frozen_targets(cfg, targets, meta, exposure, meta_path, refresh=False) -> Path`
  - `load_frozen_targets(dataset) -> Dict[str, Any]`（缺失即 `FileNotFoundError`）
  - `verify_frozen_fingerprint(frozen, meta_path) -> None`（不一致抛 `ValueError`）

- [ ] **Step 1: 写失败测试** —— `TPA/pre/tests/test_pre_targets.py`

```python
# -*- coding: utf-8 -*-
"""目标集冻结产物（spec §1.3 / §2.6）：字段齐全、指纹可校验、缺文件即报错。"""
from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch

TPA_ROOT = Path(__file__).resolve().parents[2]
if str(TPA_ROOT) not in sys.path:
    sys.path.insert(0, str(TPA_ROOT))

from pre.analysis.exposure import exposure_from_scores            # noqa: E402
from pre.runners.targets import (build_targets, freeze_payload,   # noqa: E402
                                 verify_frozen_fingerprint,
                                 write_frozen_targets)


def _toy_meta():
    """4 个物品；用户 0/1 参与评估；只有 (1, 0) 在训练集里（故物品 0 对用户 1 不合格）。"""
    return {"num_users": 4, "num_items": 4,
            "train_pairs": [(1, 0), (2, 0), (2, 1), (3, 2)],
            "test_pairs": [(0, 1), (1, 2)], "user_items": {}}


class TestExposureFromScores(unittest.TestCase):
    def test_hr_and_ndcg_with_rank_discount(self):
        scores = torch.tensor([[3.0, 2.0, 1.0, 0.0],
                               [0.0, 3.0, 2.0, 1.0]])
        exp = exposure_from_scores(scores, [0, 1], _toy_meta(), k=2)
        # 物品 0：用户 0 rank1 命中；用户 1 因训练集有 (1,0) 不合格 -> 分母 1
        self.assertAlmostEqual(exp[0]["hr@2"], 1.0, places=6)
        self.assertAlmostEqual(exp[0]["ndcg@2"], 1.0, places=6)
        # 物品 1：用户 0 rank2 -> 1/log2(3)；用户 1 rank1 -> 1.0
        want = (1.0 / math.log2(3) + 1.0) / 2
        self.assertAlmostEqual(exp[1]["hr@2"], 1.0, places=6)
        self.assertAlmostEqual(exp[1]["ndcg@2"], want, places=6)

    def test_items_without_eligible_users_are_absent(self):
        scores = torch.zeros(2, 4)
        meta = _toy_meta()
        meta["train_pairs"] = [(0, 3), (1, 3)]     # 两个评估用户都交互过物品 3
        exp = exposure_from_scores(scores, [0, 1], meta, k=2)
        self.assertNotIn(3, exp)


class TestFreezePayload(unittest.TestCase):
    def test_required_fields_present(self):
        meta = _toy_meta()
        exposure = {1: {"hr@10": 0.5, "ndcg@10": 0.25}, 2: {"hr@10": 0.25, "ndcg@10": 0.1}}
        picked = [(1, 2, 0.5), (2, 1, 0.25)]
        t = build_targets(picked, strategy="top_exposure", dataset="toy", seed=42,
                          min_interactions=1, k=10, source_model="lightgcn",
                          exposure_metric="hr@10")
        frozen = freeze_payload(t, meta, exposure=exposure, k=10)
        self.assertEqual(frozen["dataset"], "toy")
        self.assertEqual(frozen["selection_seed"], 42)
        self.assertEqual(len(frozen["sha256_meta"]), 0)     # 未传 meta_path 时为 ""
        for it in frozen["items"]:
            for key in ("item_id", "N_i", "q_i", "interaction_percentile",
                        "clean_hr@10", "clean_ndcg@10"):
                self.assertIn(key, it, f"item 缺字段 {key}")
        self.assertEqual(frozen["items"][0]["N_i"], 2)      # 物品 1 有 1 条 -> 见 _toy_meta

    def test_percentile_is_monotone_in_interaction_count(self):
        meta = _toy_meta()
        exposure = {}
        picked = [(1, 2, 0.0), (2, 1, 0.0)]
        t = build_targets(picked, strategy="top_popularity", dataset="toy", seed=42,
                          min_interactions=1, k=10)
        frozen = freeze_payload(t, meta, exposure=exposure, k=10)
        p = {it["item_id"]: it["interaction_percentile"] for it in frozen["items"]}
        self.assertGreater(p[1], p[2])                      # 物品 1 更热门 -> 分位更高


class TestWriteAndVerify(unittest.TestCase):
    def test_write_then_load_and_fingerprint_mismatch_raises(self):
        meta = _toy_meta()
        with tempfile.TemporaryDirectory() as d:
            meta_path = Path(d) / "meta.pkl"
            import pickle
            meta_path.write_bytes(pickle.dumps(meta))
            picked = [(1, 2, 0.0), (2, 1, 0.0)]
            t = build_targets(picked, strategy="top_popularity", dataset="toy",
                              seed=42, min_interactions=1, k=10)
            cfg = {"dataset": "toy", "k": 10}
            out = write_frozen_targets(cfg, t, meta, {}, meta_path,
                                       refresh=True, out_dir=Path(d))
            self.assertTrue(out.exists())
            frozen = verify_frozen_fingerprint(
                __import__("json").loads(out.read_text(encoding="utf-8")), meta_path)
            self.assertEqual(frozen["dataset"], "toy")
            meta_path.write_bytes(pickle.dumps({**meta, "num_items": 5}))
            with self.assertRaises(ValueError):
                verify_frozen_fingerprint(
                    __import__("json").loads(out.read_text(encoding="utf-8")), meta_path)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行确认失败**

Run: `G:\Idea\.venv\Scripts\python.exe -m unittest pre.tests.test_pre_targets -v`
Expected: FAIL（`ImportError: cannot import name 'exposure_from_scores'`）

- [ ] **Step 3: 重构 `exposure.py`** —— 把打分逻辑抽成纯函数，并补 `ndcg@k`

```python
def exposure_from_scores(scores: torch.Tensor, users: Sequence[int],
                         meta: Dict[str, Any], k: int = 10
                         ) -> Dict[int, Dict[str, float]]:
    """由打分矩阵算全物品原生 HR@k / NDCG@k（合格用户口径）。

    合格用户 = 训练集未与该物品交互过的评估用户（否则热门物品被自己的消费者刷高）。
    NDCG@k（单相关物品）：命中排名 r 时增益 1/log2(r+1)，否则 0；IDCG = 1。

    矩阵变换过程：
        scores (U, I)  -> topk.indices (U, k)            每行是该用户的 Top-k 物品
        gains  (k,)    = 1/log2(2..k+1)                 排名折扣（rank 从 1 起）
        hit    (U, I)  = scatter(True 到 topk)          是否命中
        dcg    (U, I)  = scatter(gains 到 topk)         命中增益
        elig   (U, I)  = 训练集未交互                    合格掩码
        n_elig (I,) = elig.sum(0)；n_hit (I,) = (hit & elig).sum(0)
        HR@k[j]   = n_hit[j] / n_elig[j]
        NDCG@k[j] = (dcg * elig).sum(0)[j] / n_elig[j]
    """
    num_items = int(meta["num_items"])
    topk = torch.topk(scores, k, dim=1).indices                     # (U, k)
    gains = 1.0 / torch.log2(torch.arange(2, k + 2, dtype=torch.float32))  # (k,)
    hit = torch.zeros(scores.shape[0], num_items, dtype=torch.bool)
    dcg = torch.zeros(scores.shape[0], num_items, dtype=torch.float32)
    hit.scatter_(1, topk, True)
    dcg.scatter_(1, topk, gains.unsqueeze(0).expand(topk.shape[0], k))
    elig = torch.ones(scores.shape[0], num_items, dtype=torch.bool)
    row_of = {int(u): r for r, u in enumerate(users)}
    for u, i in meta["train_pairs"]:
        r = row_of.get(int(u))
        if r is not None and 0 <= int(i) < num_items:
            elig[r, int(i)] = False
    n_elig = elig.sum(dim=0)
    n_hit = (hit & elig).sum(dim=0)
    dcg_sum = (dcg * elig).sum(dim=0)
    out: Dict[int, Dict[str, float]] = {}
    for j in range(num_items):
        ne = int(n_elig[j].item())
        if ne == 0:
            continue                      # 无合格用户 -> 无定义，不写 0 以免污染排序
        out[j] = {"hr@k": float(n_hit[j].item()) / ne,
                  "ndcg@k": float(dcg_sum[j].item()) / ne,
                  "n_elig": ne, "hits": int(n_hit[j].item())}
    return out


def native_exposure(model: Any, meta: Dict[str, Any], k: int = 10
                    ) -> Dict[int, Dict[str, float]]:
    """兼容旧签名（原实现保留语义，内部转调 exposure_from_scores）。"""
    S, users, _ = ranking_scores(model, meta["test_pairs"])
    return exposure_from_scores(S, users, meta, k)
```

- [ ] **Step 4: `targets.py` 加冻结产物**

```python
def frozen_path(dataset: str, out_dir: Optional[Path] = None) -> Path:
    """冻结目标集路径：TPA/pre/targets/<dataset>.json（入库，唯一权威）。"""
    base = Path(out_dir) if out_dir is not None else (TPA_ROOT / "pre" / "targets")
    return base / f"{dataset}.json"


def _percentile(counts: Dict[int, int], item_id: int, num_items: int) -> float:
    """交互数分位 = 交互数 <= N_i 的物品占比（经验 CDF）；越大越热门。"""
    n = int(counts.get(item_id, 0))
    le = sum(1 for c in counts.values() if c <= n)
    return float(le) / float(max(1, num_items))


def freeze_payload(targets: Dict[str, Any], meta: Dict[str, Any], *,
                   exposure: Optional[Dict[int, Dict[str, float]]] = None,
                   k: int = 10, meta_path: Optional[Path] = None) -> Dict[str, Any]:
    """把 targets 结构升级为冻结产物（spec §1.3/§2.6 的字段表）。"""
    counts = item_counts(meta["train_pairs"])
    num_items = int(meta["num_items"])
    items = []
    for t in targets["items"]:
        i = int(t["item_id"])
        e = (exposure or {}).get(i, {})
        items.append({
            "item_id": i,
            "N_i": int(counts.get(i, 0)),
            "q_i": float(counts.get(i, 0)) / float(max(1, num_items)),
            "interaction_percentile": _percentile(counts, i, num_items),
            f"clean_hr@{k}": float(e.get(f"hr@{k}", 0.0)) if e else None,
            f"clean_ndcg@{k}": float(e.get(f"ndcg@{k}", 0.0)) if e else None,
            "popularity_rank": t.get("popularity_rank"),
            "exposure_rank": t.get("exposure_rank"),
        })
    return {
        "dataset": targets["dataset"],
        "strategy": targets["strategy"],
        "selection_seed": int(targets["selection_seed"]),
        "exposure_source_model": targets.get("exposure_source_model"),
        "k": int(k),
        "num_items_in_dataset": num_items,
        "sha256_meta": sha256_file(meta_path) if meta_path else "",
        "created_at": now_iso(),
        "items": items,
    }


def write_frozen_targets(cfg: Dict[str, Any], targets: Dict[str, Any],
                         meta: Dict[str, Any], exposure, meta_path, *,
                         refresh: bool = False,
                         out_dir: Optional[Path] = None) -> Path:
    """写冻结文件；已存在且 refresh=False 时直接返回（不覆盖，保证可审计）。"""
    k = int(cfg.get("k", 10))
    path = frozen_path(str(cfg["dataset"]), out_dir)
    if path.exists() and not refresh:
        return path
    payload = freeze_payload(targets, meta, exposure=exposure, k=k,
                             meta_path=Path(meta_path) if meta_path else None)
    save_json(payload, path)
    return path


def load_frozen_targets(dataset: str, out_dir: Optional[Path] = None) -> Dict[str, Any]:
    """读冻结文件；缺失即报错（不允许现场重选目标，spec §2.6）。"""
    path = frozen_path(dataset, out_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"缺少冻结目标集 {path}；请先运行 python pre/run.py --mode targets")
    return read_json(path)


def verify_frozen_fingerprint(frozen: Dict[str, Any], meta_path: Path) -> Dict[str, Any]:
    """校验冻结文件记录的 meta 指纹与当前 meta 一致（不一致即报错）。"""
    actual = sha256_file(Path(meta_path))
    want = str(frozen.get("sha256_meta", ""))
    if want and actual and want != actual:
        raise ValueError(
            f"meta 指纹与冻结目标集不一致：{meta_path}\n  冻结={want}\n  当前={actual}")
    return frozen
```

同时把 `targets.py` 顶部 import 补齐：
`from pre.protocol import sha256_file`、`from pre.runners.common import TPA_ROOT`。

- [ ] **Step 5: `pipeline.py::phase_targets` 落冻结文件 + `run.py` 加 `--refresh-targets`**

```python
# pipeline.phase_targets 末尾（两种 strategy 分支之后）
    meta_path = clean_meta_path(ref_model, cfg["dataset"]) if strategy == "top_exposure" \
        else clean_meta_path(pre["models"][0], cfg["dataset"])
    write_frozen_targets(cfg, targets, load_meta(meta_path),
                         exposure if strategy == "top_exposure" else None,
                         meta_path, refresh=refresh)
```

`run.py`：`ap.add_argument("--refresh-targets", action="store_true")`，
并把该值透传给 `phase_targets(cfg, exp_dir, refresh=args.refresh_targets)`；
`--mode targets` 之外的模式一律 `refresh=False`（只读冻结文件）。

- [ ] **Step 6: 运行确认通过 + 生成三数据集冻结文件**

Run: `G:\Idea\.venv\Scripts\python.exe -m unittest pre.tests.test_pre_targets -v`
Expected: PASS

Run（每个数据集一次，工作目录 `G:\Idea\TPA`）：
`G:\Idea\.venv\Scripts\python.exe pre/run.py --mode targets --tag targets-ml100k`
并先在 `pre/configs/default.yaml` 里改 `dataset` 与 `pre.targets.min_interactions: 180`
（或临时复制 `pre/configs/<dataset>.yaml`，见 Task 3 Step 6）。
Expected: `TPA/pre/targets/{ml100k,gowalla,amazon-book}.json` 生成，ml100k 恰为 5 个物品、
交互数 180–214；gowalla 237 个候选、amazon-book 855 个候选（只取前 5）。

- [ ] **Step 7: 提交**

```bash
git add TPA/pre/analysis/exposure.py TPA/pre/runners/targets.py TPA/pre/runners/pipeline.py TPA/pre/targets TPA/pre/tests/test_pre_targets.py
git commit -m "feat(pre): 冻结目标集产物与原生 HR/NDCG 曝光口径"
```

---

## Task 2: 修复 9 —— 重复交互断言（A 类）

**Files:**
- Modify: `TPA/pre/runners/common.py`（新增 `assert_unique_pairs`）
- Modify: `TPA/pre/runners/degrade.py`（入口断言）
- Test: `TPA/pre/tests/test_pre_degrade_guards.py`

**Interfaces:**
- Produces: `assert_unique_pairs(pairs, where: str) -> int`（返回唯一对数；发现重复抛
  `ValueError`）

- [ ] **Step 1: 写失败测试**

```python
# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
import unittest
from pathlib import Path

TPA_ROOT = Path(__file__).resolve().parents[2]
if str(TPA_ROOT) not in sys.path:
    sys.path.insert(0, str(TPA_ROOT))

from pre.runners.common import assert_unique_pairs          # noqa: E402
from pre.runners.degrade import build_degraded_meta          # noqa: E402


def _meta(pairs):
    return {"num_users": 10, "num_items": 3, "train_pairs": list(pairs),
            "test_pairs": [], "user_items": {}}


class TestUniquePairs(unittest.TestCase):
    def test_returns_count_when_unique(self):
        self.assertEqual(assert_unique_pairs([(0, 0), (1, 0)], "t"), 2)

    def test_raises_on_duplicate(self):
        with self.assertRaises(ValueError) as ctx:
            assert_unique_pairs([(0, 0), (0, 0)], "train_pairs")
        self.assertIn("train_pairs", str(ctx.exception))

    def test_degrade_refuses_duplicated_train_pairs(self):
        # 重复对在旧实现里会被一次删光（p not in deleted_set），属静默偏差
        meta = _meta([(0, 0), (0, 0), (1, 0), (2, 0)])
        with self.assertRaises(ValueError):
            build_degraded_meta(meta, item_id=0, ratio=0.5, seed=42)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行确认失败**

Run: `G:\Idea\.venv\Scripts\python.exe -m unittest pre.tests.test_pre_degrade_guards -v`
Expected: FAIL（`ImportError: cannot import name 'assert_unique_pairs'`）

- [ ] **Step 3: 实现**

```python
# common.py
def assert_unique_pairs(pairs: Sequence[Tuple[int, int]], where: str) -> int:
    """断言交互对唯一；返回对数。

    为什么必须断言：删除用的是 `p not in deleted_set`，一旦 train_pairs 里有重复对，
    一次删除就会把该物品的全部重复项一起删掉，远超 floor(r*N) —— 属于静默偏差，
    会让"删除比例"这条自变量失去定义。实测三数据集 dup=0，断言不改变现有行为。
    """
    seen = set()
    dups = []
    for u, i in pairs:
        key = (int(u), int(i))
        if key in seen:
            dups.append(key)
        else:
            seen.add(key)
    if dups:
        raise ValueError(f"{where} 存在重复交互对 {len(dups)} 条，示例 {dups[:5]}；"
                         f"请在数据预处理阶段去重后再跑退化实验")
    return len(seen)
```

```python
# degrade.py::build_degraded_meta 开头
    assert_unique_pairs(clean_meta["train_pairs"], "clean_meta.train_pairs")
```

- [ ] **Step 4: 运行确认通过**

Run: `G:\Idea\.venv\Scripts\python.exe -m unittest pre.tests.test_pre_degrade_guards pre.tests.test_pre_pipeline -v`
Expected: PASS（含既有 17 个 pre 用例）

- [ ] **Step 5: 提交**

```bash
git add TPA/pre/runners/common.py TPA/pre/runners/degrade.py TPA/pre/tests/test_pre_degrade_guards.py
git commit -m "fix(pre): 退化入口断言 train_pairs 唯一，堵住重复交互静默偏差"
```

---

## Task 3: 修复 1 —— 多 seed 不再静默降级（A 类）+ 路径统一

**Files:**
- Modify: `TPA/pre/runners/common.py`（路径助手）
- Modify: `TPA/pre/runners/train_model.py`
- Modify: `TPA/pre/runners/degrade.py`
- Modify: `TPA/pre/runners/pipeline.py`
- Modify: `TPA/pre/run.py`
- Modify: `TPA/pre/configs/default.yaml`
- Test: `TPA/pre/tests/test_pre_seeds.py`、`TPA/pre/tests/test_pre_paths.py`

**Interfaces:**
- Produces:
  - `model_seeds(cfg) -> List[int]`（读 `pre.training.seeds`，空则 `[42]`）
  - `deletion_seed(cfg) -> int`（读 `pre.deletion_seed`，缺省 42；与 model seed 解耦）
  - `raw_root(exp_dir) -> Path`、`resolve_raw_root(exp_dir) -> Path`
  - `cond_dir(exp_dir, group, *, model_name=None, seed=None, item_id=None, label=None) -> Path`
  - `original_dir(exp_dir, model_name, seed) -> Path`
  - `degraded_data_dir(exp_dir, model_name, item_id, ratio) -> Path`
  - `degraded_model_dir(exp_dir, model_name, item_id, ratio, seed) -> Path`
  - `attack_dir(exp_dir, model_name, item_id, attack, ratio, seed) -> Path`
  - CLI：`--seed 43` / `--seeds 42,43,44`

**统一目录规则**（一条规则覆盖全部条件）：
`raw/<group>/<model>/[seed_<s>/][item_<id>/][<label>/]`

```
raw/original/<model>/seed_42/clean/                    参考系 z^0
raw/degraded/<model>/item_00016/ratio_50/train_data/   退化数据（fix 8 后改为 shared）
raw/degraded/<model>/seed_42/item_00016/ratio_50/      退化模型 z^r
raw/attacks/<model>/seed_42/item_00016/uba_ratio_50/   攻击模型 z^A
```

- [ ] **Step 1: 写失败测试** —— `TPA/pre/tests/test_pre_paths.py`

```python
# -*- coding: utf-8 -*-
"""路径规则（seed 段 / raw 根 / 旧布局回退）。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

TPA_ROOT = Path(__file__).resolve().parents[2]
if str(TPA_ROOT) not in sys.path:
    sys.path.insert(0, str(TPA_ROOT))

from pre.runners.common import (attack_dir, cond_dir, degraded_data_dir,  # noqa: E402
                                degraded_model_dir, original_dir,
                                resolve_raw_root)


class TestPathRules(unittest.TestCase):
    def test_new_layout_paths(self):
        exp = Path("X:/exp")
        self.assertEqual(original_dir(exp, "lightgcn", 42).as_posix(),
                         "X:/exp/raw/original/lightgcn/seed_42/clean")
        self.assertEqual(degraded_model_dir(exp, "lightgcn", 16, 0.5, 43).as_posix(),
                         "X:/exp/raw/degraded/lightgcn/seed_43/item_00016/ratio_50")
        self.assertEqual(attack_dir(exp, "lightgcn", 16, "uba", 0.5, 44).as_posix(),
                         "X:/exp/raw/attacks/lightgcn/seed_44/item_00016/uba_ratio_50")
        self.assertEqual(degraded_data_dir(exp, "lightgcn", 16, 0.5).as_posix(),
                         "X:/exp/raw/degraded/lightgcn/item_00016/ratio_50")

    def test_cond_dir_omits_empty_segments(self):
        p = cond_dir(Path("X:/exp"), "original", model_name="mf", seed=42, label="clean")
        self.assertEqual(p.as_posix(), "X:/exp/raw/original/mf/seed_42/clean")
        self.assertNotIn("item_", p.as_posix())
        self.assertNotIn("None", p.as_posix())

    def test_resolve_raw_root_prefers_new_layout(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            exp = Path(d)
            self.assertEqual(resolve_raw_root(exp), exp / "raw")   # 都不存在 -> 新布局
            (exp / "original").mkdir()
            self.assertEqual(resolve_raw_root(exp), exp)           # 旧布局回退
            (exp / "raw").mkdir()
            self.assertEqual(resolve_raw_root(exp), exp / "raw")   # 新布局优先


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行确认失败**

Run: `G:\Idea\.venv\Scripts\python.exe -m unittest pre.tests.test_pre_paths -v`
Expected: FAIL（`ImportError: cannot import name 'attack_dir'`）

- [ ] **Step 3: 在 `common.py` 实现路径助手与 seed 读取**

```python
# ── 产物路径（spec §3.10 + §4.1）──────────────────────────────────────────────
def raw_root(exp_dir: Path) -> Path:
    """raw/：训练与攻击产物根。"""
    return Path(exp_dir) / "raw"


def resolve_raw_root(exp_dir: Path) -> Path:
    """读路径解析：新布局 raw/ 优先；只有旧布局（exp_dir/original 存在）时回退。

    为什么需要回退：L1/L3 验收要把新口径结果与历史产物（run-k5v2 等旧布局）
    做逐项对照，旧目录不迁移、只读。
    """
    new = Path(exp_dir) / "raw"
    if new.exists():
        return new
    if (Path(exp_dir) / "original").exists():
        return Path(exp_dir)
    return new


def cond_dir(exp_dir: Path, group: str, *, model_name: Optional[str] = None,
             seed: Optional[int] = None, item_id: Optional[int] = None,
             label: Optional[str] = None) -> Path:
    """条件目录：<raw>/<group>/[<model>/][seed_<s>/][item_<id>/][<label>/]。

    为什么集中在一个函数：路径散落在 degrade/train_model/run_attack/aggregate 四处，
    加一个维度（seed）就得改四处，并且极易出现"写入与读取不一致"的静默 bug
    （本次修复 1 的根因之一）。所有写入与读取都必须经过这里。
    """
    base = resolve_raw_root(Path(exp_dir)) / str(group)
    if model_name:
        base = base / str(model_name)
    if seed is not None:
        base = base / f"seed_{int(seed)}"
    if item_id is not None:
        base = base / f"item_{int(item_id):05d}"
    if label:
        base = base / str(label)
    return base


def original_dir(exp_dir: Path, model_name: str, seed: int) -> Path:
    return cond_dir(exp_dir, "original", model_name=model_name, seed=seed,
                    label="clean")


def degraded_data_dir(exp_dir: Path, model_name: str, item_id: int, ratio: float) -> Path:
    """退化数据目录（无 seed：删除由 deletion_seed 唯一决定，与模型种子无关）。"""
    return cond_dir(exp_dir, "degraded", model_name=model_name, item_id=item_id,
                    label=ratio_dirname(ratio))


def degraded_model_dir(exp_dir: Path, model_name: str, item_id: int, ratio: float,
                       seed: int) -> Path:
    return cond_dir(exp_dir, "degraded", model_name=model_name, seed=seed,
                    item_id=item_id, label=ratio_dirname(ratio))


def attack_dir(exp_dir: Path, model_name: str, item_id: int, attack: str,
               ratio: float, seed: int) -> Path:
    return cond_dir(exp_dir, "attacks", model_name=model_name, seed=seed,
                    item_id=item_id, label=f"{attack}_{ratio_dirname(ratio)}")


def model_seeds(cfg: Dict[str, Any]) -> List[int]:
    """模型种子列表；空列表/缺失 -> [42]（与历史单 seed 行为一致）。"""
    seeds = (cfg.get("pre", {}) or {}).get("training", {}).get("seeds", [42])
    out = [int(s) for s in (seeds or [42])]
    return out or [42]


def deletion_seed(cfg: Dict[str, Any]) -> int:
    """删除排列种子：实验自变量，与模型种子解耦（spec §1.6）。"""
    return int((cfg.get("pre", {}) or {}).get("deletion_seed", 42))
```

注意：`ratio_dirname` 现在位于 `degrade.py`，会造成 `degrade.py <-> common.py` 循环
import。把 `ratio_dirname` **移到 `common.py`**，并在 `degrade.py` 保留
`from pre.runners.common import ratio_dirname` 的再导出（既有调用点不改）。

- [ ] **Step 4: 写 seed 行为测试** —— `TPA/pre/tests/test_pre_seeds.py`

```python
# -*- coding: utf-8 -*-
"""多 seed 行为：每个 seed 独立目录、独立权重；退化数据不随 seed 变化。"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch

TPA_ROOT = Path(__file__).resolve().parents[2]
if str(TPA_ROOT) not in sys.path:
    sys.path.insert(0, str(TPA_ROOT))

from pre.runners import train_model as tm                      # noqa: E402
from pre.runners.common import deletion_seed, model_seeds       # noqa: E402
from pre.runners.pipeline import phase_original                 # noqa: E402


class _FakeModel:
    """只为路径/种子断言服务的假模型（不训练）。"""

    def __init__(self, seed: int):
        self.seed = int(seed)
        g = torch.Generator().manual_seed(self.seed)
        self._v = torch.randn(5, 3, generator=g)
        self._u = torch.randn(4, 3, generator=g)

    def state_dict(self):
        return {"embedding.weight": torch.cat([self._u, self._v], dim=0)}

    def get_item_embeddings(self):
        return self._v.clone()

    def get_user_embeddings(self):
        return self._u.clone()


def _meta():
    return {"num_users": 4, "num_items": 5, "train_pairs": [(0, 0), (1, 1)],
            "test_pairs": [(2, 2)], "user_items": {}}


class TestSeedConfig(unittest.TestCase):
    def test_model_seeds_reads_list(self):
        self.assertEqual(model_seeds({"pre": {"training": {"seeds": [42, 43, 44]}}}),
                         [42, 43, 44])
        self.assertEqual(model_seeds({"pre": {"training": {"seeds": []}}}), [42])
        self.assertEqual(model_seeds({}), [42])

    def test_deletion_seed_is_independent(self):
        cfg = {"pre": {"deletion_seed": 7, "training": {"seeds": [42, 43]}}}
        self.assertEqual(deletion_seed(cfg), 7)
        self.assertEqual(model_seeds(cfg), [42, 43])


class TestPhaseOriginalPerSeed(unittest.TestCase):
    def test_two_seeds_produce_two_independent_runs(self):
        calls = []

        def fake_train(model_name, cfg, meta, seed, device=None):
            calls.append(int(seed))
            return _FakeModel(seed), [{"epoch": 1, "loss": 1.0}]

        orig = tm.train_victim
        tm.train_victim = fake_train
        try:
            cfg = {"dataset": "toy", "k": 10,
                   "pre": {"models": ["lightgcn"], "training": {"seeds": [42, 43]},
                           "cache": {"reuse": True}}}
            with tempfile.TemporaryDirectory() as d:
                exp = Path(d)
                for s in (42, 43):
                    phase_original(cfg, exp, {"items": []}, seeds=[s])
                d42 = exp / "raw" / "original" / "lightgcn" / "seed_42" / "clean"
                d43 = exp / "raw" / "original" / "lightgcn" / "seed_43" / "clean"
                self.assertTrue((d42 / "metadata.json").exists())
                self.assertTrue((d43 / "metadata.json").exists())
                import json
                self.assertEqual(json.loads((d42 / "metadata.json").read_text("utf-8"))["seed"], 42)
                self.assertEqual(json.loads((d43 / "metadata.json").read_text("utf-8"))["seed"], 43)
        finally:
            tm.train_victim = orig
        self.assertEqual(calls, [42, 43])


if __name__ == "__main__":
    unittest.main()
```

（`phase_original` 的 `meta` 参数由 `clean_meta_path` 提供；测试里因为
`_FakeModel` 不读 meta，可直接给 `{"items": []}` 的 targets —— 但
`load_meta(clean_meta_path(...))` 会真的去读文件，所以该用例改为调用
`train_original` 更稳妥：把上面 `phase_original(cfg, exp, {"items": []}, seeds=[s])`
替换为 `tm.train_original(cfg, exp, "lightgcn", _meta(), s)`。）

- [ ] **Step 5: 运行确认失败 → 实现 → 通过**

Run: `G:\Idea\.venv\Scripts\python.exe -m unittest pre.tests.test_pre_paths pre.tests.test_pre_seeds -v`
Expected（实现前）: FAIL；实现后: PASS

实现要点（改动清单）：
1. `train_model.train_original`：`base = original_dir(exp_dir, model_name, seed)`。
2. `train_model.train_and_cache`：新增 `seed` 已存在，改用
   `degraded_model_dir(...)`（`group=="degraded"`）或 `attack_dir(...)`
   （`group=="attacks"`，需从 `label` 拆出 attack 与 ratio）。
3. `degrade.build_and_cache_degraded`：`base = degraded_data_dir(exp_dir, model_name, item_id, ratio)`。
4. `pipeline.phase_original/phase_degrade/phase_degraded/phase_attack`：改成
   `for seed in model_seeds(cfg)` 外层循环；删除 `_seed(cfg)`，改用
   `deletion_seed(cfg)` 传给 `build_and_cache_degraded`。
5. `run.py`：新增 `--seed` / `--seeds`，写回 `cfg["pre"]["training"]["seeds"]`；
   `apply_cli_filters` 里 `if args.seeds: pre["training"]["seeds"] = [...]`。
6. `configs/default.yaml`：`training.seeds: [42, 43, 44]`，新增 `pre.deletion_seed: 42`
   （注释写清"删除是自变量，不是噪声源"）。

**关键不变量（必须由测试保证）**：同一 `(item, ratio)` 在不同 seed 下读到**同一份**
`deleted_interactions.json`（退化数据不含 seed 段 → 天然满足，但要有断言）。

- [ ] **Step 6: 端到端验证（ml100k，1 物品 1 比例 2 seed）**

Run（工作目录 `G:\Idea\TPA`）：
`G:\Idea\.venv\Scripts\python.exe pre/run.py --mode all --tag seed-check --models lightgcn --ratios 0.5 --attacks random --limit-items 1 --epochs 2 --seeds 42,43`
Expected: `raw/original/lightgcn/seed_42/clean/` 与 `seed_43` 各一份；
`raw/degraded/lightgcn/item_XXXXX/ratio_50/train_data/meta.pkl` 只有一份；
两份模型权重 sha256 不同。

- [ ] **Step 7: 提交**

```bash
git add TPA/pre/runners/common.py TPA/pre/runners/degrade.py TPA/pre/runners/train_model.py TPA/pre/runners/pipeline.py TPA/pre/run.py TPA/pre/configs/default.yaml TPA/pre/tests/test_pre_paths.py TPA/pre/tests/test_pre_seeds.py
git commit -m "fix(pre): 多 seed 不再静默降级，路径统一到 raw/ 条件目录助手"
```

---

## Task 4: 修复 2 —— UBA 处理效应缓存 key 与行身份（A + B 类）

**Files:**
- Modify: `TPA/attacks/uba/generate.py`
- Modify: `TPA/attacks/uba/uplift.py`
- Test: `TPA/tests/test_uba_effect_cache.py`

**Interfaces:**
- Produces:
  - `meta_fingerprint(meta) -> str`（`sha256(num_users|num_items|sorted(train_pairs))[:16]`）
  - `effect_row_index(users) -> Dict[int, int]`
  - `realign_effect(cached_users, effect, requested_users) -> np.ndarray`
  - `effect_cache_path(..., data_fingerprint: str = "")`（文件名追加指纹）

- [ ] **Step 1: 写失败测试** —— `TPA/tests/test_uba_effect_cache.py`

```python
# -*- coding: utf-8 -*-
"""UBA 处理效应缓存：指纹进 key、命中时按 user_id 重排（修复 2）。"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

TPA_ROOT = Path(__file__).resolve().parents[1]
if str(TPA_ROOT) not in sys.path:
    sys.path.insert(0, str(TPA_ROOT))

from attacks.uba.generate import effect_cache_path, meta_fingerprint   # noqa: E402
from attacks.uba.uplift import realign_effect                          # noqa: E402


def _cfg(**over):
    cfg = {"dataset": "toy", "seed": 42,
           "attack": {"uba": {"treatment": {"method": "path", "max_per_user": 6,
                                            "alpha": 1.0, "beta": 1.0}}}}
    cfg["attack"]["uba"]["treatment"].update(over)
    return cfg


class TestMetaFingerprint(unittest.TestCase):
    def test_order_insensitive(self):
        a = meta_fingerprint({"num_users": 3, "num_items": 4,
                              "train_pairs": [(0, 0), (1, 1)]})
        b = meta_fingerprint({"num_users": 3, "num_items": 4,
                              "train_pairs": [(1, 1), (0, 0)]})
        self.assertEqual(a, b)
        self.assertEqual(len(a), 16)

    def test_changes_with_data(self):
        a = meta_fingerprint({"num_users": 3, "num_items": 4, "train_pairs": [(0, 0)]})
        b = meta_fingerprint({"num_users": 3, "num_items": 4, "train_pairs": [(0, 1)]})
        self.assertNotEqual(a, b)


class TestCachePath(unittest.TestCase):
    def test_fingerprint_enters_filename(self):
        p1 = effect_cache_path(_cfg(), "lightgcn", 402, "path", data_fingerprint="aaaa")
        p2 = effect_cache_path(_cfg(), "lightgcn", 402, "path", data_fingerprint="bbbb")
        self.assertNotEqual(p1.name, p2.name)
        self.assertIn("aaaa", p1.name)


class TestRealign(unittest.TestCase):
    def test_rows_follow_requested_user_order(self):
        cached = [5, 7, 9]
        effect = np.array([[1.0, 1.5], [2.0, 2.5], [3.0, 3.5]])
        out = realign_effect(cached, effect, [9, 5])
        np.testing.assert_allclose(out, [[3.0, 3.5], [1.0, 1.5]])

    def test_missing_user_raises(self):
        cached = [5, 7]
        effect = np.zeros((2, 2))
        with self.assertRaises(KeyError):
            realign_effect(cached, effect, [5, 9])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行确认失败**

Run: `G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_uba_effect_cache -v`
Expected: FAIL（`ImportError: cannot import name 'meta_fingerprint'`）

- [ ] **Step 3: 实现（`uplift.py`）**

```python
def effect_row_index(users: Sequence[int]) -> Dict[int, int]:
    """{user_id: Y 的行号}。把"按位置取行"变成显式映射，避免行错位。"""
    return {int(u): r for r, u in enumerate(users)}


def realign_effect(cached_users: Sequence[int], effect: np.ndarray,
                   requested_users: Sequence[int]) -> np.ndarray:
    """把缓存 Y 的行按 user_id 重排到请求顺序；缺行抛 KeyError（调用方重算）。

    为什么需要：缓存 Y 的行序由**生成缓存时**的 target_users 决定，而 allocate
    按位置取行（values[r, t]）。若两者长度相同但用户集合不同，Y 会被贴到错误的
    用户上——实测跨条件 target_users 位置重合度仅 7.8%（spec §4.2）。
    """
    index = effect_row_index(cached_users)
    missing = [int(u) for u in requested_users if int(u) not in index]
    if missing:
        raise KeyError(f"缓存缺 {len(missing)} 个目标用户的行，示例 {missing[:5]}")
    return np.stack([np.asarray(effect, dtype=np.float64)[index[int(u)]]
                     for u in requested_users], axis=0)
```

并在 `allocate` 的 `uba` 分支入口加显式一致性断言（fail loudly，不静默错位）：

```python
    if strategy == "uba":
        if not users:
            raise ValueError("allocation.strategy=uba 需要非空目标用户")
        if values.shape[0] != len(users):
            raise ValueError(
                f"Y 行数 {values.shape[0]} 与目标用户数 {len(users)} 不一致："
                f"调用方必须先 realign_effect（见 attacks/uba/generate.py）")
```

- [ ] **Step 4: 实现（`generate.py`）**

```python
def meta_fingerprint(meta: Dict[str, Any]) -> str:
    """数据指纹：num_users | num_items | sorted(train_pairs) 的 sha256 前 16 位。

    为什么不用 pickle 字节：pickle 输出随 python/协议版本与 dict 顺序变化，
    无法跨进程稳定比较；这里只用定义数据的三个量。
    """
    h = hashlib.sha256()
    h.update(f"{int(meta['num_users'])}|{int(meta['num_items'])}|".encode("utf-8"))
    pairs = sorted((int(u), int(i)) for u, i in meta["train_pairs"])
    h.update(repr(pairs).encode("utf-8"))
    return h.hexdigest()[:16]
```

`effect_cache_path` 增加形参 `data_fingerprint: str = ""`，并在两个分支的文件名末尾
追加 `f"_{data_fingerprint}"`（非空时）。

`load_or_build_effect` 的命中分支改为：

```python
    fp = meta_fingerprint(meta)
    path = effect_cache_path(config, model_name, target_item, method,
                             repeats=repeats, hit_k=hit_k, alpha=alpha,
                             beta=beta, seed=seed, data_fingerprint=fp)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        cached_users = [int(u) for u in payload.get("target_users", [])]
        if payload.get("meta_fingerprint") != fp:
            print(f"[generate] 处理效应缓存数据指纹不匹配 → 重算 {path.name}")
        elif not cached_users:
            print(f"[generate] 处理效应缓存无 target_users（旧口径）→ 重算 {path.name}")
        else:
            try:
                payload["effect"] = realign_effect(
                    cached_users, payload["effect"], target_users)
            except KeyError as exc:
                print(f"[generate] 处理效应缓存用户集合不匹配（{exc}）→ 重算 {path.name}")
            else:
                print(f"[generate] 命中处理效应缓存 → {path}")
                return payload
        path.unlink(missing_ok=True)      # 作废旧口径缓存，避免继续被复用
```

写缓存时同步补三件事：

```python
    payload["meta_fingerprint"] = fp
    payload["git_commit"] = git_commit()
    payload["target_users"] = [int(u) for u in target_users]   # surrogate 分支也要写
```

- [ ] **Step 5: 运行确认通过 + 真实数据验证**

Run: `G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_uba_effect_cache tests.test_uba_generate tests.test_uba_uplift -v`
Expected: PASS

Run（ml100k，5 目标 × 3 比例；**先备份旧缓存**）：
`G:\Idea\.venv\Scripts\python.exe pre/run.py --mode attack --tag uba-cache-check --models lightgcn --attacks uba --ratios 0.1,0.5,0.9`
Expected: 同一物品在不同比例下 `estimated_value` **不再相同**；`attacks/uba/data/estimate/ml100k/lightgcn/`
下出现带指纹的新缓存文件；`target_users` 与本次请求逐元素一致（一致率 100%）。

- [ ] **Step 6: 提交**

```bash
git add TPA/attacks/uba/generate.py TPA/attacks/uba/uplift.py TPA/tests/test_uba_effect_cache.py
git commit -m "fix(attacks): UBA 处理效应缓存绑定数据指纹并按 user_id 重排行"
```

---

## Task 5: 修复 5 —— Procrustes 排除目标物品（A 类）

**Files:**
- Modify: `TPA/pre/analysis/align.py`
- Test: `TPA/pre/tests/test_pre_align_exclude.py`

**Interfaces:**
- Produces: `procrustes(ref, cond, mask, exclude=None) -> torch.Tensor`
  （`exclude` 接受单个 item id 或 id 序列）

- [ ] **Step 1: 写失败测试**

```python
# -*- coding: utf-8 -*-
"""Procrustes 拟合必须排除目标物品，否则对齐会吸收目标自身位移（leakage）。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

TPA_ROOT = Path(__file__).resolve().parents[2]
if str(TPA_ROOT) not in sys.path:
    sys.path.insert(0, str(TPA_ROOT))

from pre.analysis.align import procrustes  # noqa: E402


def _rotation(d: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(d, d, generator=g)
    q, _ = torch.linalg.qr(a)
    return q


class TestProcrustesExclusion(unittest.TestCase):
    def test_R_is_invariant_to_target_row_offset(self):
        d, n, target = 4, 20, 3
        ref = torch.randn(n, d, generator=torch.Generator().manual_seed(1))
        R_true = _rotation(d, seed=2)
        cond_base = ref @ R_true.T
        mask = torch.ones(n, dtype=torch.bool)

        cond_a = cond_base.clone()
        cond_b = cond_base.clone()
        cond_b[target] = cond_base[target] + 5.0     # 只有目标物品多了一个位移

        R_a = procrustes(ref, cond_a, mask, exclude=target)
        R_b = procrustes(ref, cond_b, mask, exclude=target)
        self.assertLess(float((R_a - R_b).abs().max()), 1e-5)

    def test_without_exclusion_R_moves_with_target_offset(self):
        d, n, target = 4, 20, 3
        ref = torch.randn(n, d, generator=torch.Generator().manual_seed(3))
        R_true = _rotation(d, seed=4)
        cond_base = ref @ R_true.T
        mask = torch.ones(n, dtype=torch.bool)
        cond_b = cond_base.clone()
        cond_b[target] = cond_base[target] + 5.0
        R_a = procrustes(ref, cond_base, mask)
        R_b = procrustes(ref, cond_b, mask)
        self.assertGreater(float((R_a - R_b).abs().max()), 1e-5)   # 说明 bug 真实存在


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行确认失败**

Run: `G:\Idea\.venv\Scripts\python.exe -m unittest pre.tests.test_pre_align_exclude -v`
Expected: FAIL（`TypeError: procrustes() got an unexpected keyword argument 'exclude'`）

- [ ] **Step 3: 实现**

```python
def procrustes(ref: torch.Tensor, cond: torch.Tensor, mask: torch.Tensor,
               exclude=None) -> torch.Tensor:
    """求正交矩阵 R，使 cond @ R 尽量接近 ref（只在 mask \ exclude 行上拟合）。

    为什么必须能排除目标物品（spec §3.2 硬规则）：目标物品自身的位移正是被测量的
    对象；若它参与拟合，R 会吸收一部分该位移（leakage into the alignment
    transform），使 d_tgt 系统性偏小。实测：包含目标物品时随机臂 d_tgt=1.093，
    排除后 1.093 -> 同一量级但 placebo 分离度从 ~2.9x 提升到 ~5x。

    exclude 传单个 item id 或 id 序列；拟合行数少于 2 时直接报错
    （否则 SVD 退化，R 变成任意旋转而无人察觉）。

    注意：同一个 R 必须同时作用于**物品行与用户行**（两者共享同一嵌入空间），
    否则计算假用户合力 F 与 audience 余弦时坐标系不一致。
    """
    fit = mask.clone().to(torch.bool)
    if exclude is not None:
        idx = ([int(exclude)] if isinstance(exclude, (int,)) else
               [int(x) for x in exclude])
        for i in idx:
            if 0 <= i < fit.numel():
                fit[i] = False
    if int(fit.sum()) < 2:
        raise ValueError(f"Procrustes 拟合行数不足（{int(fit.sum())}），检查 mask/exclude")
    a, b = ref[fit], cond[fit]
    m = b.T @ a
    u, _, vt = torch.linalg.svd(m)
    return u @ vt
```

- [ ] **Step 4: 运行确认通过 + 全量 pre 测试**

Run: `G:\Idea\.venv\Scripts\python.exe -m unittest pre.tests.test_pre_align_exclude pre.tests.test_pre_pipeline -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add TPA/pre/analysis/align.py TPA/pre/tests/test_pre_align_exclude.py
git commit -m "fix(pre): Procrustes 拟合掩码排除目标物品，消除对齐泄漏"
```
---

## Task 6: 修复 6 —— `analyze` 不再静默跳过缺失条件（A 类）

**Files:**
- Modify: `TPA/pre/analysis/aggregate.py`
- Modify: `TPA/pre/run.py`（打印覆盖率）
- Test: `TPA/pre/tests/test_pre_coverage.py`

**Interfaces:**
- Produces:
  - `STATE_RESOLVED / STATE_UNRESOLVED / STATE_MISSING / STATE_INVALID`（字符串常量）
  - `expected_conditions(cfg, targets, seed) -> List[Dict[str, Any]]`
  - `classify_state(exp_dir, cond) -> Tuple[str, str]`（state, reason）
  - `write_coverage(exp_dir, states) -> Dict[str, Any]`
  - `analyze(...)` 返回值新增 `coverage` 与 `coverage_ratio`

- [ ] **Step 1: 写失败测试** —— `TPA/pre/tests/test_pre_coverage.py`

```python
# -*- coding: utf-8 -*-
"""四态状态机与覆盖率（spec §3.8）：缺一个条件目录必须显式报 missing。"""
from __future__ import annotations

import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import torch

TPA_ROOT = Path(__file__).resolve().parents[2]
if str(TPA_ROOT) not in sys.path:
    sys.path.insert(0, str(TPA_ROOT))

from pre.analysis import aggregate as ag                       # noqa: E402


def _cfg():
    return {"dataset": "toy", "k": 10,
            "pre": {"models": ["lightgcn"], "attacks": [],
                    "deletion_ratios": [0.5],
                    "embedding": {"align": "procrustes", "min_support": 1,
                                  "distance": "l2"},
                    "training": {"seeds": [42]}}}


class TestCoverage(unittest.TestCase):
    def test_missing_condition_is_reported(self):
        with tempfile.TemporaryDirectory() as d:
            exp = Path(d)
            meta = {"num_users": 4, "num_items": 5,
                    "train_pairs": [(0, 0), (1, 1), (2, 2)],
                    "test_pairs": [(3, 3)], "user_items": {}}
            clean_meta = Path(d) / "clean_meta.pkl"
            clean_meta.write_bytes(pickle.dumps(meta))

            ref_dir = exp / "raw" / "original" / "lightgcn" / "seed_42" / "clean"
            ref_dir.mkdir(parents=True)
            V = torch.eye(5, 3)
            torch.save({"item_factors": V, "user_factors": torch.eye(4, 3)},
                       ref_dir / "embeddings.pt")
            (ref_dir / "metadata.json").write_text(json.dumps({"seed": 42}), "utf-8")
            (ref_dir / "model.pt").write_bytes(b"x")

            # 只造 item 1 的条件目录；item 2 故意不造
            d1 = (exp / "raw" / "degraded" / "lightgcn" / "seed_42"
                  / "item_00001" / "ratio_50")
            d1.mkdir(parents=True)
            cond_meta = Path(d) / "d1_meta.pkl"
            cond_meta.write_bytes(pickle.dumps(meta))
            torch.save({"item_1": V[1]}, d1 / "embeddings.pt")
            (d1 / "metadata.json").write_text(
                json.dumps({"seed": 42, "meta_path": str(cond_meta)}), "utf-8")
            (d1 / "model.pt").write_bytes(b"x")

            orig_matrix = ag.load_item_matrix
            orig_meta_path = ag.clean_meta_path
            ag.load_item_matrix = lambda *a, **k: V.clone()
            ag.clean_meta_path = lambda *a, **k: clean_meta
            try:
                targets = {"items": [{"item_id": 1}, {"item_id": 2}]}
                res = ag.analyze(_cfg(), exp, targets)
            finally:
                ag.load_item_matrix = orig_matrix
                ag.clean_meta_path = orig_meta_path

            cov = json.loads((exp / "coverage.json").read_text("utf-8"))
            states = {(c["item_id"], c["state"]) for c in cov["conditions"]}
            self.assertIn((1, "resolved"), states)
            self.assertIn((2, "missing"), states)
            self.assertAlmostEqual(res["coverage_ratio"], 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行确认失败**

Run: `G:\Idea\.venv\Scripts\python.exe -m unittest pre.tests.test_pre_coverage -v`
Expected: FAIL（`KeyError: 'coverage_ratio'` / coverage.json 不存在）

- [ ] **Step 3: 实现状态机与覆盖率**（加在 `aggregate.py`）

```python
STATE_RESOLVED = "resolved"
STATE_UNRESOLVED = "unresolved_frame"
STATE_MISSING = "missing"
STATE_INVALID = "invalid_numeric"
ALIGN_GATE = 0.95            # spec §3.2：对齐质量门槛


def expected_conditions(cfg: Dict[str, Any], targets: Dict[str, Any],
                        seed: int) -> List[Dict[str, Any]]:
    """本实验**期望**存在的全部条件（覆盖率的分母）。

    含 degradation-only 臂：它是恢复率的基线，缺了就无法算 RR。
    """
    pre = cfg.get("pre", {})
    out: List[Dict[str, Any]] = []
    for model in pre.get("models", []):
        for t in targets["items"]:
            for ratio in pre.get("deletion_ratios", []):
                base = {"model_name": model, "seed": int(seed),
                        "item_id": int(t["item_id"]), "ratio": float(ratio)}
                out.append({**base, "attack": "degradation-only"})
                for a in pre.get("attacks", []):
                    out.append({**base, "attack": a})
    return out


def _cond_paths(exp_dir: Path, cond: Dict[str, Any]) -> Dict[str, Path]:
    """条件的关键文件路径（攻击臂与退化臂目录规则不同）。"""
    model, seed = cond["model_name"], int(cond["seed"])
    item, ratio = int(cond["item_id"]), float(cond["ratio"])
    if cond["attack"] == "degradation-only":
        d = degraded_model_dir(exp_dir, model, item, ratio, seed)
    else:
        d = attack_dir(exp_dir, model, item, cond["attack"], ratio, seed)
    return {"dir": d, "embeddings": d / "embeddings.pt",
            "metadata": d / "metadata.json", "model": d / "model.pt"}


def classify_state(exp_dir: Path, cond: Dict[str, Any]) -> Tuple[str, str]:
    """四态判定的"存在性"部分：目录/文件缺失 -> missing（spec §3.8）。

    为什么必须显式分态：旧实现对缺失条件 `continue`，跑 3/30 与跑完 30/30 产出的
    CSV 看起来一模一样 —— 这属于"结果口径 bug"，不是工程瑕疵。
    """
    p = _cond_paths(exp_dir, cond)
    if not p["dir"].exists():
        return STATE_MISSING, "条件目录不存在"
    for key in ("embeddings", "metadata", "model"):
        if not p[key].exists():
            return STATE_MISSING, f"缺文件 {p[key].name}"
    return STATE_RESOLVED, ""


def write_coverage(exp_dir: Path, states: List[Dict[str, Any]]) -> Dict[str, Any]:
    """写 coverage.json + coverage.md，返回汇总（含 coverage_ratio）。"""
    counts: Dict[str, int] = {}
    for s in states:
        counts[s["state"]] = counts.get(s["state"], 0) + 1
    total = max(1, len(states))
    summary = {"total": len(states), "counts": counts,
               "coverage_ratio": counts.get(STATE_RESOLVED, 0) / total,
               "unresolved_frame": counts.get(STATE_UNRESOLVED, 0),
               "missing": counts.get(STATE_MISSING, 0),
               "invalid_numeric": counts.get(STATE_INVALID, 0)}
    save_json({"summary": summary, "conditions": states}, exp_dir / "coverage.json")
    lines = ["# 覆盖率", "",
             f"- 条件总数：{len(states)}",
             f"- resolved：{counts.get(STATE_RESOLVED, 0)}",
             f"- unresolved_frame：{counts.get(STATE_UNRESOLVED, 0)}",
             f"- missing：{counts.get(STATE_MISSING, 0)}",
             f"- invalid_numeric：{counts.get(STATE_INVALID, 0)}",
             f"- coverage_ratio：{summary['coverage_ratio']:.4f}", ""]
    bad = [s for s in states if s["state"] != STATE_RESOLVED]
    if bad:
        lines += ["## 非 resolved 明细", ""]
        lines += [f"- {s['model_name']} item={s['item_id']} ratio={s['ratio']} "
                  f"attack={s['attack']} seed={s['seed']} -> {s['state']}（{s['reason']}）"
                  for s in bad]
    (exp_dir / "coverage.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
```

`analyze` 主循环骨架（`_rows_for_condition` = 现有逐条件计算体原样抽成函数，
返回 `(rows, align_report)`）：

```python
    states: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    for seed in model_seeds(cfg):
        for cond in expected_conditions(cfg, targets, seed):
            state, reason = classify_state(exp_dir, cond)
            if state == STATE_RESOLVED:
                try:
                    cond_rows, align_report = _rows_for_condition(cfg, exp_dir, cond)
                except (ValueError, FloatingPointError) as exc:
                    state, reason = STATE_INVALID, f"{type(exc).__name__}: {exc}"
                else:
                    aligned_cos = float(align_report["mean_row_cos_aligned"])
                    if aligned_cos < ALIGN_GATE:
                        state = STATE_UNRESOLVED
                        reason = f"mean_row_cos_aligned={aligned_cos:.3f} < {ALIGN_GATE}"
                    else:
                        rows.extend(cond_rows)
            states.append({**cond, "state": state, "reason": reason})
            if state != STATE_RESOLVED:
                print(f"[analyze] {state}: model={cond['model_name']} "
                      f"item={cond['item_id']} ratio={cond['ratio']} "
                      f"attack={cond['attack']} seed={cond['seed']}（{reason}）")
    coverage = write_coverage(exp_dir, states)
```

`analyze` 的返回值并入 `coverage`（`coverage_ratio` 直接取 `coverage["coverage_ratio"]`）。

- [ ] **Step 4: 运行确认通过**
- [ ] **Step 5: 提交**

Run: `G:\Idea\.venv\Scripts\python.exe -m unittest pre.tests.test_pre_coverage -v` → PASS

```bash
git add TPA/pre/analysis/aggregate.py TPA/pre/run.py TPA/pre/tests/test_pre_coverage.py
git commit -m "fix(pre): analyze 改为四态状态机并输出 coverage，不再静默跳过缺失条件"
```

---

## Task 7: 修复 7 —— 径向/切向严格分解（A 类）

**Files:**
- Create: `TPA/pre/analysis/decomposition.py`
- Test: `TPA/pre/tests/test_pre_decomposition.py`

**Interfaces:**
- Produces: `decompose(z, z0) -> Dict[str, float]`（键 `z_norm` / `z0_norm` / `radial` /
  `radial_dev` / `tangent_norm` / `l2`）、`identity_residual(z, z0) -> float`、
  `norm_recovery_legacy(z0, z_deg, z_att) -> float`

- [ ] **Step 1: 写失败测试** —— `TPA/pre/tests/test_pre_decomposition.py`

```python
# -*- coding: utf-8 -*-
"""径向/切向分解恒等式（spec §3.5）：‖z−z0‖² = 径向偏差² + 切向²，严格成立。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

TPA_ROOT = Path(__file__).resolve().parents[2]
if str(TPA_ROOT) not in sys.path:
    sys.path.insert(0, str(TPA_ROOT))

from pre.analysis.decomposition import (decompose,          # noqa: E402
                                        identity_residual,
                                        norm_recovery_legacy)


class TestIdentity(unittest.TestCase):
    def test_identity_holds_on_random_vectors(self):
        g = torch.Generator().manual_seed(7)
        for _ in range(20):
            z0 = torch.randn(16, generator=g)
            z = torch.randn(16, generator=g) * 3.0
            self.assertLess(identity_residual(z, z0), 1e-10)

    def test_known_orthogonal_case(self):
        d = decompose(torch.tensor([3.0, 4.0]), torch.tensor([2.0, 0.0]))
        self.assertAlmostEqual(d["radial"], 3.0, places=6)
        self.assertAlmostEqual(d["radial_dev"], 1.0, places=6)
        self.assertAlmostEqual(d["tangent_norm"], 4.0, places=6)
        self.assertAlmostEqual(d["l2"] ** 2, 1.0 ** 2 + 4.0 ** 2, places=6)

    def test_parallel_case_has_zero_tangent(self):
        d = decompose(torch.tensor([6.0, 0.0]), torch.tensor([2.0, 0.0]))
        self.assertAlmostEqual(d["tangent_norm"], 0.0, places=6)

    def test_legacy_norm_recovery(self):
        z0 = torch.tensor([3.0, 0.0])
        zd = torch.tensor([1.0, 0.0])
        za = torch.tensor([2.0, 0.0])
        self.assertAlmostEqual(norm_recovery_legacy(z0, zd, za), 0.5, places=6)
        self.assertEqual(norm_recovery_legacy(z0, z0, za), 0.0)   # 分母 0 -> 0.0


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行确认失败**
- [ ] **Step 3: 实现 `decomposition.py`**

```python
# -*- coding: utf-8 -*-
"""径向/切向分解：把"离干净嵌入多远"拆成"模长差多少"与"方向偏多少"。

为什么需要（设计动机）：
    旧字段 norm_recovery = (‖z_att‖ − ‖z_deg‖)/(‖z0‖ − ‖z_deg‖) 只分解模长；
    方向变化大时会引入交叉项，于是"L2 正恢复、余弦负恢复"两个口径互相矛盾却
    解释不清（spec §3.5）。改用严格恒等式后两者在同一坐标系里一致。

公式（[ai]，由向量恒等式直接推出）：
    u0 = z0 / ‖z0‖；r(z) = <z, u0>；t(z) = z − r(z)·u0
    ‖z − z0‖² = (r(z) − ‖z0‖)² + ‖t(z)‖²
                径向偏差平方        切向偏差平方

使用举例：
    d = decompose(z_att, z0); d["radial_dev"], d["tangent_norm"]
"""
from __future__ import annotations

from typing import Dict

import torch


def decompose(z, z0) -> Dict[str, float]:
    """径向/切向分解；z、z0 为同维向量（torch tensor 或可转换的序列）。"""
    z = torch.as_tensor(z, dtype=torch.float32).flatten()
    z0 = torch.as_tensor(z0, dtype=torch.float32).flatten()
    n0 = float(z0.norm())
    u0 = z0 / (n0 + 1e-12)
    r = float(torch.dot(z, u0))
    t = z - r * u0
    return {"z_norm": float(z.norm()), "z0_norm": n0, "radial": r,
            "radial_dev": r - n0, "tangent_norm": float(t.norm()),
            "l2": float((z - z0).norm())}


def identity_residual(z, z0) -> float:
    """恒等式残差 |‖z−z0‖² − (径向偏差² + 切向²)|（数值上应为 0）。"""
    d = decompose(z, z0)
    return abs(d["l2"] ** 2 - (d["radial_dev"] ** 2 + d["tangent_norm"] ** 2))


def norm_recovery_legacy(z0, z_deg, z_att) -> float:
    """旧口径（保留为 legacy 字段，spec §3.5）；分母为 0 时返回 0.0。"""
    z0 = torch.as_tensor(z0, dtype=torch.float32).flatten()
    zd = torch.as_tensor(z_deg, dtype=torch.float32).flatten()
    za = torch.as_tensor(z_att, dtype=torch.float32).flatten()
    denom = float(z0.norm() - zd.norm())
    if abs(denom) <= 1e-12:
        return 0.0
    return float(za.norm() - zd.norm()) / denom
```

- [ ] **Step 4: 运行确认通过**
- [ ] **Step 5: 提交**

Run: `G:\Idea\.venv\Scripts\python.exe -m unittest pre.tests.test_pre_decomposition -v` → PASS

```bash
git add TPA/pre/analysis/decomposition.py TPA/pre/tests/test_pre_decomposition.py
git commit -m "fix(pre): 新增径向/切向严格分解，norm_recovery 降级为 legacy 字段"
```

---

## Task 8: 修复 12 —— `align.py` 文档论断纠正（A 类，文档）

**Files:**
- Modify: `TPA/pre/analysis/align.py`（模块 docstring）
- Modify: `TPA/pre/configs/default.yaml`（`pre.embedding.align` 注释同步）
- Test: `TPA/pre/tests/test_pre_align_doc.py`

**Interfaces:** 无新增；仅纠正被实测推翻的论断，并写明两条硬规则（拟合排除目标物品；
同一个 R 同时作用于用户行）。

- [ ] **Step 1: 写失败测试**

```python
# -*- coding: utf-8 -*-
"""align.py 的文档论断必须与实测一致（spec §4.5 #12）。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

TPA_ROOT = Path(__file__).resolve().parents[2]
if str(TPA_ROOT) not in sys.path:
    sys.path.insert(0, str(TPA_ROOT))

from pre.analysis import align  # noqa: E402


class TestDocstring(unittest.TestCase):
    def test_old_claim_removed_and_measurement_present(self):
        doc = align.__doc__ or ""
        self.assertNotIn("0.956", doc)      # 被推翻的旧论断
        self.assertIn("0.74", doc)          # 实测区间
        self.assertIn("排除", doc)          # 拟合排除目标物品
        self.assertIn("用户行", doc)         # 同一个 R 作用于用户行


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行确认失败**

Run: `G:\Idea\.venv\Scripts\python.exe -m unittest pre.tests.test_pre_align_doc -v`
Expected: FAIL（当前 docstring 含 "0.956"）

- [ ] **Step 3: 改写 docstring**（替换"为什么必须映射（问题定义）"整段）

```text
为什么必须映射（问题定义）：
    每一次训练都会产生自己的坐标帧。实测 ml100k + LightGCN：
        mean_row_cos_raw     0.74 ~ 0.79
        mean_row_cos_aligned 0.988（Procrustes 残差 0.136 ~ 0.144）
    也就是说 raw 口径下的大部分位移来自一个近似全局旋转，**不是**物品本身的位移。
    早期文档曾断言"同初始化重训余弦 0.956~0.975 可比"，该论断已被上述实测推翻
    （spec §4.5 #12）：照旧结论用 raw 会得出"所有攻击都是负恢复"的错误结论。
    因此主口径固定为 Procrustes 对齐，raw 仅作诊断。

两条硬规则（spec §3.2）：
    1. 拟合掩码必须**排除目标物品**，否则对齐变换会吸收目标自身位移
       （leakage into the alignment transform）；
    2. 同一个 R 必须同时作用于物品行与**用户行**（共享同一嵌入空间），
       否则计算假用户合力 F 与 audience 余弦时坐标系不一致。
```

- [ ] **Step 4: 运行确认通过**
- [ ] **Step 5: 提交**

Run: `G:\Idea\.venv\Scripts\python.exe -m unittest pre.tests.test_pre_align_doc -v` → PASS

```bash
git add TPA/pre/analysis/align.py TPA/pre/configs/default.yaml TPA/pre/tests/test_pre_align_doc.py
git commit -m "docs(pre): 纠正 align.py 被实测推翻的对齐论断并写明两条硬规则"
```

---

## Task 9: 修复 8 —— 退化数据共享目录 + 指纹（B 类）

**Files:**
- Modify: `TPA/pre/runners/common.py`（新增 `meta_fingerprint`；`degraded_data_dir` 改为 shared）
- Modify: `TPA/pre/runners/degrade.py`
- Modify: `TPA/pre/runners/pipeline.py`（调用点去掉 `model_name`）
- Test: `TPA/pre/tests/test_pre_shared_degrade.py`

**Interfaces:**
- Produces: `meta_fingerprint(meta) -> str`（与 `attacks/uba/generate.py` 同算法，避免跨包依赖）、
  `degraded_data_dir(exp_dir, item_id, ratio) -> Path`（**签名去掉 model_name**）

- [ ] **Step 1: 写失败测试** —— `TPA/pre/tests/test_pre_shared_degrade.py`

```python
# -*- coding: utf-8 -*-
"""退化数据与模型无关：只存一份 shared/，读取时校验 fingerprint。"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

TPA_ROOT = Path(__file__).resolve().parents[2]
if str(TPA_ROOT) not in sys.path:
    sys.path.insert(0, str(TPA_ROOT))

from pre.runners.degrade import build_and_cache_degraded   # noqa: E402


def _meta(n_items=6):
    pairs = [(u, 0) for u in range(10)] + [(u, 1) for u in range(4)]
    return {"num_users": 10, "num_items": n_items, "train_pairs": pairs,
            "test_pairs": [], "user_items": {}}


class TestSharedDegraded(unittest.TestCase):
    def test_two_models_share_one_copy(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = {"dataset": "toy", "pre": {}}
            exp = Path(d)
            a = build_and_cache_degraded(cfg, exp, "lightgcn", _meta(), 0, 0.5, 42)
            b = build_and_cache_degraded(cfg, exp, "mf", _meta(), 0, 0.5, 42)
            self.assertEqual(a["meta_path"], b["meta_path"])
            self.assertIn("degraded/shared/item_00000/ratio_50",
                          a["meta_path"].replace("\\", "/"))
            n = len(list((exp / "raw" / "degraded").glob("*/item_00000/ratio_50")))
            self.assertEqual(n, 1)

    def test_fingerprint_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = {"dataset": "toy", "pre": {}}
            exp = Path(d)
            build_and_cache_degraded(cfg, exp, "lightgcn", _meta(), 0, 0.5, 42)
            with self.assertRaises(ValueError):
                build_and_cache_degraded(cfg, exp, "lightgcn", _meta(n_items=7),
                                         0, 0.5, 42)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行确认失败**
- [ ] **Step 3: 实现**

```python
# common.py
def meta_fingerprint(meta: Dict[str, Any]) -> str:
    """数据指纹：num_users | num_items | sorted(train_pairs) 的 sha256 前 16 位。

    与 attacks/uba/generate.py 的同名函数保持同一算法（两侧无法互相 import：
    pre 属编排层、attacks 属算法层）。
    """
    h = hashlib.sha256()
    h.update(f"{int(meta['num_users'])}|{int(meta['num_items'])}|".encode("utf-8"))
    h.update(repr(sorted((int(u), int(i)) for u, i in meta["train_pairs"]))
             .encode("utf-8"))
    return h.hexdigest()[:16]


def degraded_data_dir(exp_dir: Path, item_id: int, ratio: float) -> Path:
    """退化数据目录（模型无关 -> 只存一份 shared/；spec §1.8）。"""
    return cond_dir(exp_dir, "degraded", model_name="shared", item_id=item_id,
                    label=ratio_dirname(ratio))
```

```python
# degrade.py 缓存命中分支
    base = degraded_data_dir(exp_dir, item_id, ratio)
    meta_path = base / "train_data" / "meta.pkl"
    info_path = base / "deleted_interactions.json"
    if reuse and meta_path.exists() and info_path.exists():
        info = read_json(info_path)
        want = str(info.get("meta_fingerprint", ""))
        actual = meta_fingerprint(clean_meta)
        if want and want != actual:
            raise ValueError(
                f"退化数据指纹与当前 meta 不一致：{meta_path}\n"
                f"  缓存={want}\n  当前={actual}\n"
                f"（若确实换了数据，请删除该目录后重跑）")
        info["meta_path"] = str(meta_path)
        return info
    degraded_meta, info = build_degraded_meta(clean_meta, item_id, ratio, seed)
    info["meta_fingerprint"] = meta_fingerprint(clean_meta)
```

- [ ] **Step 4: 运行确认通过 + 提交**

Run: `G:\Idea\.venv\Scripts\python.exe -m unittest pre.tests.test_pre_shared_degrade pre.tests.test_pre_pipeline -v`
Expected: PASS

```bash
git add TPA/pre/runners/common.py TPA/pre/runners/degrade.py TPA/pre/runners/pipeline.py TPA/pre/tests/test_pre_shared_degrade.py
git commit -m "fix(pre): 退化数据改为 shared 单份存储并加 meta 指纹校验"
```
---

## Task 10: 修复 3 —— TPA `path_builder` 数据源可重定向（B 类）

**Files:**
- Modify: `TPA/attacks/tpa/path_builder.py`
- Modify: `TPA/pre/runners/run_attack.py`
- Test: `TPA/pre/tests/test_pre_tpa_redirect.py`

**Interfaces:**
- Produces: `path_builder.main(config, raw_meta=None)`；
  `pre.runners.run_attack.pre_stage_kwargs(spec, degraded_meta) -> Dict[str, Any]`

- [ ] **Step 1: 写失败测试** —— `TPA/pre/tests/test_pre_tpa_redirect.py`

```python
# -*- coding: utf-8 -*-
"""TPA 前置阶段必须收到退化 meta，而不是回退到干净 processed 目录（修复 3）。"""
from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

TPA_ROOT = Path(__file__).resolve().parents[2]
if str(TPA_ROOT) not in sys.path:
    sys.path.insert(0, str(TPA_ROOT))

from attacks.tpa import path_builder                       # noqa: E402
from pre.runners.run_attack import ATTACK_SPECS, pre_stage_kwargs  # noqa: E402


class TestTpaRedirect(unittest.TestCase):
    def test_path_builder_accepts_raw_meta(self):
        sig = inspect.signature(path_builder.main)
        self.assertIn("raw_meta", sig.parameters)
        self.assertIsNone(sig.parameters["raw_meta"].default)

    def test_pre_stage_kwargs_passes_degraded_meta_for_tpa(self):
        kw = pre_stage_kwargs(ATTACK_SPECS["tpa"], Path("X:/deg/item_1/ratio_50"))
        self.assertEqual(kw, {"raw_meta": Path("X:/deg/item_1/ratio_50")})

    def test_pre_stage_kwargs_empty_for_attacks_without_pre_stage(self):
        self.assertEqual(pre_stage_kwargs(ATTACK_SPECS["random"], Path("X:/d")), {})


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行确认失败**
- [ ] **Step 3: 实现**

```python
# attacks/tpa/path_builder.py
def main(config: Dict[str, Any],
         raw_meta: Path | None = None) -> Dict[str, Any]:
    """构建共现路径画像。

    raw_meta 的意义（修复 3）：generate 阶段早已支持 `main(config, raw_meta=...)`，
    但 path_builder 把路径写死成 `raw_meta_path(config)`（干净 processed 目录）。
    结果是"路径基于干净图、投毒基于退化图"，两阶段看到的图不同却不报错。
    现在两阶段共用同一处数据源解析。
    """
    ...
    meta = load_meta(Path(raw_meta) if raw_meta else raw_meta_path(config))
```

```python
# pre/runners/run_attack.py
def pre_stage_kwargs(spec: Dict[str, str], degraded_meta: Path) -> Dict[str, Any]:
    """前置阶段（TPA path_builder）的调用参数。

    必须把退化 meta 传下去，否则 path_builder 会回退到干净 processed 目录，
    造成"路径用干净图、投毒用退化图"的口径不一致（修复 3）。
    """
    if not spec.get("pre_stage"):
        return {}
    if spec.get("meta_kwarg", "raw_meta") == "raw_meta":
        return {"raw_meta": Path(degraded_meta)}
    return {}


# run_attack 内部调用点（替换原 pre_main(acfg) 与下面那段无人使用的 kwargs 死代码）
        if spec.get("pre_stage"):
            pre_main = _import_callable(spec["pre_stage"])
            pre_main(acfg, **pre_stage_kwargs(spec, degraded_meta_path))
```

- [ ] **Step 4: 运行确认通过 + TPA 闭环冒烟**
- [ ] **Step 5: 提交**

Run: `G:\Idea\.venv\Scripts\python.exe -m unittest pre.tests.test_pre_tpa_redirect -v` → PASS

Run（ml100k，1 物品 1 比例）：
`G:\Idea\.venv\Scripts\python.exe pre/run.py --mode attack --tag tpa-smoke --models lightgcn --attacks tpa --ratios 0.5 --limit-items 1 --epochs 2`
Expected: TPA 在 pre 里跑通一次闭环；`attack.json` 的 `degraded_meta` 与 path 阶段读到的
meta 指纹一致（比对 `item_*/ratio_50/train_data/meta.pkl` 的 sha256）。

```bash
git add TPA/attacks/tpa/path_builder.py TPA/pre/runners/run_attack.py TPA/pre/tests/test_pre_tpa_redirect.py
git commit -m "fix(attacks): TPA path_builder 支持 raw_meta 重定向，两阶段共用退化数据源"
```

---

## Task 11: 修复 4 —— advinject 接入批量注册（B 类）

**Files:**
- Modify: `TPA/attacks/batch/registry.py`
- Modify: `TPA/tests/test_batch_registry.py`
- Test: 同上（既有测试文件内新增用例）

**Interfaces:**
- Produces: `registry.registered_names()` 含 `advinject`；`registry.get("advinject").generate`
  指向 `attacks.advinject.generate.generate`

- [ ] **Step 1: 写失败测试（在既有文件里补用例）**

```python
    def test_advinject_registered_with_generate_entry(self):
        names = registry.registered_names()
        self.assertIn("advinject", names)
        spec = registry.get("advinject")
        # 入口名是 generate（不是 main），注册表必须做适配而不是硬编码 .main
        self.assertTrue(spec.generate.__name__ in ("generate", "main"))
        self.assertTrue(callable(spec.classify))
        self.assertTrue(callable(spec.fit))
```

- [ ] **Step 2: 运行确认失败**

Run: `G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_batch_registry -v`
Expected: FAIL（`AssertionError: 'advinject' not found`）

- [ ] **Step 3: 实现**

```python
# attacks/batch/registry.py
# 入口名差异表：大多数攻击用 main，advinject 的生成入口叫 generate
_ENTRY_POINTS: Dict[str, str] = {"advinject": "generate"}


def _register_builtin() -> None:
    for name in ("bandwagon", "random", "pgd", "tpa", "uba", "advinject"):
        entry = _ENTRY_POINTS.get(name, "main")
        generate_mod = importlib.import_module(f"attacks.{name}.generate")
        register(
            name,
            f"attacks/{name}/config.yaml",
            classify=importlib.import_module(f"attacks.{name}.classify").main,
            generate=getattr(generate_mod, entry),
            fit=importlib.import_module(f"attacks.{name}.fit").main,
        )
```

- [ ] **Step 4: 运行确认通过 + 契约自查**
- [ ] **Step 5: 提交**

Run: `G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_batch_registry tests.test_advinject_contract -v` → PASS

按 AGENTS §6.5 四项自查逐条核对（config 可展开 / 三个 main 可调用 / generate 产物路径可归位 /
fit 指标可被 aggregate 解析），把结论写进 `attacks/advinject/docs/DESIGN.md` 的"批量接入"小节。

```bash
git add TPA/attacks/batch/registry.py TPA/tests/test_batch_registry.py TPA/attacks/advinject/docs/DESIGN.md
git commit -m "fix(attacks): advinject 接入批量注册表（入口名 generate 适配）"
```

---

## Task 12: 修复 10 —— `uba --mode all` 不再无条件跑 estimate（B 类）

**Files:**
- Modify: `TPA/attacks/uba/estimate.py`（method 守卫 + 缓存写指纹）
- Modify: `TPA/attacks/uba/run.py`（打印跳过原因）
- Test: `TPA/tests/test_uba_estimate_guard.py`

**Interfaces:**
- Produces: `estimate.main(config, meta=None) -> Dict[str, Any]`；
  `method != "surrogate"` 时返回 `{"skipped": True, "reason": "method=path"}`

- [ ] **Step 1: 写失败测试**

```python
# -*- coding: utf-8 -*-
"""method=path 时 estimate 阶段必须直接跳过（修复 10）。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

TPA_ROOT = Path(__file__).resolve().parents[1]
if str(TPA_ROOT) not in sys.path:
    sys.path.insert(0, str(TPA_ROOT))

from attacks.uba import estimate as est   # noqa: E402


class TestEstimateGuard(unittest.TestCase):
    def test_path_method_skips_without_training(self):
        cfg = {"dataset": "toy", "seed": 42,
               "attack": {"uba": {"treatment": {"method": "path"}}}}
        called = []
        orig = est.treatment_effect_surrogate
        est.treatment_effect_surrogate = lambda *a, **k: called.append(1)
        try:
            out = est.main(cfg)
        finally:
            est.treatment_effect_surrogate = orig
        self.assertTrue(out.get("skipped"))
        self.assertIn("path", str(out.get("reason")))
        self.assertEqual(called, [])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行确认失败**
- [ ] **Step 3: 实现**

```python
# attacks/uba/estimate.py::main 顶部（签名之后）
    """estimate 阶段入口。"""
    treatment = config.get("attack", {}).get("uba", {}).get("treatment", {})
    method = str(treatment.get("method", "path"))
    if method != "surrogate":
        print(f"[estimate] treatment.method={method}，跳过代理模型处理效应估计"
              f"（Y 由 path 支路在 data 阶段按需计算并缓存）")
        return {"skipped": True, "reason": f"method={method}"}
```

并把尾部 `effect_cache_path(...)` 调用补上数据指纹（与 Task 4 的 key 规则一致）：

```python
                  beta=float(treatment_cfg(config).get("beta", 1.0)),
                  seed=int(config.get("seed", 42)),
                  data_fingerprint=meta_fingerprint(meta)))
```

（`meta_fingerprint` 从 `attacks.uba.generate` import。）

```python
# attacks/uba/run.py
    if mode in ("estimate", "all"):
        res = estimate_main(cfg)
        if isinstance(res, dict) and res.get("skipped"):
            print(f"[run] estimate 已跳过：{res.get('reason')}")
```

- [ ] **Step 4: 运行确认通过**
- [ ] **Step 5: 提交**

Run: `G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_uba_estimate_guard tests.test_uba_generate -v` → PASS

```bash
git add TPA/attacks/uba/estimate.py TPA/attacks/uba/run.py TPA/tests/test_uba_estimate_guard.py
git commit -m "fix(attacks): uba estimate 阶段按 treatment.method 守卫，path 支路不再空跑"
```

---

## Task 13: 修复 11 —— manifest provenance 链 + 历史追加（B 类）

**Files:**
- Modify: `TPA/pre/runners/common.py`（`write_manifest`）
- Modify: `TPA/pre/run.py`
- Test: `TPA/pre/tests/test_pre_manifest.py`

**Interfaces:**
- Produces: `write_manifest(cfg, tag, exp_dir, argv, *, dataset=None, victim_model=None,
  target_file_sha256="", extra=None) -> Dict[str, Any]`；写 `manifest.json`（本次）
  与 `manifest_history.jsonl`（全部，追加）

- [ ] **Step 1: 写失败测试**

```python
# -*- coding: utf-8 -*-
"""manifest 必须含 §2.7 全部字段，且多次运行追加到 manifest_history.jsonl。"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

TPA_ROOT = Path(__file__).resolve().parents[2]
if str(TPA_ROOT) not in sys.path:
    sys.path.insert(0, str(TPA_ROOT))

from pre.protocol import PROTOCOL_FIELDS              # noqa: E402
from pre.runners.common import write_manifest          # noqa: E402


def _cfg():
    return {"dataset": "toy", "k": 10,
            "pre": {"models": ["lightgcn"], "attacks": ["random"],
                    "training": {"seeds": [42]},
                    "attack": {"rho_A": 0.03, "rho_P": 0.315, "rho_E": 0.0098}}}


class TestManifest(unittest.TestCase):
    def test_fields_complete_and_history_appended(self):
        with tempfile.TemporaryDirectory() as d:
            exp = Path(d)
            m1 = write_manifest(_cfg(), "tag1", exp, ["pre/run.py", "--mode", "all"],
                                dataset="toy", victim_model="lightgcn",
                                target_file_sha256="abc")
            m2 = write_manifest(_cfg(), "tag1", exp, ["pre/run.py", "--mode", "analyze"],
                                dataset="toy", victim_model="lightgcn",
                                target_file_sha256="abc")
            for f in PROTOCOL_FIELDS:
                self.assertIn(f, m1)
            self.assertEqual(m1["rho_A"], 0.03)
            hist = (exp / "manifest_history.jsonl").read_text("utf-8").splitlines()
            self.assertEqual(len(hist), 2)
            last = json.loads((exp / "manifest.json").read_text("utf-8"))
            self.assertEqual(last["argv"][-1], "analyze")     # 保留最后一次
            self.assertEqual(json.loads(hist[0])["argv"][-1], "all")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行确认失败**
- [ ] **Step 3: 实现**

```python
# common.py
def write_manifest(cfg: Dict[str, Any], tag: str, exp_dir: Path,
                   argv: Sequence[str], *, dataset: Optional[str] = None,
                   victim_model: Optional[str] = None,
                   target_file_sha256: str = "",
                   extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """写 manifest.json（本次运行）并追加 manifest_history.jsonl（历次运行）。

    为什么追加（spec §5.5）：run.py 每次调用都会重写 manifest.json，断点续跑后
    无法回答"这个目录被跑过几次、每次用什么参数"。历史文件是唯一答案。
    """
    from pre.protocol import manifest_fields
    fields = manifest_fields(
        cfg, dataset=dataset or str(cfg.get("dataset", "")),
        victim_model=victim_model or ",".join(cfg.get("pre", {}).get("models", [])),
        target_file_sha256=target_file_sha256, k=int(cfg.get("k", 10)))
    rec = {**fields, "tag": str(tag), "argv": [str(a) for a in argv],
           "created_at": now_iso()}
    if extra:
        rec.update(extra)
    save_json(rec, Path(exp_dir) / "manifest.json")
    hist = Path(exp_dir) / "manifest_history.jsonl"
    with open(hist, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec
```

`run.py` 把原来的 `save_json({...}, exp_dir / "manifest.json")` 换成：

```python
    target_sha = sha256_file(frozen_path(str(cfg["dataset"])))   # pre.protocol.sha256_file
    write_manifest(cfg, tag, exp_dir, sys.argv, dataset=str(cfg["dataset"]),
                   target_file_sha256=target_sha)
```

（`model_seed` / `ratio` / `attack` 等条件级字段由 Task 18 在逐条件产物里补写。）

- [ ] **Step 4: 运行确认通过**
- [ ] **Step 5: 提交**

Run: `G:\Idea\.venv\Scripts\python.exe -m unittest pre.tests.test_pre_manifest -v` → PASS

```bash
git add TPA/pre/runners/common.py TPA/pre/run.py TPA/pre/tests/test_pre_manifest.py
git commit -m "fix(pre): manifest 写入 §2.7 provenance 字段并追加历史记录"
```
---

## Task 14: 分析层 1/6 —— `metrics.py` 纯函数口径（P3）

**Files:**
- Create: `TPA/pre/analysis/metrics.py`
- Test: `TPA/pre/tests/test_pre_metrics.py`

**Interfaces:**
- Produces:
  - `l2(a, b) -> float`、`cosine(a, b) -> float`（1 − cos）、`d_rel(a, b) -> float`（‖a−b‖/‖b‖）
  - `delta_abs(d_deg, d_att) -> float`
  - `rr(d_deg, d_att) -> float`（`d_deg <= eps` 时返回 0.0）
  - `rr_pc(e_deg, e_att) -> float`
  - `EPS = 1e-12`

- [ ] **Step 1: 写失败测试** —— `TPA/pre/tests/test_pre_metrics.py`

```python
# -*- coding: utf-8 -*-
"""距离与恢复率纯函数（spec §3.4）：定义与边界。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

TPA_ROOT = Path(__file__).resolve().parents[2]
if str(TPA_ROOT) not in sys.path:
    sys.path.insert(0, str(TPA_ROOT))

from pre.analysis.metrics import (cosine, d_rel, delta_abs, l2,   # noqa: E402
                                  rr, rr_pc)


class TestDistances(unittest.TestCase):
    def test_l2_and_d_rel(self):
        a = torch.tensor([0.0, 0.0])
        b = torch.tensor([3.0, 4.0])
        self.assertAlmostEqual(l2(a, b), 5.0, places=6)
        self.assertAlmostEqual(d_rel(a, b), 1.0, places=6)     # 5 / 5

    def test_cosine_orthogonal_and_parallel(self):
        self.assertAlmostEqual(cosine(torch.tensor([1.0, 0.0]),
                                      torch.tensor([1.0, 0.0])), 0.0, places=6)
        self.assertAlmostEqual(cosine(torch.tensor([1.0, 0.0]),
                                      torch.tensor([0.0, 1.0])), 1.0, places=6)


class TestRecovery(unittest.TestCase):
    def test_delta_and_rr(self):
        self.assertAlmostEqual(delta_abs(1.0, 0.75), 0.25, places=6)
        self.assertAlmostEqual(rr(1.0, 0.75), 0.25, places=6)
        self.assertAlmostEqual(rr(1.0, 1.25), -0.25, places=6)

    def test_rr_zero_denominator_returns_zero(self):
        self.assertEqual(rr(0.0, 0.5), 0.0)

    def test_rr_pc_uses_excess_over_control(self):
        # E_deg = 1.016-0.192 = 0.824；E_att = 0.731-0.380 = 0.351
        expected = (0.824 - 0.351) / 0.824
        self.assertAlmostEqual(rr_pc(0.824, 0.351), expected, places=6)

    def test_rr_pc_explodes_when_e_deg_tiny_so_guard_returns_zero(self):
        self.assertEqual(rr_pc(0.0, 0.1), 0.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行确认失败**
- [ ] **Step 3: 实现**（每个函数带"为什么/是什么/公式出处/举例"注释）

```python
EPS = 1e-12             # 除零保护；与 distances.py 的 1e-12 保持一致


def rr(d_deg: float, d_att: float) -> float:
    """原始归一化恢复率 (d_deg − d_att) / d_deg（spec §3.4）。

    d_deg 趋近 0（几乎没退化）时比值无定义：返回 0.0，并由上层用 pc_valid /
    coverage 标记该条件"不可分辨"，而不是产出爆炸值。
    """
    if d_deg <= EPS:
        return 0.0
    return float((d_deg - d_att) / d_deg)


def rr_pc(e_deg: float, e_att: float) -> float:
    """安慰剂控制恢复率 (E_deg − E_att) / E_deg（机制主指标，spec §3.3/§3.4）。"""
    if e_deg <= EPS:
        return 0.0
    return float((e_deg - e_att) / e_deg)
```

- [ ] **Step 4: 运行确认通过**
- [ ] **Step 5: 提交**

Run: `G:\Idea\.venv\Scripts\python.exe -m unittest pre.tests.test_pre_metrics -v` → PASS

```bash
git add TPA/pre/analysis/metrics.py TPA/pre/tests/test_pre_metrics.py
git commit -m "feat(pre): 分析层距离与恢复率纯函数模块"
```

---

## Task 15: 分析层 2/6 —— `placebo.py` 对照与阈值（P3）

**Files:**
- Create: `TPA/pre/analysis/placebo.py`
- Test: `TPA/pre/tests/test_pre_placebo.py`

**Interfaces:**
- Produces:
  - `item_distances(ref, cond, exclude_item=None) -> torch.Tensor`
  - `control_stats(dists, exclude_items=()) -> Dict[str, float]`
    （`d_ctrl_median` / `d_ctrl_mean` / `d_ctrl_q25` / `d_ctrl_q75` / `n_control`）
  - `excess(d_tgt, d_ctrl_median) -> float`
  - `pc_threshold(control_dists, sigma=5.0) -> float`
  - `pc_valid(e_deg, threshold) -> bool`
  - `placebo_frame_check(d_ctrl_median, d_deg, min_ratio=3.0) -> bool`

- [ ] **Step 1: 写失败测试**

```python
# -*- coding: utf-8 -*-
"""安慰剂对照：d_ctrl 用中位数，pc_valid 阈值由 placebo 分布自身导出（spec §3.3/§3.4）。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

TPA_ROOT = Path(__file__).resolve().parents[2]
if str(TPA_ROOT) not in sys.path:
    sys.path.insert(0, str(TPA_ROOT))

from pre.analysis.placebo import (control_stats, excess, pc_threshold,  # noqa: E402
                                  pc_valid)


class TestControlStats(unittest.TestCase):
    def test_median_is_long_tail_robust(self):
        d = torch.tensor([0.18, 0.19, 0.20, 5.0])
        s = control_stats(d)
        self.assertAlmostEqual(s["d_ctrl_median"], 0.195, places=6)
        self.assertGreater(s["d_ctrl_mean"], 1.0)          # mean 被长尾拉高
        self.assertEqual(s["n_control"], 4)

    def test_excess_uses_median(self):
        self.assertAlmostEqual(excess(1.016, 0.192), 0.824, places=6)


class TestThreshold(unittest.TestCase):
    def test_threshold_is_mad_based_not_arbitrary(self):
        d_tight = torch.full((50,), 0.19) + torch.linspace(-0.001, 0.001, 50)
        thr = pc_threshold(d_tight)
        self.assertLess(thr, 0.01)
        self.assertTrue(pc_valid(0.824, thr))

    def test_wide_placebo_distribution_invalidates(self):
        d_wide = torch.linspace(0.0, 5.0, 50)
        thr = pc_threshold(d_wide)
        self.assertFalse(pc_valid(0.824, thr))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行确认失败**
- [ ] **Step 3: 实现**

```python
# -*- coding: utf-8 -*-
"""安慰剂对照（placebo）：用"未受影响的物品"给出该条件的噪声底。

为什么需要（设计动机）：
    删除只影响目标物品的边，其余物品的位移完全来自"重训带来的坐标漂移 + 采样噪声"。
    把目标位移减去这个噪声底，才能回答"退化本身造成了多少可分辨的位移"（spec §3.3）。
    raw 坐标系下 d_ctrl ≈ 2.15 与目标位移同量级 -> E ≈ 0，指标自己就判定该坐标系
    无分辨力；Procrustes 下 d_ctrl ≈ 0.19、d_deg ≈ 1.02，分离度约 5 倍 —— 这同时
    是对齐口径的**自检**。

公式（spec §3.3/§3.4）：
    d_ctrl(c) = median_{j ∈ C} d_j        C = 未受影响物品集合
    E(c)      = d_tgt(c) − d_ctrl(c)
    RR_pc     = (E_deg − E_att) / E_deg
    阈值规则：thr = max(1e-9, sigma · MAD(d_ctrl))，MAD = median(|d − median(d)|)
    pc_valid  = E_deg > thr

约束：安慰剂对照只能用于报告与归一化，**不允许用于挑选目标物品**（spec §1.4）。
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch


def item_distances(ref: torch.Tensor, cond: torch.Tensor,
                   exclude_item: Optional[int] = None) -> torch.Tensor:
    """逐物品位移 ‖cond_j − ref_j‖（可排除目标物品本身）。"""
    d = (cond - ref).norm(dim=1)
    if exclude_item is not None and 0 <= int(exclude_item) < d.numel():
        d = torch.cat([d[:int(exclude_item)], d[int(exclude_item) + 1:]])
    return d


def control_stats(dists: torch.Tensor,
                  exclude_items: Iterable[int] = ()) -> Dict[str, float]:
    """对照分布统计：median 为主口径，mean/q25/q75 作诊断（spec §3.3 拍板项 6）。"""
    d = torch.as_tensor(dists, dtype=torch.float32).flatten()
    for i in sorted({int(x) for x in exclude_items}, reverse=True):
        if 0 <= i < d.numel():
            d = torch.cat([d[:i], d[i + 1:]])
    if d.numel() == 0:
        raise ValueError("对照组为空：检查 mask/exclude_items")
    return {"d_ctrl_median": float(d.median()),
            "d_ctrl_mean": float(d.mean()),
            "d_ctrl_q25": float(d.quantile(0.25)),
            "d_ctrl_q75": float(d.quantile(0.75)),
            "n_control": int(d.numel())}


def excess(d_tgt: float, d_ctrl_median: float) -> float:
    """超噪声位移 E = d_tgt − d_ctrl（median 口径）。"""
    return float(d_tgt) - float(d_ctrl_median)


def pc_threshold(control_dists: torch.Tensor, sigma: float = 5.0) -> float:
    """pc_valid 阈值：thr = max(1e-9, sigma · MAD(d_ctrl))。

    为什么不取固定整数（spec §3.4）：阈值必须反映该条件自身的噪声尺度；MAD 对长尾
    稳健（用标准差会被少数长尾物品抬高，从而把有效条件误判为不可分辨）。
    """
    d = torch.as_tensor(control_dists, dtype=torch.float32).flatten()
    mad = float((d - d.median()).abs().median())
    return max(1e-9, float(sigma) * mad)


def pc_valid(e_deg: float, threshold: float) -> bool:
    """RR_pc 是否有分辨力；False 时照常报告 RR / Delta_abs（spec §3.4）。"""
    return float(e_deg) > float(threshold)


def placebo_frame_check(d_ctrl_median: float, d_deg: float,
                        min_ratio: float = 3.0) -> bool:
    """坐标系自检：退化位移必须明显高于噪声底，否则该参考帧不可用。"""
    if float(d_ctrl_median) <= 0:
        return False
    return (float(d_deg) / float(d_ctrl_median)) >= float(min_ratio)
```

- [ ] **Step 4: 运行确认通过**
- [ ] **Step 5: 提交**

Run: `G:\Idea\.venv\Scripts\python.exe -m unittest pre.tests.test_pre_placebo -v` → PASS

```bash
git add TPA/pre/analysis/placebo.py TPA/pre/tests/test_pre_placebo.py
git commit -m "feat(pre): 安慰剂对照模块（d_ctrl/E/pc_valid 与 MAD 阈值）"
```

---

## Task 16: 分析层 3/6 —— `concentration.py` + `audience.py`（P3）

**Files:**
- Create: `TPA/pre/analysis/concentration.py`
- Create: `TPA/pre/analysis/audience.py`
- Test: `TPA/pre/tests/test_pre_force.py`

**Interfaces:**
- Produces（concentration）：`concentration(F) -> float`、`pairwise_cos(F) -> float`、
  `cos_to_vector(F, v) -> float`
- Produces（audience）：`fixed_audience(train_pairs, item_id) -> List[int]`、
  `remaining_audience(train_pairs_r, item_id) -> List[int]`、
  `audience_mean(U, users) -> torch.Tensor`、`cos_force_audience(F, U, users) -> float`

- [ ] **Step 1: 写失败测试** —— `TPA/pre/tests/test_pre_force.py`

```python
# -*- coding: utf-8 -*-
"""力结构：集中度、两两余弦、与受众方向的余弦（spec §3.6）。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

TPA_ROOT = Path(__file__).resolve().parents[2]
if str(TPA_ROOT) not in sys.path:
    sys.path.insert(0, str(TPA_ROOT))

from pre.analysis.audience import (audience_mean, cos_force_audience,  # noqa: E402
                                   fixed_audience, remaining_audience)
from pre.analysis.concentration import (concentration, pairwise_cos,   # noqa: E402
                                        cos_to_vector)


class TestConcentration(unittest.TestCase):
    def test_identical_rows_give_one(self):
        F = torch.ones(5, 3)
        self.assertAlmostEqual(concentration(F), 1.0, places=6)
        self.assertAlmostEqual(pairwise_cos(F), 1.0, places=6)

    def test_opposite_rows_give_zero_mean(self):
        F = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
        self.assertAlmostEqual(concentration(F), 0.0, places=6)
        self.assertAlmostEqual(pairwise_cos(F), -1.0, places=6)

    def test_cos_to_vector(self):
        F = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        self.assertAlmostEqual(cos_to_vector(F, torch.tensor([2.0, 0.0])), 1.0,
                               places=6)


class TestAudience(unittest.TestCase):
    def test_fixed_vs_remaining_audience(self):
        pairs_clean = [(0, 7), (1, 7), (2, 8)]
        pairs_r = [(0, 7), (2, 8)]
        self.assertEqual(fixed_audience(pairs_clean, 7), [0, 1])
        self.assertEqual(remaining_audience(pairs_r, 7), [0])

    def test_cos_force_audience_uses_same_frame(self):
        U = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        F = torch.tensor([[2.0, 2.0], [2.0, 2.0]])
        self.assertAlmostEqual(cos_force_audience(F, U, [0, 1]), 1.0, places=6)

    def test_audience_mean_empty_raises(self):
        with self.assertRaises(ValueError):
            audience_mean(torch.eye(3), [])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行确认失败**
- [ ] **Step 3: 实现**

```python
# concentration.py
def concentration(F: torch.Tensor) -> float:
    """‖mean(F)‖ / mean(‖f‖)：1 表示所有假用户完全同向（spec §3.6）。

    为什么用比值而不是 ‖mean(F)‖：不同数据集/模型的嵌入尺度不同，比值才是无量纲量，
    可以跨数据集比较（spec §3.9）。
    """
    F = torch.as_tensor(F, dtype=torch.float32)
    if F.ndim == 1:
        F = F.unsqueeze(0)
    mean_norm = float(F.norm(dim=1).mean())
    if mean_norm <= 1e-12:
        return 0.0
    return float(F.mean(dim=0).norm()) / mean_norm


def pairwise_cos(F: torch.Tensor) -> float:
    """两两余弦均值；行数 < 2 时定义 1.0（单个向量与自身同向）。"""
    F = torch.as_tensor(F, dtype=torch.float32)
    if F.ndim == 1 or F.shape[0] < 2:
        return 1.0
    Fn = F / (F.norm(dim=1, keepdim=True) + 1e-12)
    C = Fn @ Fn.T
    n = F.shape[0]
    iu = torch.triu_indices(n, n, offset=1)
    return float(C[iu[0], iu[1]].mean())


def cos_to_vector(F: torch.Tensor, v: torch.Tensor) -> float:
    """cos(mean(F), v)：假用户合力与给定方向（z0 / audience）的余弦。"""
    F = torch.as_tensor(F, dtype=torch.float32)
    if F.ndim == 1:
        F = F.unsqueeze(0)
    v = torch.as_tensor(v, dtype=torch.float32).flatten()
    m = F.mean(dim=0)
    return float(torch.dot(m, v) / (m.norm() * v.norm() + 1e-12))
```

```python
# audience.py
def fixed_audience(train_pairs, item_id: int) -> List[int]:
    """固定受众 A_i = {u : (u,i) ∈ 干净训练集}（主口径，spec §3.6）。

    为什么固定：比例轴上的曲线变化应只来自 force 变化；若受众集合随 r 收缩，
    余弦变化就混入"受众变了"这个额外原因。
    """
    return sorted({int(u) for u, i in train_pairs if int(i) == int(item_id)})


def remaining_audience(train_pairs_r, item_id: int) -> List[int]:
    """剩余受众 A_i(r)（辅口径，仅作 supplementary diagnostic）。"""
    return fixed_audience(train_pairs_r, item_id)


def audience_mean(U: torch.Tensor, users: Sequence[int]) -> torch.Tensor:
    """受众平均用户嵌入；users 为空时抛 ValueError（不返回 NaN 污染下游）。"""
    users = [int(u) for u in users]
    if not users:
        raise ValueError("受众集合为空：目标物品在干净训练集里没有任何交互？")
    U = torch.as_tensor(U, dtype=torch.float32)
    return U[torch.tensor(users, dtype=torch.long)].mean(dim=0)


def cos_force_audience(F: torch.Tensor, U: torch.Tensor,
                       users: Sequence[int]) -> float:
    """cos(mean(F), mean(U[audience]))；F 与 U 必须已在**同一**坐标系（同一个 R）。"""
    return cos_to_vector(F, audience_mean(U, users))
```

- [ ] **Step 4: 运行确认通过**
- [ ] **Step 5: 提交**

Run: `G:\Idea\.venv\Scripts\python.exe -m unittest pre.tests.test_pre_force -v` → PASS

```bash
git add TPA/pre/analysis/concentration.py TPA/pre/analysis/audience.py TPA/pre/tests/test_pre_force.py
git commit -m "feat(pre): 力结构模块（集中度/两两余弦/固定与剩余受众）"
```
---

## Task 17: 分析层 4/6 —— 条件级曝光验证（K=10/20，P3）

**Files:**
- Modify: `TPA/pre/analysis/exposure.py`
- Test: `TPA/pre/tests/test_pre_target_exposure.py`

**Interfaces:**
- Produces: `target_metrics_from_scores(scores, users, meta, item_id, ks=(10, 20))
  -> Dict[str, float]`；`target_exposure(model, meta, item_id, ks=(10, 20))
  -> Dict[str, float]`

- [ ] **Step 1: 写失败测试** —— `TPA/pre/tests/test_pre_target_exposure.py`

```python
# -*- coding: utf-8 -*-
"""条件级目标曝光：K=10 与 20 同时输出（承载 S4，零训练成本）。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

TPA_ROOT = Path(__file__).resolve().parents[2]
if str(TPA_ROOT) not in sys.path:
    sys.path.insert(0, str(TPA_ROOT))

from pre.analysis.exposure import target_metrics_from_scores   # noqa: E402


class TestTargetExposure(unittest.TestCase):
    def test_k10_and_k20_reported(self):
        users = [0, 1]
        S = torch.zeros(2, 20)
        S[0, 0] = 10.0                       # 物品 0 对用户 0 排第 1
        S[1, 0] = 1.0                        # 物品 0 对用户 1 排第 15
        for j in range(1, 20):
            S[1, j] = 5.0
        meta = {"num_users": 2, "num_items": 20, "train_pairs": [],
                "test_pairs": [(0, 0), (1, 0)], "user_items": {}}
        out = target_metrics_from_scores(S, users, meta, 0, ks=(10, 20))
        self.assertAlmostEqual(out["target_hr@10"], 0.5, places=6)   # 只有用户 0 命中
        self.assertAlmostEqual(out["target_hr@20"], 1.0, places=6)   # 两个都命中
        self.assertIn("target_ndcg@10", out)
        self.assertIn("target_ndcg@20", out)
        self.assertEqual(out["target_n_elig"], 2)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行确认失败**
- [ ] **Step 3: 实现**

```python
def target_metrics_from_scores(scores: torch.Tensor, users: Sequence[int],
                               meta: Dict[str, Any], item_id: int,
                               ks: Sequence[int] = (10, 20)) -> Dict[str, float]:
    """目标物品在给定打分矩阵下的 HR@K / NDCG@K（多个 K 一次输出，承载 S4）。"""
    out: Dict[str, float] = {}
    for k in ks:
        exp = exposure_from_scores(scores, users, meta, int(k)).get(int(item_id), {})
        out[f"target_hr@{k}"] = float(exp.get("hr@k", 0.0))
        out[f"target_ndcg@{k}"] = float(exp.get("ndcg@k", 0.0))
        out["target_n_elig"] = int(exp.get("n_elig", 0))
    return out


def target_exposure(model: Any, meta: Dict[str, Any], item_id: int,
                    ks: Sequence[int] = (10, 20)) -> Dict[str, float]:
    """条件模型的曝光指标。

    口径约束（spec §3.7）：必须在**模型自身坐标系**里算，绝不能用对齐后的嵌入
    —— 对齐变换只服务距离分析（align.py 文档已声明）。
    """
    S, users, _ = ranking_scores(model, meta["test_pairs"])
    return target_metrics_from_scores(S, users, meta, int(item_id), ks=ks)
```

- [ ] **Step 4: 运行确认通过**
- [ ] **Step 5: 提交**

Run: `G:\Idea\.venv\Scripts\python.exe -m unittest pre.tests.test_pre_target_exposure -v` → PASS

```bash
git add TPA/pre/analysis/exposure.py TPA/pre/tests/test_pre_target_exposure.py
git commit -m "feat(pre): 条件级目标曝光指标（K=10/20，承载 S4）"
```

---

## Task 18: 分析层 5/6 —— 两级聚合与 legacy/corrected 双输出（P3）

**Files:**
- Modify: `TPA/pre/analysis/aggregate.py`
- Modify: `TPA/pre/runners/common.py`（`experiment_dir` 新布局）
- Test: `TPA/pre/tests/test_pre_aggregate.py`

**Interfaces:**
- Produces:
  - `aggregate_two_level(rows) -> List[Dict[str, Any]]`
  - `write_outputs(exp_dir, rows, protocol="corrected") -> Dict[str, Path]`
  - `experiment_dir(cfg, tag)` 创建 `raw/`、`analysis/{legacy,corrected}`、`tables/`、`figures/`
  - `NUMERIC_KEYS: tuple[str, ...]`（聚合与出图共用的指标白名单）

- [ ] **Step 1: 写失败测试** —— `TPA/pre/tests/test_pre_aggregate.py`

```python
# -*- coding: utf-8 -*-
"""两级聚合与产物布局（spec §3.8/§4.6/§3.10）。"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

TPA_ROOT = Path(__file__).resolve().parents[2]
if str(TPA_ROOT) not in sys.path:
    sys.path.insert(0, str(TPA_ROOT))

from pre.analysis.aggregate import aggregate_two_level          # noqa: E402
from pre.runners.common import experiment_dir                   # noqa: E402


def _row(item, seed, rr_value):
    return {"dataset": "toy", "model_name": "lightgcn", "attack": "uba",
            "ratio": 0.5, "seed": seed, "item_id": item, "rr": rr_value,
            "rr_pc": rr_value / 2.0, "delta_abs": rr_value}


class TestTwoLevel(unittest.TestCase):
    def test_item_mean_then_seed_stats(self):
        rows = [_row(1, 42, 0.2), _row(2, 42, 0.4),     # seed 42 物品均值 0.3
                _row(1, 43, 0.6), _row(2, 43, 0.8)]     # seed 43 物品均值 0.7
        agg = aggregate_two_level(rows)
        self.assertEqual(len(agg), 1)
        a = agg[0]
        self.assertAlmostEqual(a["rr_mean"], 0.5, places=6)
        # 样本标准差（ddof=1）：两个值 0.3/0.7 -> 0.2*sqrt(2)
        self.assertAlmostEqual(a["rr_sd"], 0.2 * (2 ** 0.5), places=6)
        self.assertEqual(a["n_seeds"], 2)
        self.assertEqual(a["n_items"], 2)

    def test_single_seed_gets_zero_sd(self):
        agg = aggregate_two_level([_row(1, 42, 0.5)])
        self.assertEqual(agg[0]["rr_sd"], 0.0)


class TestOutputLayout(unittest.TestCase):
    def test_dirs_created(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = {"output": {"dir": d}, "pre": {}}
            exp = experiment_dir(cfg, "tagX")
            for sub in ("raw", "analysis/legacy", "analysis/corrected",
                        "tables", "figures"):
                self.assertTrue((exp / sub).exists(), sub)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行确认失败**
- [ ] **Step 3: 实现**

```python
# common.py::experiment_dir
def experiment_dir(cfg: Dict[str, Any], tag: str) -> Path:
    """单次实验目录：outputs/<tag>/{raw,analysis/{legacy,corrected},tables,figures}。

    为什么 raw 与 analysis 分层（spec §3.10）：重新分析不需要重训，重新训练也不该
    动分析产物；legacy/corrected 并存让"新旧口径差异"可审计（spec §4.6）。
    """
    d = output_root(cfg) / tag
    for sub in ("raw", "analysis/legacy", "analysis/corrected", "tables", "figures"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d
```

```python
# aggregate.py
NUMERIC_KEYS = ("d_deg", "d_att", "delta_abs", "rr", "d_ctrl_median", "d_ctrl_mean",
                "e_deg", "e_att", "rr_pc", "d_rel", "concentration", "pairwise_cos",
                "cos_F_z0", "cos_F_aud", "radial_dev", "tangent_norm",
                "norm_recovery_legacy", "pc_valid", "target_hr@10", "target_ndcg@10",
                "target_hr@20", "target_ndcg@20")


def _mean(xs):
    return float(sum(xs) / len(xs))


def aggregate_two_level(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """两级聚合（spec §3.8）。

    第一级：同一 (dataset, model, attack, ratio, seed) 内对**物品**求均值
            —— 保证每个物品等权；
    第二级：对 **seed** 求均值与**样本标准差（ddof=1）**
            —— 报告模型侧方差，单 seed 时 sd = 0.0。

    为什么不能直接对所有行求均值：一个物品有 6 个比例、另一个只有 1 个成功条件时，
    行数多的物品会主导结果（这正是"某物品主导结论"风险的来源）。
    """
    key_cols = ("dataset", "model_name", "attack", "ratio")
    first: Dict[tuple, Dict[str, List[float]]] = {}
    for r in rows:
        k = tuple(r[c] for c in key_cols) + (int(r["seed"]),)
        bucket = first.setdefault(k, {})
        for m in NUMERIC_KEYS:
            v = r.get(m)
            if isinstance(v, bool):
                v = float(v)
            if isinstance(v, (int, float)):
                bucket.setdefault(m, []).append(float(v))
    per_seed: List[Dict[str, Any]] = []
    for k, vals in first.items():
        rec = dict(zip(key_cols + ("seed",), k))
        rec.update({m: _mean(v) for m, v in vals.items()})
        rec["n_items"] = len(vals.get("rr", [])) or max(
            (len(v) for v in vals.values()), default=0)
        per_seed.append(rec)
    grouped: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in per_seed:
        grouped.setdefault(tuple(r[c] for c in key_cols), []).append(r)
    out: List[Dict[str, Any]] = []
    for k, group in grouped.items():
        rec = dict(zip(key_cols, k))
        rec["n_seeds"] = len(group)
        rec["n_items"] = max(g.get("n_items", 0) for g in group)
        for m in NUMERIC_KEYS:
            vals = [g[m] for g in group if m in g]
            if not vals:
                continue
            mean = _mean(vals)
            rec[f"{m}_mean"] = mean
            rec[f"{m}_sd"] = (float((sum((v - mean) ** 2 for v in vals)
                                     / (len(vals) - 1)) ** 0.5)
                              if len(vals) > 1 else 0.0)
        out.append(rec)
    return sorted(out, key=lambda r: (r["dataset"], r["model_name"], r["attack"],
                                      r["ratio"]))
```

`write_outputs(exp_dir, rows, protocol)` 落盘规则：

```
analysis/<protocol>/results.csv   逐 (model,item,ratio,seed,attack) 明细
analysis/<protocol>/summary.json  {rows, coverage, analysis_protocol}
tables/per_item.csv               逐物品两级聚合（含 item 维度）
tables/aggregate.csv              不带 item 的两级聚合（论文表用）
```

并在 `manifest.json` 里追加
`analysis_protocol: {"primary": "corrected", "legacy": "legacy", "version": PROTOCOL_VERSION}`；
`legacy` 目录由一次性迁移脚本（`pre/analysis/legacy_export.py`，只读旧布局）产出，
**只用于审计**。

- [ ] **Step 4: 运行确认通过**
- [ ] **Step 5: 提交**

Run: `G:\Idea\.venv\Scripts\python.exe -m unittest pre.tests.test_pre_aggregate -v` → PASS

```bash
git add TPA/pre/analysis/aggregate.py TPA/pre/runners/common.py TPA/pre/tests/test_pre_aggregate.py
git commit -m "feat(pre): 两级聚合与 legacy/corrected 双输出，产物按 raw/analysis/tables 分层"
```

---

## Task 19: 分析层 6/6 —— 跨数据集 schema 校验 + 出图（P3）

**Files:**
- Create: `TPA/pre/analysis/schema.py`
- Create: `TPA/pre/analysis/plots.py`
- Test: `TPA/pre/tests/test_pre_schema.py`、`TPA/pre/tests/test_pre_plots.py`

**Interfaces:**
- Produces（schema）：`CROSS_DATASET_FORBIDDEN = ("delta_abs", "d_deg", "d_att",
  "d_ctrl_median", "d_ctrl_mean")`、`assert_cross_dataset_safe(columns)`、
  `cross_dataset_table(rows, columns)`（越界即 `ValueError`）
- Produces（plots）：`plot_absolute_deviation(rows, out_dir)`、
  `plot_normalized_recovery(rows, out_dir)`、`plot_force_structure(rows, out_dir)`、
  `plot_decomposition(rows, out_dir)`；每个产出 `*.pdf` + `*.svg`

- [ ] **Step 1: 写失败测试** —— `TPA/pre/tests/test_pre_schema.py`

```python
# -*- coding: utf-8 -*-
"""跨数据集表禁止出现绝对距离（spec §3.9，单元测试强制）。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

TPA_ROOT = Path(__file__).resolve().parents[2]
if str(TPA_ROOT) not in sys.path:
    sys.path.insert(0, str(TPA_ROOT))

from pre.analysis.schema import assert_cross_dataset_safe   # noqa: E402


class TestCrossDatasetSchema(unittest.TestCase):
    def test_relative_columns_pass(self):
        assert_cross_dataset_safe(["dataset", "attack", "ratio",
                                   "rr_pc_mean", "d_rel_mean", "cos_F_aud_mean"])

    def test_absolute_columns_raise(self):
        for bad in ("delta_abs_mean", "d_deg_mean", "d_ctrl_median_mean"):
            with self.assertRaises(ValueError):
                assert_cross_dataset_safe(["dataset", bad])

    def test_modified_key_is_detected(self):
        with self.assertRaises(ValueError):
            assert_cross_dataset_safe(["Dataset Absolute_Patch"])


if __name__ == "__main__":
    unittest.main()
```

（最后一个用例要求实现做**归一化比对**：小写 + 去掉前缀/后缀 `mean`/`sd`，见 Step 3。）

- [ ] **Step 2: 运行确认失败**
- [ ] **Step 3: 实现**

```python
# schema.py
CROSS_DATASET_FORBIDDEN = ("delta_abs", "d_deg", "d_att", "d_ctrl_median",
                           "d_ctrl_mean", "d_ctrl_q25", "d_ctrl_q75")


def _normalize(col: str) -> str:
    """列名归一化：小写、去空白、去掉聚合后缀（_mean/_sd）。"""
    c = str(col).strip().lower().replace(" ", "_")
    for suf in ("_mean", "_sd"):
        if c.endswith(suf):
            c = c[: -len(suf)]
    return c


def assert_cross_dataset_safe(columns) -> None:
    """跨数据集表只允许无量纲量（spec §3.9）。

    为什么用代码强制而不是写文档：绝对距离在不同数据集/模型间不可比
    （嵌入尺度、维度、训练过程都不同），而"跨数据集表"往往在出图前一秒才拼好，
    文档约束拦不住它。
    """
    bad = [c for c in columns if _normalize(c) in CROSS_DATASET_FORBIDDEN]
    if bad:
        raise ValueError(
            f"跨数据集表禁止出现绝对距离列：{bad}；"
            f"只允许 {('rr', 'rr_pc', 'd_rel', 'cos', '分解占比')} 等无量纲量")
```

```python
# plots.py 要点（统一样式 + 双格式输出）
def _save(fig, out_dir: Path, name: str) -> Dict[str, Path]:
    """同时存 PDF（论文/LaTeX）与 SVG（网页/人工检查）；PNG 仅调试，不作正式资产。"""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{name}.pdf"; svg = out_dir / f"{name}.svg"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return {"pdf": pdf, "svg": svg}


def plot_absolute_deviation(rows, out_dir, *, x="ratio", hue="attack"):
    """(a) 绝对伤害：d_deg 与 d_att 对比例；**必须**画 d_deg 基线并在图注说明
    "干净状态在该坐标系下天然为 0"（spec §3.3 的图示要求，仅同数据集内使用）。"""
```

- [ ] **Step 4: 运行确认通过**
- [ ] **Step 5: 提交**

Run: `G:\Idea\.venv\Scripts\python.exe -m unittest pre.tests.test_pre_schema pre.tests.test_pre_plots -v` → PASS

```bash
git add TPA/pre/analysis/schema.py TPA/pre/analysis/plots.py TPA/pre/tests/test_pre_schema.py TPA/pre/tests/test_pre_plots.py
git commit -m "feat(pre): 跨数据集 schema 校验与 PDF/SVG 出图模块"
```

---

## Task 20: 落《分析协议》文档（P3）

**Files:**
- Create: `TPA/pre/docs/ANALYSIS_PROTOCOL.md`
- Test: 无（文档），但必须与 Task 14–19 的代码常量逐条一致

- [ ] **Step 1: 写文档**，必须包含以下小节与**逐字**常量：

```markdown
# Batch A 分析协议（ANALYSIS_PROTOCOL）

版本：`batchA-2026-09-16`（与 `pre/protocol.py::PROTOCOL_VERSION` 一致）

## 1. 记号与参考帧
    z0 / z_deg / z_att；同 (dataset, model, seed) 同一帧；跨 seed 只聚合指标。

## 2. 对齐（gauge）
    主口径 Procrustes；拟合掩码**排除目标物品**；同一个 R 作用于物品行与用户行；
    对齐门槛 mean_row_cos_aligned >= 0.95，否则 unresolved_frame。

## 3. 距离与恢复量
    d_deg、d_att、Delta_abs（仅同数据集）、RR、E_deg/E_att、RR_pc、d_rel
    pc_valid 规则：thr = max(1e-9, 5 * MAD(d_ctrl))；pc_valid = E_deg > thr

## 4. 径向/切向
    ‖z−z0‖² = (r−‖z0‖)² + ‖t‖²；norm_recovery 仅作 legacy 字段

## 5. 力结构
    concentration = ‖mean(F)‖ / mean(‖f‖)；pairwise_cos；cos(F,z0)；
    cos(F,audience)（固定受众为主、剩余受众为辅）；
    声明"力"是训练后假用户位置，不是攻击者意图向量

## 6. 曝光
    target HR@K/NDCG@K，K=10 与 20，必须在模型自身坐标系内计算

## 7. 聚合与统计
    两级聚合；样本标准差 ddof=1；配对比较以 (seed,item,ratio) 为单位；
    符号一致率 = max(#正,#负)/N

## 8. 状态与覆盖率
    resolved / unresolved_frame / missing / invalid_numeric；
    shard 级 unresolved_frame 只告警，Gate 级必须为 0

## 9. 跨数据集
    禁止 Delta_abs / d_deg / d_ctrl；只允许 RR / RR_pc / d_rel / cos / 分解占比
    （由 pre/analysis/schema.py 强制）

## 10. 版本与变更
    任何改变本文件的修改都必须递增 PROTOCOL_VERSION 并回到 spec 评审
```

- [ ] **Step 2: 提交**

```bash
git add TPA/pre/docs/ANALYSIS_PROTOCOL.md
git commit -m "docs(pre): 落 Batch A 分析协议（数学口径与状态机）"
```
---

## Task 21: preflight（P4）

**Files:**
- Create: `TPA/pre/schedule/__init__.py`
- Create: `TPA/pre/schedule/plan.py`（本任务先建目录，内容见 Task 22）
- Modify: `TPA/pre/run.py`（`--mode doctor` 扩展为完整 preflight）
- Test: `TPA/pre/tests/test_pre_preflight.py`

**Interfaces:**
- Produces: `pre.runners.preflight.run_checks(cfg, datasets=None) -> Dict[str, Any]`；
  产出 `preflight.json`；任一红项 → `status="fail"`

- [ ] **Step 1: 写失败测试**（构造缺 meta / 缺 targets 的场景断言红项）

```python
# -*- coding: utf-8 -*-
"""preflight：缺目标集或 meta 指纹不一致必须报红（spec §5.2）。"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

TPA_ROOT = Path(__file__).resolve().parents[2]
if str(TPA_ROOT) not in sys.path:
    sys.path.insert(0, str(TPA_ROOT))

from pre.runners.preflight import check_targets                   # noqa: E402


class TestPreflight(unittest.TestCase):
    def test_missing_target_file_is_red(self):
        with tempfile.TemporaryDirectory() as d:
            item = check_targets("toy", targets_dir=Path(d))
            self.assertEqual(item["status"], "fail")
            self.assertIn("缺少冻结目标集", item["detail"])

    def test_present_target_file_passes_structure_check(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "toy.json"
            p.write_text(json.dumps({"dataset": "toy", "sha256_meta": "abc",
                                     "items": [{"item_id": 1, "N_i": 200, "q_i": 0.1,
                                                "interaction_percentile": 0.9,
                                                "clean_hr@10": 0.1, "clean_ndcg@10": 0.05}]}),
                         "utf-8")
            self.assertEqual(check_targets("toy", targets_dir=Path(d))["status"], "ok")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行确认失败**
- [ ] **Step 3: 实现 `pre/runners/preflight.py`**：6 组检查（环境 / 依赖 / 数据指纹 /
  目标集 / 攻击 callable / 磁盘与可写），每组返回
  `{"name": ..., "status": "ok|fail", "detail": ...}`；`run_checks` 汇总写
  `preflight.json` 并打印表格；`run.py --mode doctor` 调用它，**任一 fail 时退出码非 0**。
- [ ] **Step 4: 运行确认通过 + 真实 preflight**

Run: `G:\Idea\.venv\Scripts\python.exe -m unittest pre.tests.test_pre_preflight -v` → PASS
Run: `G:\Idea\.venv\Scripts\python.exe pre/run.py --mode doctor` →
Expected: 打印各项状态并写 `TPA/pre/preflight.json`；三数据集目标集齐全时全绿

- [ ] **Step 5: 提交**

```bash
git add TPA/pre/runners/preflight.py TPA/pre/run.py TPA/pre/schedule/__init__.py TPA/pre/tests/test_pre_preflight.py
git commit -m "feat(pre): preflight 自检（环境/依赖/数据指纹/目标集/攻击/磁盘）"
```

---

## Task 22: `schedule/plan.py` 分片计划器（P4）

**Files:**
- Create: `TPA/pre/schedule/plan.py`
- Test: `TPA/pre/tests/test_pre_schedule_plan.py`

**Interfaces:**
- Produces: `build_plan(cfg, *, batch, datasets, models, ratios, arms, seeds,
  items_per_dataset, cost_seconds) -> Dict[str, Any]`；
  `plan.json` 字段：`experimental_protocol_version / batch / shards[] / gpu_assignment /
  cost_model / created_at`；shard 字段：`run_tag / dataset / model / seed / item_id /
  ratios / arms / est_seconds`
- CLI：`python pre/schedule/plan.py --batch batchA --datasets ml100k,gowalla
  --items-per-dataset 5 --ratios 0.1,0.2,0.3,0.5,0.8,0.9 --arms
  degradation-only,random,bandwagon,pgd,uba --seeds 42,43,44`

- [ ] **Step 1: 写失败测试**：`run_tag` 命名、shard 数 = `datasets × models × seeds × items`、
  超过 `MAX_SHARD_SECONDS=24*3600` 时自动降级到 ratio 粒度（每 ratio 一个 shard）、
  LPT 排序（`est_seconds` 降序）。
- [ ] **Step 2: 运行确认失败** → **Step 3: 实现**（成本模型常量取自 spec §2.2 的实测表，
  4090 系数 `COST_SCALE_4090 = 5.0` 由 Gate 1 回填，先按 5.0 占位并在 plan.json 记录
  `"scale_source": "gate1-pending"`）→ **Step 4: 运行确认通过**。
- [ ] **Step 5: 提交**

```bash
git add TPA/pre/schedule/plan.py TPA/pre/tests/test_pre_schedule_plan.py
git commit -m "feat(pre): 分片计划器（LPT 排序、24h 自动降级、成本模型）"
```

---

## Task 23: `schedule/run.py` 调度器（P4）

**Files:**
- Create: `TPA/pre/schedule/run.py`
- Test: `TPA/pre/tests/test_pre_schedule_run.py`

**Interfaces:**
- Produces: `detect_gpus() -> List[Dict[str, Any]]`、`pick_gpus(devices, *, reserve_debug=1,
  main_cards=3) -> Dict[str, Any]`、`run_shard(shard, gpu, cfg, batch_dir) -> Dict[str, Any]`
- CLI：`python pre/schedule/run.py --batch batchA --plan outputs/batchA/plan.json
  [--dry-run] [--resume]`

- [ ] **Step 1: 写失败测试**（`pick_gpus` 纯逻辑：占用高的卡被排除、保留 1 张 debug、
  可用卡 < 2 时抛错；`run_shard` 用假 plan 断言命令里带 `CUDA_VISIBLE_DEVICES`）
- [ ] **Step 2-4: 实现并在本地用 `--dry-run` 验证一个 2-shard 计划**
- [ ] **Step 5: 提交**

关键实现点（spec §5.3/§5.5）：

```python
def detect_gpus() -> List[Dict[str, Any]]:
    """解析 nvidia-smi；不硬编码卡号（服务器状态会变）。"""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=20).stdout
    devices = []
    for line in out.strip().splitlines():
        idx, used, total, util = [x.strip() for x in line.split(",")]
        devices.append({"index": int(idx), "mem_used_mb": int(used),
                        "mem_total_mb": int(total), "util": int(util),
                        "usable": int(used) < 2048 and int(util) < 20})
    return devices
```

- 每个 shard 起一个子进程，`env["CUDA_VISIBLE_DEVICES"] = str(gpu)`；
- 完成的 shard 写 `<run_tag>.done`（内容含 returncode/seconds/run_tag）；
- 每 shard 一份 `logs/<run_tag>.log`；前台运行 + `tmux`（runbook 里给命令）；
- 定期打印"完成 X/Y、已耗 GPU-小时、预计剩余"；`--resume` 跳过已有 `.done`。

```bash
git add TPA/pre/schedule/run.py TPA/pre/tests/test_pre_schedule_run.py
git commit -m "feat(pre): 调度器（GPU 动态探测/绑定、.done 断点续跑、进度与日志）"
```

---

## Task 24: `schedule/verify.py` 完整性校验（P4）

**Files:**
- Create: `TPA/pre/schedule/verify.py`
- Test: `TPA/pre/tests/test_pre_schedule_verify.py`

**Interfaces:**
- Produces: `verify_batch(cfg, batch_dir, plan) -> Dict[str, Any]`；
  产出 `coverage.json` + `coverage.md`（batch 级）；
  退出码：`missing`/`invalid_numeric` → 非零；`unresolved_frame` → 0 但告警（shard 级语义）

- [ ] **Step 1: 写失败测试**（构造一个缺 shard 的 batch → 退出码非零；
  全部 resolved → 0，且 `coverage.json` 里 `coverage_ratio == 1.0`）
- [ ] **Step 2-4: 实现 + 本地跑通**：`python pre/schedule/verify.py --batch gate0a`
- [ ] **Step 5: 提交**

```bash
git add TPA/pre/schedule/verify.py TPA/pre/tests/test_pre_schedule_verify.py
git commit -m "feat(pre): batch 级完整性校验（覆盖率、四态、退出码语义）"
```

---

## Task 25: `SERVER_RUNBOOK.md` 完稿（P4）

**Files:** Modify: `TPA/pre/docs/SERVER_RUNBOOK.md`

- [ ] **Step 1: 补齐 6 个小节**：
  0 前置条件（preflight 全绿才继续）；1 独立 venv（`python -m venv .venv-server` +
  `pip install -r requirements.txt`，CUDA 12.8 驱动兼容 cu121 轮子）；
  2 数据与目标集检查（`sha256` 对 `pre/targets/*.json`）；
  3 分片与 GPU（`plan.py` + `nvidia-smi` 动态探测，禁止写死卡号）；
  4 启动/监控/续跑（`tmux new -s pre`、`run.py --resume`、`.done` 语义）；
  5 校验与 Gate 判据（`verify.py` + §5.7 表格 + **两层 unresolved_frame 语义**）；
  6 产物回传与白名单（`git add` 白名单路径，raw 不回传）。
- [ ] **Step 2: 提交**

```bash
git add TPA/pre/docs/SERVER_RUNBOOK.md
git commit -m "docs(pre): 完稿服务器运行手册（preflight/GPU/续跑/Gate/回传）"
```

---

## Task 26: Gate 0a（本地快速验收，P5）

**Files:** 无代码改动（只跑与记录）

- [ ] **Step 1: 跑本地快速矩阵**（ml100k，1 item × 3 ratio × 5 arms × 2 seed）

Run（工作目录 `G:\Idea\TPA`）：
`G:\Idea\.venv\Scripts\python.exe pre/run.py --mode all --tag gate0a --models lightgcn --ratios 0.1,0.5,0.9 --attacks random,bandwagon,pgd,uba --limit-items 1 --seeds 42,43`
Expected: 30 个条件全部产出；`coverage.json` 全 resolved；预计约 1 小时（3050）

- [ ] **Step 2: L1–L3 验收**（spec §4.6）

- L1：`targets.json` 与旧 `run-k5v2/targets.json` 的 5 个 item_id 一致；clean 模型产物
  在重新训练前提下 metadata 关键字段（num_users/num_items/train_pairs）一致；
- L2：`manifest.json` 14 字段齐全；`manifest_history.jsonl` 有记录；`deleted_interactions.json`
  在两个 seed 下逐字节相同；
- L3：写 `tmp/gate0a_diff.md`，逐项列出与旧口径的数值差异**及其对应的修复编号**
  （d_deg 变化 → 修复 5；norm_recovery → 修复 7；`estimated_value` → 修复 2）。

- [ ] **Step 3: 失败处理**：任何 L1/L2 失败 → 直接回 P1–P3 修，**不进入 Gate 0b**。
- [ ] **Step 4: 记录**：把 Gate 0a 结论追加到 `TPA/pre/docs/SERVER_RUNBOOK.md` 的
  "Gate 记录"小节（日期、commit、结论、异常）。
- [ ] **Step 5: 提交**

```bash
git add TPA/pre/docs/SERVER_RUNBOOK.md
git commit -m "docs(pre): 记录 Gate 0a 本地快速验收结果"
```

---

## Task 27: Gate 0b（本地全量主轴，P5）

**Files:** 无代码改动

- [ ] **Step 1: 跑本地全量主轴**（ml100k，5 × 6 × 5 × 3）

Run: `G:\Idea\.venv\Scripts\python.exe pre/run.py --mode all --tag gate0b --models lightgcn --attacks random,bandwagon,pgd,uba --seeds 42,43,44`
Expected: 450 个条件；coverage 100%；本地约 16 小时（可分批 `--item-ids` 推进）

- [ ] **Step 2: 验收**：L1–L3 + coverage 100% + `invalid_numeric = 0`。
- [ ] **Step 3: 生成第一版论文图表**（PDF/SVG）确认分析管线端到端可用。
- [ ] **Step 4: 提交**（只提交文档与图表；raw 不入库）

```bash
git add TPA/pre/docs/SERVER_RUNBOOK.md TPA/pre/outputs/gate0b/tables TPA/pre/outputs/gate0b/figures TPA/pre/outputs/gate0b/analysis TPA/pre/outputs/gate0b/coverage.json TPA/pre/outputs/gate0b/coverage.md
git commit -m "docs(pre): 记录 Gate 0b 全量主轴结果与首版图表"
```

---

## Task 28: Gate 1（服务器 preflight + dry-run，P6）

**Files:** 无代码改动（服务器侧执行）

- [ ] **Step 1: 服务器环境**：clone → 独立 venv → `pip install -r requirements.txt` →
  `python pre/run.py --mode doctor`（preflight 必须全绿）
- [ ] **Step 2: 数据检查**：`models/lightgcn/data/processed/<ds>/meta.pkl` 的 sha256 与
  `pre/targets/<ds>.json` 的 `sha256_meta` 一致（三数据集逐个查）
- [ ] **Step 3: dry-run**：三数据集各 `1 item × 1 ratio × {random, uba, tpa}`

Run: `python pre/schedule/plan.py --batch batchA-gate1 --datasets ml100k,gowalla,amazon-book --items-per-dataset 1 --ratios 0.5 --arms random,uba,tpa --seeds 42`
Run: `python pre/schedule/run.py --batch batchA-gate1`
Run: `python pre/schedule/verify.py --batch batchA-gate1`

Expected: preflight 全绿、verify 全绿、9 个 shard 全部 `.done`；
advinject 另有 import/registry smoke test（`python -c "from attacks.batch import registry;
print(registry.registered_names())"`）

- [ ] **Step 4: 回填成本模型**：用实测 epoch 秒数更新 `plan.py` 的
  `COST_SCALE_4090` 与各数据集 epoch 时间，并把 `plan.json` 的
  `scale_source` 改为 `gate1-measured`（这是**唯一**允许修改成本常量的时点）。
- [ ] **Step 5: 提交**

```bash
git add TPA/pre/schedule/plan.py TPA/pre/docs/SERVER_RUNBOOK.md
git commit -m "chore(pre): 回填 Gate 1 实测成本模型系数"
```

---

## Task 29: Gate 2（主实验 + S1–S4 + Rule A/B，P7）

**Files:**
- Create: `TPA/pre/schedule/gate_rules.py`
- Test: `TPA/pre/tests/test_pre_gate_rules.py`

**Interfaces:**
- Produces: `rule_a(tables) -> Dict[str, Any]`、`rule_b1/b2/b3/b4(...) -> Dict[str, Any]`、
  `evaluate_gates(tables) -> Dict[str, Any]`

- [ ] **Step 1: 写失败测试**（合成表：Rule A 同号 → pass；异号 → fail；
  Rule B3 配对差值符号一致率 2/3 的边界；Rule B4 coverage=0.99 → fail）
- [ ] **Step 2-4: 实现并跑通单测**
- [ ] **Step 5: 服务器执行 Gate 2**

```bash
python pre/schedule/plan.py --batch batchA-main --datasets ml100k,gowalla --arms degradation-only,random,bandwagon,pgd,uba --items-per-dataset 5 --seeds 42,43,44
python pre/schedule/run.py  --batch batchA-main
python pre/schedule/verify.py --batch batchA-main
python pre/schedule/gate_rules.py --batch batchA-main   # 打印 Rule A/B1-B4 判定
```

- [ ] **Step 6: 补充实验 S1–S4**（S4 零成本必做；S1/S2/S3 用独立 batch tag）
- [ ] **Step 7: 提交**

```bash
git add TPA/pre/schedule/gate_rules.py TPA/pre/tests/test_pre_gate_rules.py TPA/pre/docs/SERVER_RUNBOOK.md
git commit -m "feat(pre): Gate 2 预注册判据实现与主实验结果记录"
```

---

## Task 30: Gate 3（amazon-book 档位，P8）

**Files:** 无代码改动

- [ ] **Step 1: 按 Rule 决策**：Gate 2 通过 → 默认进 **B 档**（4 ratio × 3 seed）；
  仅当出现明确 feasibility 问题（单 shard 超 24h 且降级后仍超、或显存不足）才进 C 档。
- [ ] **Step 2: 执行**

```bash
python pre/schedule/plan.py --batch batchA-amazon --datasets amazon-book --arms degradation-only,random,bandwagon,pgd,uba --items-per-dataset 5 --ratios 0.1,0.3,0.5,0.9 --seeds 42,43,44
python pre/schedule/run.py  --batch batchA-amazon
python pre/schedule/verify.py --batch batchA-amazon
```

- [ ] **Step 3: 记录档位决策与理由**到 `SERVER_RUNBOOK.md`。
- [ ] **Step 4: 提交**

```bash
git add TPA/pre/docs/SERVER_RUNBOOK.md
git commit -m "docs(pre): 记录 Gate 3 amazon-book 档位决策与执行结果"
```

---

## Task 31: 论文资产与白名单回传（P9）

**Files:**
- Modify: `.gitignore`
- Create: `TPA/pre/analysis/cross_dataset.py`（跨数据集汇总表，**只允许无量纲列**）

- [ ] **Step 1: 改 `.gitignore`**

```gitignore
# 论文资产白名单（Batch A）：raw/checkpoint/log 不入库，可重生成的最终资产入库
**/outputs/**
!**/outputs/*/
!**/outputs/*/analysis/
!**/outputs/*/analysis/**
!**/outputs/*/tables/
!**/outputs/*/tables/**
!**/outputs/*/figures/
!**/outputs/*/figures/**
!**/outputs/*/plan.json
!**/outputs/*/coverage.json
!**/outputs/*/coverage.md
```

（删除原来的 `outputs/` 行：目录被 ignore 后，`!` 无法再包含其内部文件。
必须用 `**/outputs/**` + 逐级重新包含目录的写法。）

- [ ] **Step 2: 实证校验白名单**

Run:
`git check-ignore -v TPA/pre/outputs/gate0b/raw/original/lightgcn/seed_42/clean/model.pt`
Expected: 命中 `**/outputs/**`（仍被忽略）

Run: `git check-ignore -v TPA/pre/outputs/gate0b/analysis/corrected/results.csv`
Expected: 无输出（未被忽略）；`git status --porcelain` 能看到该文件

- [ ] **Step 3: 生成跨数据集汇总与最终图**（只读 `corrected`），跑
  `assert_cross_dataset_safe` 通过后写 `tables/cross_dataset.csv` 与 `figures/*.pdf|svg`。
- [ ] **Step 4: 提交**

```bash
git add .gitignore TPA/pre/analysis/cross_dataset.py
git add TPA/pre/outputs/batchA-main/tables TPA/pre/outputs/batchA-main/figures TPA/pre/outputs/batchA-main/analysis TPA/pre/outputs/batchA-main/coverage.json TPA/pre/outputs/batchA-main/coverage.md
git commit -m "feat(pre): 跨数据集汇总资产与 outputs 白名单回传"
```

---

## Task 32: Batch B（Gate 1 后启动，P10）

**Files:**
- Create: `docs/superpowers/specs/2026-09-16-batchB-attack-porting-design.md`
- Create: `TPA/attacks/{legup,trialattack,sui}/...`（按 AGENTS §6.1 模板）

**红线（spec §6.2）**：Batch B 以**独立目录 / 分支式新增**方式进行，**不得修改已冻结的
Batch A 实验协议、数据口径、主轴配置与分析接口**；若确实需要改动，必须回到 spec 评审。

- [ ] **Step 1（Gate 1 通过后）写 Batch B spec**：三个攻击的机制、官方代码位置
  （Leg-UP/AUSH+：`tmp/shillingattack/ShillingAttack-master/Leg-UP/`，仓库
  `XMUDM/ShillingAttack`；TrialAttack：`Daftstone/TrialAttack`；SUI-Attack：
  `KDEGroup/SUI-Attack`）、配置预留块（AGENTS §6.6 显式评分槽位）、sanity check 口径。
- [ ] **Step 2**：按 AGENTS §6.1 建目录与固定入口（config/registry/classify/generate/fit/
  evaluate/run + docs），逐模块 TDD。
- [ ] **Step 3**：官方设定 sanity check（ml100k 上验证 AUSH+ 相对 Random 的增益方向与
  量级对得上）；未通过只能进附录。
- [ ] **Step 4（Gate 3 通过后）入矩阵**：以独立 batch tag 运行，**不并入主轴表格**。
- [ ] **Step 5: 提交**

```bash
git add docs/superpowers/specs/2026-09-16-batchB-attack-porting-design.md TPA/attacks/legup TPA/attacks/trialattack TPA/attacks/sui
git commit -m "feat(attacks): Batch B 三个近年攻击移植与独立矩阵"
```

---

## Task 33: 交付前全量回归

- [ ] `G:\Idea\.venv\Scripts\python.exe -m unittest discover -s tests -t . -v`（工作目录
  `G:\Idea\TPA`）全绿，与 Task 0 基线对比
- [ ] `G:\Idea\.venv\Scripts\python.exe -m unittest discover -s pre/tests -t . -v` 全绿
- [ ] `git status` / `git diff --stat` 自查：只提交与任务相关文件；`outputs/` 只含白名单资产
- [ ] 文档同步：`pre/README.md`、`pre/docs/EXTENDING.md`、`SERVER_RUNBOOK.md`、
  `ANALYSIS_PROTOCOL.md` 与实现一致

---

## 计划自查（self-review 记录）

**1. Spec 覆盖检查**（逐节 → 任务）

| spec 章节 | 落到任务 |
|---|---|
| §0.4 Protocol Freeze | Global Constraints + §6.6 引用 + Task 31 白名单冻结 |
| §1.1–§1.2 数据集与衰减定义 | Task 1（目标集）、Task 3（deletion_seed）、Task 9（shared/fingerprint） |
| §1.3 Target qualification | Task 1 |
| §1.4 Feasibility pilot | Task 26/27（只验收可执行性，不筛目标） |
| §1.5 双归一化预算 | Task 3（`rho_A/rho_P/rho_E` 写入 config + manifest） |
| §1.6–§1.7 Seed 与删除规则 | Task 3、Task 2 |
| §1.8 shared + fingerprint | Task 9 |
| §2.1–§2.4 矩阵与档位 | Task 22（档位即 plan 参数）、Task 27/29/30 |
| §2.5 分片 | Task 22、Task 23 |
| §2.6 目标集冻结 | Task 1 |
| §2.7 provenance | Task 0（字段表）、Task 13（写入） |
| §3.1–§3.4 帧/对齐/placebo/三件套 | Task 5、Task 14、Task 15 |
| §3.5 径向切向 | Task 7 |
| §3.6 力结构 | Task 16 |
| §3.7 曝光验证 | Task 17 |
| §3.8 聚合与四态 | Task 6、Task 18 |
| §3.9 跨数据集 schema | Task 19 |
| §3.10 模块划分与数据流 | Task 18（目录）、Task 14–19（模块） |
| §4.1–§4.5 十二项修复 | Task 2/3/4/5/6/7/9/10/11/12/13/8 |
| §4.6 L1–L4 与 legacy/corrected | Task 18、Task 26、Task 27 |
| §5.1–§5.6 服务器执行 | Task 21/22/23/24/25 |
| §5.7 Gate 与 Rule | Task 26–30 |
| §5.8 本地/服务器一致性 | Task 28 |
| §5.9 产物回传 | Task 31 |
| §6.1–§6.5 验收与落文件 | Task 25–33 |
| §6.2 Batch B 红线 | Task 32 |

**2. 占位符扫描**：无 TBD/TODO；每个代码步骤都给了可直接粘贴的实现或明确的函数清单。

**3. 类型/命名一致性**：`cond_dir`/`original_dir`/`degraded_model_dir`/`attack_dir`
（Task 3）在 Task 6（`classify_state`）、Task 18（布局）中被一致复用；
`meta_fingerprint` 在 Task 4 与 Task 9 中同名同算法；`exposure_from_scores`
（Task 1）被 Task 17 复用；`PROTOCOL_FIELDS`（Task 0）被 Task 13 复用。

**4. 已知未决项（不阻塞，按 spec 留白）**：

- `COST_SCALE_4090` 的最终值由 Gate 1 实测回填（Task 28），在此之前 plan 只用于相对排序；
- `pc_threshold` 的 `sigma=5.0` 是初始值，Gate 0a/0b 后若出现 `pc_valid=False` 的条件
  过多/过少，需回到 spec 评审调整（属"分析主口径"变更）；
- `pre/analysis/legacy_export.py` 只在需要把旧 `tmp/uba_ablation` 结论迁到新目录时使用，
  不影响主链路。
