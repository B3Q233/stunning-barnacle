"""训练 history 完整性回归测试（unittest，无第三方新依赖）。

背景：历史文件里评估指标（recall@K / ndcg@K / target_*）只在部分 epoch
出现（攻击侧默认 eval_every=5；模型侧只写入 eval_log.csv、不写进 history.json），
导致可视化时每轮数据缺失。本测试锁定修复后的行为：
- 攻击配置默认每轮全量评估（eval_every=1），保证每轮都有评估指标；
- 模型评估回调返回本轮评估结果；
- 模型训练循环把回调结果合并进当前轮的 history 条目。
"""
import ast
import tempfile
import unittest
from pathlib import Path

import torch
import yaml

from models.mf.train import FullRankingCallback


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ATTACK_CONFIGS = [
    PROJECT_ROOT / "attacks" / "tpa" / "config.yaml",
    PROJECT_ROOT / "attacks" / "pgd" / "config.yaml",
    PROJECT_ROOT / "attacks" / "bandwagon" / "config.yaml",
    PROJECT_ROOT / "attacks" / "random" / "config.yaml",
]
TEMPLATE_CONFIG = (
    PROJECT_ROOT.parent / ".codex" / "skills" / "paper-code-implementation"
    / "assets" / "attack-imp-direct-poison" / "config.yaml"
)
MODEL_TRAIN_PY = {
    "lightgcn": PROJECT_ROOT / "models" / "lightgcn" / "train.py",
    "mf": PROJECT_ROOT / "models" / "mf" / "train.py",
}


def load_config_eval_every(path: Path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)["training"]["eval_every"]


def _for_loop_merges_eval_into_history(train_py: Path) -> bool:
    """训练 for 循环里：history.append(entry) 且 entry.update(评估回调返回值)。"""
    tree = ast.parse(train_py.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue
        appended: set[str] = set()
        eval_result_names: set[str] = set()
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
                and sub.func.attr == "append"
                and isinstance(sub.func.value, ast.Name)
                and sub.func.value.id == "history"
                and sub.args
                and isinstance(sub.args[0], ast.Name)
            ):
                appended.add(sub.args[0].id)
            if (
                isinstance(sub, ast.Assign)
                and isinstance(sub.value, ast.Call)
                and isinstance(sub.value.func, ast.Attribute)
                and sub.value.func.attr == "on_epoch_end"
            ):
                for target in sub.targets:
                    if isinstance(target, ast.Name):
                        eval_result_names.add(target.id)
        if appended and eval_result_names:
            for sub in ast.walk(node):
                if (
                    isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Attribute)
                    and sub.func.attr == "update"
                    and isinstance(sub.func.value, ast.Name)
                    and sub.func.value.id in appended
                    and sub.args
                    and isinstance(sub.args[0], ast.Name)
                    and sub.args[0].id in eval_result_names
                ):
                    return True
    return False


class AttackConfigEvalEveryTest(unittest.TestCase):
    def test_attack_configs_evaluate_every_epoch(self):
        for path in ATTACK_CONFIGS:
            with self.subTest(config=path.name):
                self.assertEqual(
                    load_config_eval_every(path), 1,
                    f"{path} 的 training.eval_every 应为 1，"
                    "保证每个 epoch 的 history 都含评估指标",
                )

    def test_template_config_evaluates_every_epoch(self):
        if not TEMPLATE_CONFIG.exists():
            self.skipTest("攻击技能模板不在本地")
        self.assertEqual(load_config_eval_every(TEMPLATE_CONFIG), 1)


class _FakeOptimizer:
    def state_dict(self):
        return {}


class _FakeModel:
    def __init__(self, n_users, n_items, dim=4):
        self.n_users = n_users
        self.emb = torch.randn(n_users + n_items, dim)
        self._optimizer = _FakeOptimizer()

    def set_eval(self):
        pass

    def get_user_embeddings(self):
        return self.emb[: self.n_users]

    def get_item_embeddings(self):
        return self.emb[self.n_users :]

    def state_dict(self):
        return {"emb": self.emb}


class _FakeLoader:
    def __init__(self, test_pairs, user_items):
        self.test_pairs = test_pairs
        self.user_items = user_items


def _build_callback(tag_dir, eval_every):
    loader = _FakeLoader(
        test_pairs=[(0, 1), (1, 2), (2, 3)],
        user_items={0: {0}, 1: {1}, 2: {2}},
    )
    model = _FakeModel(n_users=3, n_items=5)
    config = {
        "eval_every": eval_every,
        "k": 2,
        "metrics": [{"recall@2": "upper"}, {"ndcg@2": "upper"}],
        "checkpoint_mode": "per_metric",
    }
    return FullRankingCallback(loader, model, config, str(tag_dir))


class FullRankingCallbackTest(unittest.TestCase):
    def test_on_epoch_end_returns_eval_metrics_dict(self):
        with tempfile.TemporaryDirectory() as td:
            cb = _build_callback(Path(td), eval_every=1)
            out = cb.on_epoch_end(1, {})
        self.assertIsInstance(out, dict)
        self.assertIn("recall@2", out)
        self.assertIn("ndcg@2", out)
        self.assertGreaterEqual(out["recall@2"], 0.0)
        self.assertLessEqual(out["recall@2"], 1.0)

    def test_skipped_epoch_returns_none(self):
        with tempfile.TemporaryDirectory() as td:
            cb = _build_callback(Path(td), eval_every=2)
            out = cb.on_epoch_end(3, {})
        self.assertIsNone(out)


class ModelHistoryCompletenessTest(unittest.TestCase):
    def test_model_train_loop_merges_eval_into_history_entry(self):
        for name, path in MODEL_TRAIN_PY.items():
            with self.subTest(model=name):
                self.assertTrue(
                    _for_loop_merges_eval_into_history(path),
                    f"{path} 的训练循环应把评估回调返回值合并进每轮 history 条目",
                )


if __name__ == "__main__":
    unittest.main()
