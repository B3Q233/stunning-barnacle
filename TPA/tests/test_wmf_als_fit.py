"""WMF 中毒数据 ALS 训练（攻击 fit 阶段分支）单元测试"""
import json
import tempfile
import unittest
from pathlib import Path

from models.wmf.model import WMFModel
from models.wmf.train import train_wmf_from_meta
from training.framework import TrainingConfig


METRICS = [{"recall@5": "upper"}, {"ndcg@5": "upper"}]
ATTACK_METRICS = [
    {"target_ndcg@5": "upper"},
    {"target_hr@5": "upper"},
    {"recall@5": "upper"},
    {"ndcg@5": "upper"},
]


def make_poisoned_meta():
    """3 个真实用户 + 2 个注入假用户（uid 3/4），8 个物品。"""
    train = [
        (0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6),
        (3, 1), (3, 2), (4, 7), (4, 0),   # 假用户注入的交互
    ]
    test = [(0, 1), (1, 3), (2, 5)]
    user_items = {}
    for u, i in train:
        user_items.setdefault(u, set()).add(i)
    return {
        "num_users": 5,
        "num_items": 8,
        "train_pairs": train,
        "test_pairs": test,
        "user_items": user_items,
    }


def make_cfg(epochs=2):
    return TrainingConfig(overrides={
        "factors": 4,
        "alpha": 40.0,
        "epsilon": 1e-8,
        "confidence_scheme": "minimal",
        "lambda_reg": 0.01,
        "epochs": epochs,
        "eval_every": 1,
        "k": 5,
        "device": "cpu",
        "metrics": METRICS,
    })


class WMFAlsFitTest(unittest.TestCase):
    """中毒数据 = 普通数据：直接 ALS 全量训练 + 攻击评估回调。"""

    def test_train_on_poisoned_meta(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            model, history = train_wmf_from_meta(
                make_cfg(), make_poisoned_meta(), out, WMFModel,
                metrics_cfg=METRICS, checkpoint_mode="per_metric",
            )
            self.assertEqual(model.num_users, 5)  # 模型含注入假用户
            self.assertEqual(len(history), 2)
            entry = history[0]
            self.assertIn("train_loss", entry)
            self.assertIn("val_loss", entry)
            self.assertIn("recall@5", entry)
            self.assertTrue((out / "checkpoints" / "latest.pt").exists())
            hist = json.loads(
                (out / "history.json").read_text(encoding="utf-8"))
            self.assertIn("best", hist)
            self.assertIn("recall@5", hist["best"])

    def test_warm_start_skipped_for_wmf(self):
        """WMF 无 embedding 属性：warm_start=True 不崩溃，回退随机初始化。"""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            model, history = train_wmf_from_meta(
                make_cfg(epochs=1), make_poisoned_meta(), out, WMFModel,
                metrics_cfg=METRICS, warm_start=True,
            )
            self.assertEqual(len(history), 1)

    def test_targets_evaluated(self):
        """目标物品指标进入 history（攻击效果口径）。"""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            model, history = train_wmf_from_meta(
                make_cfg(epochs=1), make_poisoned_meta(), out, WMFModel,
                metrics_cfg=ATTACK_METRICS, targets=[7],
            )
            self.assertIn("targets", history[0])
            self.assertIn(7, history[0]["targets"])


class AttackFitDispatchTest(unittest.TestCase):
    """攻击 fit 模板对 WMF 模型自动走 ALS 分支。"""

    def test_random_fit_dispatches_wmf(self):
        from attacks.random.fit import train_poisoned_model
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            model, history = train_poisoned_model(
                make_cfg(epochs=1), make_poisoned_meta(), out,
                warm_start=True, warm_ckpt=None, clean_num_users=3,
                model_cls=WMFModel, dataset_cls=None,
                metrics_cfg=METRICS, targets=[7],
            )
            self.assertEqual(len(history), 1)
            self.assertIn("recall@5", history[0])
            self.assertTrue((out / "history.json").exists())


if __name__ == "__main__":
    unittest.main()
