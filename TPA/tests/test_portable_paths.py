"""跨平台路径回归测试：仓库代码/配置中不得硬编码 Windows 盘符绝对路径。

背景：此前 dataset.py / preprocess.py 硬编码 ``g:/Idea/TPA/...``，攻击 config
硬编码 ``G:\\Idea\\TPA\\...``；在 Linux 上运行时会在 CWD 下创建 ``g:/Idea/...``
目录、或找不到数据/checkpoint。本测试锁定所有数据与 checkpoint 路径都基于
TPA 项目根做相对解析。
"""
import builtins
import io
import unittest
from pathlib import Path
from unittest import mock

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]  # TPA 项目根


def _fake_meta():
    return {
        "num_users": 4,
        "num_items": 5,
        "train_pairs": [(0, 1), (1, 2)],
        "test_pairs": [(0, 2)],
        "user_items": {0: {1}, 1: {2}},
    }


class DatasetMetaPathTest(unittest.TestCase):
    """数据集加载器读取的 meta.pkl 必须位于模块推导出的 TPA 根目录下。"""

    def _captured_open(self, module, loader_cls, dataset):
        opened = {}

        def fake_open(path, *args, **kwargs):
            opened["path"] = str(path)
            return io.BytesIO(b"x")

        def fake_pickle_load(f):
            return _fake_meta()

        with mock.patch.object(builtins, "open", fake_open), \
                mock.patch.object(module.pickle, "load", fake_pickle_load):
            loader_cls({"dataset": dataset})
        return opened["path"]

    def test_mf_meta_path_derived_from_project_root(self):
        import models.mf.dataset as ds

        path = self._captured_open(ds, ds.MFDataLoader, "ml100k")
        expected = (
            ds.PROJECT_ROOT / "models" / "mf" / "data" / "processed"
            / "ml100k" / "meta.pkl"
        )
        self.assertEqual(Path(path), expected)
        self.assertNotIn("g:/idea", path.lower())

    def test_lightgcn_meta_path_derived_from_project_root(self):
        import models.lightgcn.dataset as ds

        path = self._captured_open(ds, ds.LightGCNDataLoader, "gowalla")
        expected = (
            ds.PROJECT_ROOT / "models" / "lightgcn" / "data" / "processed"
            / "gowalla" / "meta.pkl"
        )
        self.assertEqual(Path(path), expected)
        self.assertNotIn("g:/idea", path.lower())


class PreprocessDefaultPathTest(unittest.TestCase):
    """preprocess 默认输入/输出目录必须基于 TPA 根，禁止盘符硬编码。"""

    def _assert_defaults(self, module, model_dir):
        self.assertEqual(module.DEFAULT_RAW_DIR, PROJECT_ROOT / "data" / "raw")
        self.assertEqual(
            module.DEFAULT_OUT_DIR,
            PROJECT_ROOT / "models" / model_dir / "data" / "processed",
        )
        for p in (module.DEFAULT_RAW_DIR, module.DEFAULT_OUT_DIR):
            self.assertTrue(p.is_absolute(), f"默认目录应为绝对路径: {p}")
            self.assertNotIn("g:/idea", str(p).lower())

    def test_mf_preprocess_defaults(self):
        import models.mf.scripts.preprocess as pre

        self._assert_defaults(pre, "mf")

    def test_lightgcn_preprocess_defaults(self):
        import models.lightgcn.scripts.preprocess as pre

        self._assert_defaults(pre, "lightgcn")


class AttackConfigPathTest(unittest.TestCase):
    """攻击 config 中的 checkpoint 一律使用相对 TPA 根的路径。"""

    ATTACK_DIRS = ["tpa", "bandwagon", "pgd"]

    @staticmethod
    def _checkpoint_entries(cfg):
        entries = []
        cls = cfg.get("classification") or {}
        if cls.get("checkpoint"):
            entries.append(("classification.checkpoint", cls["checkpoint"]))
        if cfg.get("clean_checkpoint"):
            entries.append(("clean_checkpoint", cfg["clean_checkpoint"]))
        ws = cfg.get("warm_start") or {}
        if ws.get("checkpoint"):
            entries.append(("warm_start.checkpoint", ws["checkpoint"]))
        sur = cfg.get("surrogate") or {}
        if sur.get("checkpoint"):
            entries.append(("surrogate.checkpoint", sur["checkpoint"]))
        return entries

    def test_checkpoint_values_are_relative(self):
        for name in self.ATTACK_DIRS:
            cfg_path = PROJECT_ROOT / "attacks" / name / "config.yaml"
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            entries = self._checkpoint_entries(cfg)
            self.assertTrue(entries, f"{name}: 未发现任何 checkpoint 配置")
            for label, value in entries:
                self.assertFalse(
                    Path(value).is_absolute(),
                    f"{name}: {label} 使用了绝对路径: {value}",
                )
                self.assertNotRegex(
                    value, r"^[A-Za-z]:[\\/]",
                    f"{name}: {label} 含盘符: {value}",
                )
                self.assertNotIn("g:/idea", value.lower())


class ResolveFromRootTest(unittest.TestCase):
    """checkpoint 相对路径统一解析到项目根，绝对路径保持原样。"""

    def test_relative_path_joined_to_root(self):
        from training.paths import resolve_from_root

        root = PROJECT_ROOT
        rel = Path("models/lightgcn/outputs/latest.pt")
        self.assertEqual(resolve_from_root(rel, root), root / rel)

    def test_absolute_path_kept_unchanged(self):
        from training.paths import resolve_from_root

        root = PROJECT_ROOT
        absolute = root / "some" / "checkpoint.pt"
        self.assertEqual(resolve_from_root(absolute, root), absolute)


if __name__ == "__main__":
    unittest.main()
