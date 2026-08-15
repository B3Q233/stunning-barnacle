"""WMF 数据处理（步骤①）单元测试（unittest，仅标准库）"""
import os
import pickle
import tempfile
import unittest
from pathlib import Path

from models.wmf.scripts.preprocess import (
    build_meta,
    parse_pairs_file,
    run_preprocess,
)


class ParsePairsFileTest(unittest.TestCase):
    """成对格式与 NGCF 多物品行格式的解析。"""

    def test_single_pair_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.txt"
            path.write_text("0 1\n0 2\n3 4\n", encoding="utf-8")
            self.assertEqual(
                parse_pairs_file(path), [(0, 1), (0, 2), (3, 4)]
            )

    def test_multi_item_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.txt"
            path.write_text("0 1 2 3\n1 4\n", encoding="utf-8")
            self.assertEqual(
                parse_pairs_file(path),
                [(0, 1), (0, 2), (0, 3), (1, 4)],
            )

    def test_blank_lines_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.txt"
            path.write_text("0 1\n\n1 2\n", encoding="utf-8")
            self.assertEqual(parse_pairs_file(path), [(0, 1), (1, 2)])


class BuildMetaTest(unittest.TestCase):
    """id 重映射为 0..n-1 连续空间，保证与 num_users/num_items 一致。"""

    def test_remap_contiguous(self):
        train = [(5, 100), (5, 2)]
        test = [(7, 2)]
        meta = build_meta(train, test)
        # 用户排序 [5,7] -> 5->0, 7->1；物品排序 [2,100] -> 2->0, 100->1
        self.assertEqual(meta["num_users"], 2)
        self.assertEqual(meta["num_items"], 2)
        self.assertEqual(meta["train_pairs"], [(0, 1), (0, 0)])
        self.assertEqual(meta["test_pairs"], [(1, 0)])
        self.assertEqual(meta["user_items"], {0: {0, 1}})

    def test_already_contiguous_unchanged(self):
        train = [(0, 0), (1, 1)]
        meta = build_meta(train, [])
        self.assertEqual(meta["num_users"], 2)
        self.assertEqual(meta["num_items"], 2)
        self.assertEqual(meta["train_pairs"], [(0, 0), (1, 1)])


class RunPreprocessTest(unittest.TestCase):
    """端到端：原始成对文件 -> meta.pkl + train/test_pairs.txt。"""

    def test_full_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            out_dir = root / "processed"
            (raw_dir / "ml100k").mkdir(parents=True)
            (raw_dir / "ml100k" / "train.txt").write_text(
                "0 1\n0 2\n1 3\n", encoding="utf-8"
            )
            (raw_dir / "ml100k" / "test.txt").write_text(
                "0 1\n2 4\n", encoding="utf-8"
            )

            stats = run_preprocess(raw_dir, out_dir, dataset="ml100k")

            self.assertEqual(stats["num_users"], 3)
            self.assertEqual(stats["num_items"], 4)
            self.assertEqual(stats["train_pairs"], 3)
            self.assertEqual(stats["test_pairs"], 2)

            meta_path = out_dir / "ml100k" / "meta.pkl"
            self.assertTrue(meta_path.exists())
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            self.assertEqual(meta["num_users"], 3)
            self.assertEqual(meta["num_items"], 4)
            self.assertEqual(len(meta["train_pairs"]), 3)
            self.assertEqual(len(meta["test_pairs"]), 2)

            train_txt = (out_dir / "ml100k" / "train_pairs.txt").read_text(
                encoding="utf-8"
            ).strip().splitlines()
            self.assertEqual(len(train_txt), 3)
            self.assertIn("0 0", train_txt)  # 原始 (0,1) 重映射为 (0,0)


if __name__ == "__main__":
    unittest.main()
