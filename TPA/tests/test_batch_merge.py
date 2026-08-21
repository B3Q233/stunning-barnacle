"""Deep Merge 配置继承单测（CPU）。"""
import unittest

from attacks.batch.utils import deep_merge


class DeepMergeTest(unittest.TestCase):

    def test_nested_dict_merge(self):
        out = deep_merge(
            {"attack": {"ratio": 0.03, "filler_size": 20},
             "training": {"epochs": 30}},
            {"training": {"epochs": 10}})
        self.assertEqual(out["training"]["epochs"], 10)
        self.assertEqual(out["attack"]["ratio"], 0.03)

    def test_scalar_and_list_overwrite(self):
        out = deep_merge({"a": 1, "b": [1, 2], "c": {"x": 1}},
                         {"b": [3], "c": 5})
        self.assertEqual(out, {"a": 1, "b": [3], "c": 5})

    def test_inputs_not_mutated(self):
        base = {"a": {"x": 1}}
        deep_merge(base, {"a": {"y": 2}})
        self.assertEqual(base, {"a": {"x": 1}})
