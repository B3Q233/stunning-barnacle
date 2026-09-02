"""配置 canonical 化与 schema 校验测试（unittest，仅标准库）。"""
from __future__ import annotations

import copy
import tempfile
import unittest
import warnings
from pathlib import Path


def _canonicalize(cfg):
    from training.config_utils import canonicalize_config

    return canonicalize_config(cfg)


def _schema():
    from training.config_utils import schema_leaf_paths

    return schema_leaf_paths()


class LegacyMappingTest(unittest.TestCase):
    """旧别名 → canonical（rename / restructure / conversion）。"""

    def test_dataset_variants_move_to_top(self):
        out = _canonicalize({
            "data": {"dataset": "ml100k"},
            "experiment": {"seed": 7},
        })
        self.assertEqual(out["dataset"], "ml100k")
        self.assertEqual(out["seed"], 7)
        self.assertNotIn("data", out)
        self.assertNotIn("experiment", out)

    def test_use_cuda_becomes_device(self):
        out = _canonicalize({"use_cuda": False})
        self.assertEqual(out["training"]["device"], "cpu")
        self.assertNotIn("use_cuda", out)
        out2 = _canonicalize({"use_cuda": True})
        self.assertEqual(out2["training"]["device"], "cuda")

    def test_eval_every_moves_into_training(self):
        out = _canonicalize({"evaluation": {"eval_every": 5, "k": 20}})
        self.assertEqual(out["training"]["eval_every"], 5)
        self.assertNotIn("eval_every", out["evaluation"])
        self.assertEqual(out["k"], 20)
        self.assertNotIn("k", out["evaluation"])

    def test_output_dir_maps_to_output_dir(self):
        out = _canonicalize({"output_dir": "outputs/x"})
        self.assertEqual(out["output"]["dir"], "outputs/x")
        self.assertNotIn("output_dir", out)

    def test_lambda_reg_maps_to_weight_decay(self):
        out = _canonicalize({"training": {"lambda_reg": 0.01}})
        self.assertEqual(out["training"]["weight_decay"], 0.01)
        self.assertNotIn("lambda_reg", out["training"])

    def test_checkpoint_aliases(self):
        out = _canonicalize({
            "clean_checkpoint": "a.pt",
            "classification": {"checkpoint": "a.pt"},
            "warm_start": {"enabled": True},
        })
        self.assertEqual(out["checkpoint"]["clean"], "a.pt")
        self.assertNotIn("clean_checkpoint", out)
        self.assertNotIn("checkpoint", out["classification"])

    def test_surrogate_flat_keys_restructured(self):
        out = _canonicalize({
            "surrogate": {
                "model_name": "itemae",
                "epochs": 50,
                "lr": 0.001,
                "l2": 1e-6,
                "batch_size": 2048,
                "weight_alpha": 20.0,
                "unroll_steps": 5,
            },
            "attack": {"unroll_steps": 3},
        })
        sur = out["surrogate"]
        self.assertEqual(sur["name"], "itemae")
        tr = sur["training"]
        self.assertEqual(tr["epochs"], 50)
        self.assertEqual(tr["lr"], 0.001)
        self.assertEqual(tr["weight_decay"], 1e-6)
        self.assertEqual(tr["batch_size"], 2048)
        self.assertEqual(tr["weight_alpha"], 20.0)
        self.assertEqual(tr["unroll_steps"], 5)
        self.assertNotIn("model_name", sur)
        self.assertNotIn("epochs", sur)
        self.assertNotIn("unroll_steps", sur)

    def test_adv_flattened_keys_restructured(self):
        out = _canonicalize({
            "attack": {
                "attack_type": "adversarial",
                "adv_epochs": 30,
                "adv_lr": 1.0,
                "adv_momentum": 0.95,
                "proj_threshold": 0.1,
                "click_targets": True,
            },
        })
        adv = out["attack"]["adv"]
        self.assertEqual(adv["epochs"], 30)
        self.assertEqual(adv["lr"], 1.0)
        self.assertEqual(adv["momentum"], 0.95)
        self.assertEqual(adv["proj_threshold"], 0.1)
        self.assertTrue(adv["click_targets"])
        self.assertEqual(adv["attack_type"], "adversarial")

    def test_n_fakes_value_branch(self):
        out = _canonicalize({"attack": {"n_fakes": 0.01}})
        self.assertEqual(out["attack"]["ratio"], 0.01)
        out2 = _canonicalize({"attack": {"n_fakes": 50}})
        self.assertEqual(out2["attack"]["num_fake_users"], 50)

    def test_percentile_and_zone_renames_only(self):
        out = _canonicalize({
            "classification": {
                "popular_percentile": 95,
                "torso_percentile": 75,
                "tail_percentile": 50,
            },
            "attack": {
                "n_target_items": 5,
                "target_item_popularity": "upper_torso",
                "target_items": [3, 9],
            },
        })
        cls = out["classification"]
        self.assertEqual(cls["percentile_head"], 95)
        self.assertEqual(cls["percentile_upper_torso"], 75)
        self.assertEqual(cls["percentile_lower_torso"], 50)
        ti = out["attack"]["target_items"]
        self.assertEqual(ti["count"], 5)
        self.assertEqual(ti["zone"], "upper_torso")
        self.assertEqual(ti["ids"], [3, 9])
        self.assertEqual(ti["strategy"], "specified")


class PrecedenceAndConflictTest(unittest.TestCase):
    """canonical 优先；多 legacy 冲突抛错。"""

    def test_canonical_wins_over_legacy(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = _canonicalize({
                "dataset": "yelp2018",
                "data": {"dataset": "ml100k"},
                "training": {"device": "cuda:1"},
                "use_cuda": False,
            })
        self.assertEqual(out["dataset"], "yelp2018")
        self.assertEqual(out["training"]["device"], "cuda:1")
        self.assertTrue(any("忽略旧键" in str(w.message) for w in caught))

    def test_conflicting_legacy_raises(self):
        with self.assertRaises(ValueError):
            _canonicalize({
                "clean_checkpoint": "a.pt",
                "classification": {"checkpoint": "b.pt"},
            })

    def test_input_not_mutated(self):
        cfg = {"data": {"dataset": "ml100k"}, "use_cuda": True}
        _canonicalize(cfg)
        self.assertEqual(cfg, {"data": {"dataset": "ml100k"},
                               "use_cuda": True})


class SchemaTest(unittest.TestCase):
    """合法叶子路径 + 语义唯一性。"""

    def test_schema_contains_canonical_paths(self):
        leaves = _schema()
        for path in ("dataset", "training.device", "training.eval_every",
                     "checkpoint.clean", "attack.target_items.zone",
                     "surrogate.training.weight_decay",
                     "surrogate.training.unroll_steps",
                     "classification.percentile_head"):
            self.assertIn(path, leaves)

    def test_schema_rejects_deprecated_flat_paths(self):
        leaves = _schema()
        for path in ("surrogate.weight_decay", "surrogate.unroll_steps",
                     "training.k", "surrogate.epochs"):
            self.assertNotIn(path, leaves)

    def test_load_config_roundtrip(self):
        from training.config_utils import load_config

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.yaml"
            path.write_text(
                "data:\n  dataset: ml100k\nuse_cuda: true\n",
                encoding="utf-8",
            )
            out = load_config(path)
        self.assertEqual(out["dataset"], "ml100k")
        self.assertEqual(out["training"]["device"], "cuda")


if __name__ == "__main__":
    unittest.main()
