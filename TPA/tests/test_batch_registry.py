"""Attack Registry 插件注册单测（CPU）。"""
import unittest

from attacks.batch import registry


class RegistryTest(unittest.TestCase):

    def test_builtin_attacks_registered(self):
        names = registry.registered_names()
        for name in ("bandwagon", "random", "pgd", "tpa"):
            self.assertIn(name, names)

    def test_get_returns_spec(self):
        spec = registry.get("bandwagon")
        self.assertEqual(spec.name, "bandwagon")
        self.assertTrue(spec.config_path.endswith(
            "attacks/bandwagon/config.yaml"))
        self.assertTrue(callable(spec.classify))
        self.assertTrue(callable(spec.generate))
        self.assertTrue(callable(spec.fit))

    def test_unknown_raises(self):
        with self.assertRaises(KeyError):
            registry.get("not_exist")

    def test_duplicate_register_raises(self):
        with self.assertRaises(ValueError):
            registry.register("bandwagon", "x.yaml",
                              lambda: None, lambda: None, lambda: None)
