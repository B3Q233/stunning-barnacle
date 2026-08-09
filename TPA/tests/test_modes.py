"""run.py 三阶段编排模式单元测试"""
import unittest

from training.modes import stages_for_mode


class StagesForModeTest(unittest.TestCase):
    def test_single_modes(self):
        self.assertEqual(stages_for_mode("classify"), (True, False, False))
        self.assertEqual(stages_for_mode("data"), (False, True, False))
        self.assertEqual(stages_for_mode("model"), (False, False, True))

    def test_both(self):
        self.assertEqual(stages_for_mode("both"), (False, True, True))

    def test_all_runs_all_stages(self):
        """--mode all 必须依次执行 classify + data + model。"""
        self.assertEqual(stages_for_mode("all"), (True, True, True))

    def test_invalid_mode(self):
        with self.assertRaises(ValueError):
            stages_for_mode("unknown")


if __name__ == "__main__":
    unittest.main()
