"""环节计时工具单测。"""
import io
import unittest
from contextlib import redirect_stdout

from training.timing import (
    SectionTimer, format_duration, section_enter, section_exit, timed)


class FormatDurationTest(unittest.TestCase):

    def test_format(self):
        self.assertEqual(format_duration(0), "0分0.0秒")
        self.assertEqual(format_duration(65.2), "1分5.2秒")
        self.assertEqual(format_duration(3600), "60分0.0秒")


class SectionTest(unittest.TestCase):

    def test_context_manager_prints_start_end(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            with SectionTimer("数据注入"):
                pass
        text = buf.getvalue()
        self.assertIn("【数据注入开始】", text)
        self.assertIn("[数据注入结束 耗时", text)

    def test_enter_exit_helpers(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            t0 = section_enter("中毒训练")
            section_exit("中毒训练", t0)
        text = buf.getvalue()
        self.assertIn("【中毒训练开始】", text)
        self.assertIn("[中毒训练结束 耗时", text)

    def test_timed_decorator_preserves_return(self):
        buf = io.StringIO()

        @timed("推荐频次分类")
        def f(x):
            return x * 2

        with redirect_stdout(buf):
            self.assertEqual(f(21), 42)
        self.assertIn("【推荐频次分类开始】", buf.getvalue())
        self.assertIn("[推荐频次分类结束 耗时", buf.getvalue())
