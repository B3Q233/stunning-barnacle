"""epoch 输出组件测试（行格式 / history schema / csv）。"""
from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from training.epoch_log import (
    log_eval_line,
    log_train_line,
    write_eval_log,
    write_history,
)


class EpochLogLineTest(unittest.TestCase):

    def test_train_line_format(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            log_train_line(1, 30, 0.4452, val_loss=0.3821)
        self.assertIn("[epoch 1/30]", buf.getvalue())
        self.assertIn("train_loss=0.4452", buf.getvalue())
        self.assertIn("val_loss=0.3821", buf.getvalue())

    def test_train_line_without_val(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            log_train_line(2, 5, 0.1234)
        self.assertNotIn("val_loss", buf.getvalue())

    def test_eval_line_prints_all_metrics(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            log_eval_line({"recall@10": 0.1588, "fake_metric@5": 1.0,
                           "target_avg_rank": 222.9})
        text = buf.getvalue()
        self.assertIn("[eval]", text)
        self.assertIn("recall@10=0.1588", text)
        self.assertIn("fake_metric@5=1.0000", text)
        self.assertIn("target_avg_rank=222.9000", text)


class EpochHistoryWriteTest(unittest.TestCase):

    def test_history_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            history = [{
                "epoch": 1,
                "epoch_seconds": 1.5,
                "train_loss": 0.4,
                "val_loss": 0.3,
                "recall@10": 0.1,
            }]
            best = {"recall@10": {"epoch": 1, "value": 0.1}}
            write_history(out, history, best)
            data = json.loads((out / "history.json").read_text(
                encoding="utf-8"))
        self.assertEqual(list(data.keys()), ["history", "best"])
        self.assertEqual(data["history"][0]["epoch_seconds"], 1.5)

    def test_eval_log_csv_dynamic_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            write_eval_log(
                out,
                rows=[{"epoch": 1, "recall@10": 0.1, "fake@5": 2.0}],
                metric_names=["recall@10", "fake@5"],
            )
            with open(out / "eval_log.csv", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)
                row = next(reader)
        self.assertEqual(header, ["epoch", "recall@10", "fake@5"])
        self.assertEqual(row[0], "1")


if __name__ == "__main__":
    unittest.main()
