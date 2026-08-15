r"""WMF 训练入口
用法（在 TPA 目录下）:
  ..\.venv\Scripts\python.exe models\wmf\main.py
  ..\.venv\Scripts\python.exe models\wmf\main.py --resume
  ..\.venv\Scripts\python.exe models\wmf\main.py --tag 2026-08-15-01 --epochs 5
"""
import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


if __name__ == "__main__":
    from models.wmf.train import main
    parser = argparse.ArgumentParser(description="WMF 训练入口")
    parser.add_argument("--config", type=str, default=None,
                        help="config.yaml 路径（缺省=模型目录下 config.yaml）")
    parser.add_argument("--resume", action="store_true", help="断点续训")
    parser.add_argument("--tag", type=str, default=None,
                        help="实验标签 run_tag（缺省=当前时间）")
    parser.add_argument("--epochs", type=int, default=None,
                        help="覆盖 config.training.epochs（冒烟测试用）")
    args = parser.parse_args()
    main(tag=args.tag, resume=args.resume, config_path=args.config,
         epochs_override=args.epochs)
