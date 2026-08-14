r"""LightGCN 训练入口
用法（在仓库根下执行）:
  python TPA/models/lightgcn/main.py
  python TPA/models/lightgcn/main.py --resume
"""
import sys
import os

# 确保项目根目录在 sys.path 中（无论从哪里启动）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

if __name__ == "__main__":
    from models.lightgcn.train import main
    import argparse
    parser = argparse.ArgumentParser(description="LightGCN 训练入口")
    parser.add_argument("--resume", action="store_true", help="断点续训")
    parser.add_argument("--tag", type=str, default=None,
                        help="实验标签 run_tag（缺省=当前时间）")
    args = parser.parse_args()
    main(tag=args.tag, resume=args.resume)
