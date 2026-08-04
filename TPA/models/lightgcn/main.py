r"""LightGCN 训练入口
用法（在任意目录下执行）:
  G:\Idea\.venv\Scripts\python.exe G:\Idea\TPA\models\lightgcn\main.py
  G:\Idea\.venv\Scripts\python.exe G:\Idea\TPA\models\lightgcn\main.py --resume
"""
import sys
import os

# 确保项目根目录在 sys.path 中（无论从哪里启动）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

if __name__ == "__main__":
    from models.lightgcn.train import main
    main()
