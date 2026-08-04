"""LightGCN 训练入口
  python models/lightgcn/main.py              # 新训练
  python models/lightgcn/main.py --resume      # 断点续训
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

if __name__ == "__main__":
    from models.lightgcn.train import main
    main()
