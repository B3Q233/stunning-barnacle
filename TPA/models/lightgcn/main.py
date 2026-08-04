"""LightGCN 训练入口 — 极薄入口，实际逻辑在 train.py 中"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from models.lightgcn.train import main

if __name__ == "__main__":
    main()
