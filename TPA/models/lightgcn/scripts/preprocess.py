"""LightGCN 数据预处理脚本
将 NGCF 格式 (user_id item1 item2 ...) 的 train/test 数据转换为 BPR 训练所需的格式。
- train：每行一对 (user_id, item_id) 的正样本对
- 同时生成邻接矩阵所需的边列表供后续 Dataset 使用
"""
import os
import argparse

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # TPA 项目根
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUT_DIR = PROJECT_ROOT / "models" / "lightgcn" / "data" / "processed"


def parse_ngcf_file(filepath):
    """解析 NGCF 格式文件: user_id item1 item2 ..."""
    pairs = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            user = int(parts[0])
            for item_str in parts[1:]:
                pairs.append((user, int(item_str)))
    return pairs

def save_pairs(pairs, outpath):
    """保存为 (user, item) 对的文本文件"""
    with open(outpath, 'w') as f:
        for u, i in pairs:
            f.write(f"{u} {i}\n")

def get_dataset_stats(raw_dir):
    """读取原始数据统计信息"""
    train_path = os.path.join(raw_dir, 'train.txt')
    test_path = os.path.join(raw_dir, 'test.txt')

    train_pairs = parse_ngcf_file(train_path)
    test_pairs = parse_ngcf_file(test_path)

    all_users = set(p[0] for p in train_pairs) | set(p[0] for p in test_pairs)
    all_items = set(p[1] for p in train_pairs) | set(p[1] for p in test_pairs)

    return {
        'num_users': len(all_users),
        'num_items': max(all_items) + 1,  # 假设 ID 从 0 开始连续
        'num_train': len(train_pairs),
        'num_test': len(test_pairs),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['gowalla', 'yelp2018', 'amazon-book'])
    parser.add_argument('--raw_dir', type=str, default=str(DEFAULT_RAW_DIR))
    parser.add_argument('--out_dir', type=str, default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    raw_dir = os.path.join(args.raw_dir, args.dataset)
    out_dir = os.path.join(args.out_dir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    # 解析原始文件
    train_pairs = parse_ngcf_file(os.path.join(raw_dir, 'train.txt'))
    test_pairs = parse_ngcf_file(os.path.join(raw_dir, 'test.txt'))

    # 统计信息
    all_users = set(p[0] for p in train_pairs) | set(p[0] for p in test_pairs)
    all_items = set(p[1] for p in train_pairs) | set(p[1] for p in test_pairs)

    print(f"=== {args.dataset} 预处理 ===")
    print(f"用户数: {len(all_users)}")
    print(f"物品数: {max(all_items) + 1}")
    print(f"训练交互: {len(train_pairs)}")
    print(f"测试交互: {len(test_pairs)}")

    # 保存为 (user, item) 对的格式
    save_pairs(train_pairs, os.path.join(out_dir, 'train_pairs.txt'))
    save_pairs(test_pairs, os.path.join(out_dir, 'test_pairs.txt'))

    # 生成交互字典 (用户 -> 物品集合) 用于负采样
    user_items = {}
    for u, i in train_pairs:
        if u not in user_items:
            user_items[u] = set()
        user_items[u].add(i)

    # 保存为 pickle 供 dataset 快速加载
    import pickle
    meta = {
        'num_users': len(all_users),
        'num_items': max(all_items) + 1,
        'train_pairs': train_pairs,
        'test_pairs': test_pairs,
        'user_items': user_items,
    }
    with open(os.path.join(out_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    print(f"预处理完成 → {out_dir}/")
    print(f"  产出: train_pairs.txt ({len(train_pairs)} 行)")
    print(f"        test_pairs.txt ({len(test_pairs)} 行)")
    print(f"        meta.pkl")

if __name__ == '__main__':
    main()
