"""MF 数据预处理脚本

与 LightGCN 的 preprocess.py 同一套逻辑（NGCF 多物品行 / 单对行均可解析），
把 data/raw/{dataset}/ 下的 train.txt / test.txt 转为 BPR 训练所需的
models/mf/data/processed/{dataset}/meta.pkl。
"""
import os
import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]  # TPA 项目根
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUT_DIR = PROJECT_ROOT / "models" / "mf" / "data" / "processed"


def parse_pairs_file(filepath):
    """解析 (user, item) 对：支持 NGCF 多物品行（user item1 item2 ...）
    与单对行（user item）两种格式。"""
    pairs = []
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            user = int(parts[0])
            for item_str in parts[1:]:
                pairs.append((user, int(item_str)))
    return pairs


def save_pairs(pairs, outpath):
    with open(outpath, "w") as f:
        for u, i in pairs:
            f.write(f"{u} {i}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["ml100k", "gowalla", "yelp2018", "amazon-book"])
    parser.add_argument("--raw_dir", type=str, default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    raw_dir = os.path.join(args.raw_dir, args.dataset)
    out_dir = os.path.join(args.out_dir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    train_pairs = parse_pairs_file(os.path.join(raw_dir, "train.txt"))
    test_pairs = parse_pairs_file(os.path.join(raw_dir, "test.txt"))

    all_users = set(p[0] for p in train_pairs) | set(p[0] for p in test_pairs)
    all_items = set(p[1] for p in train_pairs) | set(p[1] for p in test_pairs)

    print(f"=== {args.dataset} 预处理 ===")
    print(f"用户数: {len(all_users)}")
    print(f"物品数: {max(all_items) + 1}")
    print(f"训练交互: {len(train_pairs)}")
    print(f"测试交互: {len(test_pairs)}")

    save_pairs(train_pairs, os.path.join(out_dir, "train_pairs.txt"))
    save_pairs(test_pairs, os.path.join(out_dir, "test_pairs.txt"))

    user_items = {}
    for u, i in train_pairs:
        user_items.setdefault(u, set()).add(i)

    import pickle
    meta = {
        "num_users": len(all_users),
        "num_items": max(all_items) + 1,
        "train_pairs": train_pairs,
        "test_pairs": test_pairs,
        "user_items": user_items,
    }
    with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    print(f"预处理完成 -> {out_dir}/")
    print(f"  产出: train_pairs.txt ({len(train_pairs)} 行)")
    print(f"        test_pairs.txt ({len(test_pairs)} 行)")
    print(f"        meta.pkl")


if __name__ == "__main__":
    main()
