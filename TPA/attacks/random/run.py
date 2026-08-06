"""攻击模板编排入口

攻击流程（与确认后的口径一致）：
  1. classify: 加载干净模型 → 全量评分 → 每用户 Top-K → 统计推荐频次
               → 划分 流行(前20%) / 普通 / 冷门，缓存供后续使用
  2. data:     指定目标物品 → 生成假用户/假交互 → 注入中毒数据
  3. model:    选择模型（config model.name）→ 投毒训练 → 对比评估

用法:
  python attacks/random/run.py --mode classify  # 第 1 步：推荐频次分类
  python attacks/random/run.py --mode data      # 第 2 步：只生成中毒数据
  python attacks/random/run.py --mode model     # 第 3 步：拟合中毒模型 + 评估
  python attacks/random/run.py --mode both      # data + model（默认）
  python attacks/random/run.py --mode all       # classify + data + model 全流程

模块分离：
- classify 模式只调用 classify.py（依赖模型 checkpoint，产出频次分类缓存）
- data  模式只调用 generate.py（纯数据层，只读分类缓存，不 import 模型代码）
- model 模式才调用 fit.py（新建受害模型实例并在中毒数据上训练）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]  # G:\Idea\TPA
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from attacks.random.generate import load_yaml_config, main as gen_main
from attacks.random.fit import main as fit_main
from attacks.random.classify import main as classify_main


def main() -> None:
    parser = argparse.ArgumentParser(description="攻击模板编排")
    parser.add_argument(
        "--config", type=str,
        default=str(PROJECT_ROOT / "attacks" / "random" / "config.yaml"),
    )
    parser.add_argument(
        "--mode", type=str, default=None,
        choices=["classify", "data", "model", "both", "all"],
        help="classify=推荐频次分类；data=只生成中毒数据；"
             "model=拟合中毒模型；both=data+model；all=全流程（默认按配置）",
    )
    args = parser.parse_args()

    cfg = load_yaml_config(Path(args.config))
    mode = args.mode or cfg.get("mode", "both")
    print(f"[run] mode={mode}, dataset={cfg.get('dataset')}")

    if mode in ("classify", "all"):
        classify_main(cfg)
    if mode in ("data", "both"):
        gen_main(cfg)
    if mode in ("model", "both"):
        fit_main(cfg)


if __name__ == "__main__":
    main()
