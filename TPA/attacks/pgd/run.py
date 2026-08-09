"""PGD 攻击编排入口

攻击流程（与确认过的口径一致）：
  1. classify: 加载干净模型 → 全量评分 → 每用户 Top-K → 统计推荐频次
                → 划分 流行(前20%) / 普通 / 冷门，缓存供后续使用
  2. data:     指定目标物品 → PGD 投影梯度上升生成假用户画像 → 注入中毒数据
  3. model:    选择模型（config model.name: mf / lightgcn）→ 投毒训练 → 对比评估

用法:
  python attacks/pgd/run.py --mode classify  # 第 1 步：推荐频次分类
  python attacks/pgd/run.py --mode data      # 第 2 步：PGD 生成中毒数据
  python attacks/pgd/run.py --mode model     # 第 3 步：拟合中毒模型 + 评估
  python attacks/pgd/run.py --mode both      # data + model（默认）
  python attacks/pgd/run.py --mode all       # classify + data + model 全流程

模块分工：
- classify 模式只调用 classify.py（依赖模型 checkpoint，产出频次分类缓存）
- data 模式调用 generate.py（PGD 需要模型权重计算梯度，因此会加载干净模型；
  与 bandwagon 的"纯数据层"不同——这是 PGD 攻击的本质要求）
- model 模式才调用 fit.py（新建受害模型实例并在中毒数据上训练）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]  # G:\Idea\TPA
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from attacks.pgd.generate import load_yaml_config, main as gen_main
from attacks.pgd.fit import main as fit_main
from attacks.pgd.classify import main as classify_main


def main() -> None:
    parser = argparse.ArgumentParser(description="PGD 攻击编排")
    parser.add_argument(
        "--config", type=str,
        default=str(PROJECT_ROOT / "attacks" / "pgd" / "config.yaml"),
    )
    parser.add_argument(
        "--mode", type=str, default=None,
        choices=["classify", "data", "model", "both", "all"],
        help="classify=推荐频次分类；data=PGD 生成中毒数据；"
             "model=拟合中毒模型；both=data+model；all=全流程（默认按配置）",
    )
    parser.add_argument(
        "--tag", type=str, default=None,
        help="实验标签 run_tag（优先于 config.run_tag；缺省=当前时间）",
    )
    args = parser.parse_args()

    cfg = load_yaml_config(Path(args.config))
    if args.tag:
        cfg["run_tag"] = args.tag
    mode = args.mode or cfg.get("mode", "both")
    from training.run_tag import resolve_run_tag
    print(f"[run] mode={mode}, dataset={cfg.get('dataset')}, "
          f"model={cfg.get('model', {}).get('name')}, "
          f"run_tag={resolve_run_tag(cfg)}")

    from training.modes import stages_for_mode
    run_classify, run_data, run_model = stages_for_mode(mode)
    if run_classify:
        classify_main(cfg)
    if run_data:
        gen_main(cfg)
    if run_model:
        fit_main(cfg)


if __name__ == "__main__":
    main()
