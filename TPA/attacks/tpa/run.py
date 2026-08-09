"""TPA（传递式路径投毒攻击）编排入口

攻击流程（已冻结：无 PGD、最短路径法、数据集不变）：
  1. classify: 加载干净模型 → 推荐频次分类（流行/普通/冷门），供目标选择与统计
  2. paths:    构建物品共现图 → CF 距离最短路径 → 平庸基座 + 路径 + 目标画像
  3. data:     读取路径画像缓存 → 注入中毒数据
  4. model:    选择模型（config model.name）→ 投毒训练 → 对比评估

用法:
  python attacks/tpa/run.py --mode classify  # 第 1 步：推荐频次分类
  python attacks/tpa/run.py --mode paths     # 第 2 步：共现图 + 路径画像构造
  python attacks/tpa/run.py --mode data      # 第 3 步：只生成中毒数据
  python attacks/tpa/run.py --mode model     # 第 4 步：拟合中毒模型 + 评估
  python attacks/tpa/run.py --mode both      # data + model（默认）
  python attacks/tpa/run.py --mode all       # classify + paths + data + model 全流程

模块分离：
- classify / paths 模式依赖模型 checkpoint（只读，产出缓存）
- data 模式是纯数据层（只读路径缓存，不 import 模型代码）
- model 模式才新建受害模型实例并训练
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]  # G:\Idea\TPA
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from attacks.tpa.generate import load_yaml_config, main as gen_main
from attacks.tpa.fit import main as fit_main
from attacks.tpa.classify import main as classify_main
from attacks.tpa.path_builder import main as paths_main


def main() -> None:
    parser = argparse.ArgumentParser(description="TPA 编排")
    parser.add_argument(
        "--config", type=str,
        default=str(PROJECT_ROOT / "attacks" / "tpa" / "config.yaml"),
    )
    parser.add_argument(
        "--mode", type=str, default=None,
        choices=["classify", "paths", "data", "model", "both", "all"],
        help="classify=推荐频次分类；paths=共现图+路径画像；data=只生成中毒数据；"
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
          f"run_tag={resolve_run_tag(cfg)}")

    from training.modes import stages_for_mode
    run_classify, run_data, run_model = stages_for_mode(mode)
    if run_classify:
        classify_main(cfg)
    if mode in ("paths", "all"):
        paths_main(cfg)
    if run_data:
        gen_main(cfg)
    if run_model:
        fit_main(cfg)


if __name__ == "__main__":
    main()
