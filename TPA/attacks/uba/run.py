"""UBA（Uplift-guided Budget Allocation）攻击编排入口

攻击流程（论文 4.1 的两个估计支路 + Algorithm 1 + 实例化）：
  1. classify: 统计训练集交互数 → 划分流行/普通/冷门（目标选择与 filler 池用）
  2. estimate: 处理效应 Y（w/ S_φ 支路：代理模型模拟实验；path 支路可跳过）
  3. data:     读 Y → 分组背包 DP 分配 T* → 按画像规则实例化假档案 → 注入中毒数据
  4. model:    选择模型（config model.name）→ warm-start 投毒训练 → 对比评估

用法:
  python attacks/uba/run.py --mode classify   # 第 1 步：交互数分类
  python attacks/uba/run.py --mode estimate   # 第 2 步：代理模型处理效应估计（仅 surrogate 支路需要）
  python attacks/uba/run.py --mode data       # 第 3 步：分配 + 生成中毒数据（path 支路自动补算并缓存 Y）
  python attacks/uba/run.py --mode model      # 第 4 步：拟合中毒模型 + 评估
  python attacks/uba/run.py --mode both       # data + model（默认）
  python attacks/uba/run.py --mode all        # classify + estimate + data + model 全流程

模块分离（与仓库硬模板一致，estimate 是 UBA 自定义阶段）：
- classify 是纯数据层；
- estimate 依赖代理模型（只读训练数据 + surrogate 配置，不改 victim）；
- data 在 method=path 时是纯数据层（不 import 模型代码）；method=surrogate 且缓存
  缺失时会惰性调用 estimate 补算；
- model 才新建受害模型实例并训练。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]  # TPA
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from attacks.uba.generate import load_yaml_config, main as gen_main
from attacks.uba.fit import main as fit_main
from attacks.uba.classify import main as classify_main
from attacks.uba.estimate import main as estimate_main


VALID_MODES = ("classify", "estimate", "data", "model", "both", "all")


def main() -> None:
    parser = argparse.ArgumentParser(description="UBA 攻击编排")
    parser.add_argument(
        "--config", type=str,
        default=str(PROJECT_ROOT / "attacks" / "uba" / "config.yaml"),
    )
    parser.add_argument(
        "--mode", type=str, default=None, choices=list(VALID_MODES),
        help="classify=交互数分类；estimate=代理模型处理效应估计；"
             "data=分配+中毒数据；model=拟合中毒模型；"
             "both=data+model；all=全流程（默认按配置）",
    )
    parser.add_argument("--tag", type=str, default=None,
                        help="实验标签 run_tag（优先于 config.run_tag；缺省=当前时间）")
    args = parser.parse_args()

    cfg = load_yaml_config(Path(args.config))
    if args.tag:
        cfg["run_tag"] = args.tag
    mode = args.mode or cfg.get("mode", "both")
    if mode not in VALID_MODES:
        raise ValueError(f"未知 mode {mode!r}，可选 {VALID_MODES}")
    from training.run_tag import resolve_run_tag

    print(f"[run] mode={mode}, dataset={cfg.get('dataset')}, "
          f"run_tag={resolve_run_tag(cfg)}")

    # 三阶段语义复用 training.modes（classify/data/model），estimate 是 UBA 自有阶段：
    #   all = classify + estimate + data + model；both = data + model（与其它攻击一致）；
    #   estimate 单独运行时只跑处理效应估计（不触发 data/model）
    from training.modes import stages_for_mode

    if mode == "estimate":
        run_classify = run_data = run_model = False
    else:
        run_classify, run_data, run_model = stages_for_mode(mode)
    if run_classify:
        classify_main(cfg)
    if mode in ("estimate", "all"):
        estimate_main(cfg)
    if run_data:
        gen_main(cfg)
    if run_model:
        fit_main(cfg)


if __name__ == "__main__":
    main()
