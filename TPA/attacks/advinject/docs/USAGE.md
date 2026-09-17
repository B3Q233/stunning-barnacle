# AdvInject 使用说明

## 复现范围

本模块实现论文中的代理模型投毒闭环：选择目标物品、初始化伪用户交互、在 ItemAE surrogate 上优化伪用户数据、保存投毒数据，并在 victim model 上重新训练评估。

## 运行

```powershell
cd TPA
<repo>\.venv\Scripts\python.exe -m attacks.advinject.run --config attacks/advinject/config.yaml --mode all
```

建议先使用 `--mode data` 检查目标和伪用户数据，再执行完整流程。结果按 `outputs/{dataset}/{victim}/{run_tag}/` 隔离。

## 与上游差异

当前实现不新增依赖、不创建隔离环境；代理优化使用 PyTorch 可微目标作为最小可运行实现。上游仓库的 `higher` 有限 unroll、完整 victim trainer 对齐和 Gowalla 多模型数值复现仍需在实际数据与版本条件下单独核验。
