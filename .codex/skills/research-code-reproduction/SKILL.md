---
name: research-code-reproduction
description: Use when analyzing a local, GitHub, or archived research code repository to produce a Chinese reproduction guide covering environment, datasets, weights, configuration, training, evaluation, inference, and troubleshooting without modifying the target repository.
---

# Research Code Reproduction

只读分析科研开源代码仓库，输出从环境准备到实验复现的证据化中文指南。

## 硬性边界

- 目标仓库只读：不创建、修改、删除或覆盖其中任何文件。
- 不安装依赖、不创建环境、不下载数据或权重。
- 不执行训练、推理、批量评估或可能写入 checkpoint、日志、缓存的命令。
- 只运行轻量、可逆、只读检查；无法安全验证时标记限制。

## 输入

接受本地仓库路径、GitHub URL 或仓库压缩包路径。可选输入包括论文、指定实验、目标设备、已有环境、权重路径和输出偏好。先确认输入根目录；压缩包只在用户允许且使用隔离临时目录时读取。

## 分析顺序

1. 读取 `workflow/repository.md`，确认版本、结构和入口。
2. 读取 `workflow/environment.md`，提取依赖和环境要求。
3. 读取 `workflow/dataset.md`，追踪数据加载、预处理和路径。
4. 读取 `workflow/weights.md`，追踪 checkpoint 加载和权重位置。
5. 读取 `workflow/config.md`，识别配置、参数覆盖和输出。
6. 读取 `workflow/training.md`，静态追踪训练入口和保存逻辑。
7. 读取 `workflow/inference.md`，静态追踪测试、评估和推理。
8. 将论文实验与代码入口、配置、指标建立映射。
9. 读取 `workflow/troubleshooting.md`，生成有证据的排错步骤。
10. 使用 `templates/usage-guide.md` 组装最终指南。

## 证据与不确定性

读取 `rules/source-of-truth.md`、`rules/command-validation.md` 和 `rules/uncertainty.md`。冲突按以下顺序裁决：

```text
源代码事实 > 配置文件 > 可执行命令验证 > README > 论文 > 推断
```

每个关键结论引用文件路径、函数/类、配置键或命令，并注明状态：源码确认、配置确认、文档声明、命令验证、合理推断或无法确认。禁止把推断写成确定事实。

## 固定输出

最终必须包含：项目简介、项目结构、环境准备、数据集准备、预训练模型与权重、配置说明、训练、测试与评估、推理、论文实验与源码映射、常见问题、最小可运行流程、源码调用关系、复现检查清单。

每条关键命令包含：命令、来源、前置条件、验证状态、限制。使用 `⚠️ 未从源码确认`、`⚠️ 需要用户补充` 或 `推断：` 标注缺口。

## 不适用场景

需要修改、实现或调试目标仓库代码时，不使用本 Skill；需要论文 PDF 提取、论文结构化理解或按模板实现模型时，转交 `paper-pipeline`、`paper-understanding` 或 `paper-code-implementation`。