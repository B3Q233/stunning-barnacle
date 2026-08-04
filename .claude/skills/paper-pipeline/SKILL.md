---
name: paper-pipeline
description: >
  论文理解工具链——串联 MinerU 文档提取与 paper-understanding 结构化分析，将论文 PDF（或 arXiv URL）自动转化为可验证的结构化理解文档。支持两个独立子步骤：步骤一「提取」（MinerU PDF→Markdown），步骤二「理解」（Markdown→结构化理解文档）。用户可全自动串行执行，也可选择只跑其中一个步骤。用于论文复现工作流的第一步，当用户提到"分析这篇论文"、"解读这篇论文"、"提取并理解论文"、上传 PDF 或提供 arXiv URL 并要求产出理解文档时，必须使用此 skill。
metadata:
  requires:
    bins: [mineru-open-api, pdftoppm]
  optional:
    bins: [pdfinfo, pdffonts]
---

# 论文理解工具链 Skill

将论文 PDF 自动转化为结构化理解文档。串联两个子步骤：**提取**（MinerU）和 **理解**（paper-understanding 方法论）。

## 子步骤总览

```
用户输入: PDF 路径 / arXiv URL / 已有 Markdown
│
├─ 步骤一「提取」: MinerU PDF → Markdown
│   输入: PDF 或 URL
│   输出: {paper}.md + 图片补漏（可选）
│   可跳过: 如果用户已有 Markdown 或只需要步骤二
│
├─ 步骤二「理解」: Markdown → 结构化理解文档
│   输入: Markdown 文件（来自步骤一或用户提供）
│   输出: {paper}_understanding.md
│   依赖: paper-understanding 的模板和方法论
│
└─ 最终产物: 结构化理解文档（四模块：数据集/模型/对比实验/实现约束）
```

## 步骤一「提取」：MinerU PDF → Markdown

### 前置条件

- `mineru-open-api` 已安装（`npm install -g mineru-open-api`）
- 对于大文件（>10MB 或 >20页），需注册 MinerU token（https://mineru.net/apiManage/token）

### 输入

| 类型 | 示例 |
|------|------|
| arXiv URL | `https://arxiv.org/pdf/2511.05845` |
| 本地 PDF | `g:/Idea/papers/indirect_ad/2511.05845.pdf` |
| 其他 URL | 指向 PDF 的 HTTP/HTTPS 链接 |

### 执行

```bash
# 创建输出目录
mkdir -p g:/Idea/papers/{paper_name}/

# 小文件（<10MB, <20页）：使用 flash-extract（免 token）
mineru-open-api flash-extract "{pdf_path_or_url}" --language en -o "g:/Idea/papers/{paper_name}/"

# 大文件：先用 curl 下载 PDF，再用 extract（需 token）
curl -L --ssl-no-revoke -o "g:/Idea/papers/{paper_name}/{paper_name}.pdf" "{url}"
mineru-open-api extract "g:/Idea/papers/{paper_name}/{paper_name}.pdf" -o "g:/Idea/papers/{paper_name}/" -f md
```

### 输出

```
g:/Idea/papers/{paper_name}/
├── {paper_name}.pdf    ← 原始 PDF（从 URL 下载时）
└── {paper_name}.md     ← MinerU 提取的 Markdown
```

### 图片补漏（可选，仅步骤二需要架构图视觉确认时执行）

MinerU 提取时图片被替换为 `<!-- image-->` 占位符。如果论文包含关键架构图（步骤二需要视觉确认时），从原始 PDF 提取对应页面：

```bash
# 扫描 Markdown 中丢失的图片位置
grep -n '<!-- image-->' "g:/Idea/papers/{paper_name}/{paper_name}.md"

# 对架构图所在页面光栅化（假设第 3-4 页）
pdftoppm -jpeg -r 150 -f 3 -l 4 "g:/Idea/papers/{paper_name}/{paper_name}.pdf" "g:/Idea/papers/{paper_name}/page"
# 产出: page-3.jpg, page-4.jpg
```

**注意**：当前模型可能不支持多模态视觉输入。如果无法直接查看提取的图片，在这一步终止，提示用户：
> 以下页面包含架构图/关键图表，请人工查看后继续步骤二：[页码列表]。图片已保存至 `g:/Idea/papers/{paper_name}/page-*.jpg`。

### 验证

- 确认 Markdown 文件生成且非空（`wc -l` > 50）
- 浏览标题/摘要/章节结构是否完整
- 如果 MinerU 报错（`-60007` 服务不可用），等待 1 分钟后重试；如果 flash-extract 一直失败，提示用户注册 token 使用 extract 模式

---

## 步骤二「理解」：Markdown → 结构化理解文档

### 前置条件

- 步骤一产出（或用户已有的）Markdown 文件
- `paper-understanding` skill 的模板文件：`g:/Idea/.claude/skills/paper-understanding/references/template.md`

### 输入

| 类型 | 说明 |
|------|------|
| Markdown 文件 | 步骤一产出的 `{paper}.md` |
| 原始 PDF（可选） | 仅在需要架构图视觉确认或公式/表格补漏时使用 |

### 核心区别：Markdown 模式 vs PDF 模式

`paper-understanding` skill 原生假设 PDF 输入，内部用 `pdftotext`/`pdfplumber`/`pdftoppm` 提取。本步骤使用 **Markdown 模式**：

| 内容类型 | Markdown 模式（本步骤） | PDF 模式（原 paper-understanding） |
|---------|----------------------|----------------------------------|
| 正文叙述 | 直接读 Markdown | pdftotext 提取 |
| 表格 | MinerU 已识别为 Markdown 表格 | pdfplumber / 光栅化 |
| 公式 | MinerU 已识别为 LaTeX | pdftotext（可能乱码） |
| 架构图 | 需 PDF 光栅化补漏（步骤一图片补漏） | pdftoppm 光栅化 |

### 执行

严格按照 `paper-understanding` skill 的**第二步**（填充四模块字段模板）和**第三步**（自查清单）执行，但文本来源改为步骤一的 Markdown 文件：

1. **读 Markdown 全文**：获取完整的结构化文本（标题/段落/表格/公式）
2. **填充模块一：数据集**
   - 数据组成、获取方式、数据结构、预处理方式
   - 每项标注来源等级：【论文明确写出】/【AI推断补全】/【论文未提及】/【论文不明确】
3. **填充模块二：模型**
   - 输入/输出、模型结构（**必须用 mermaid 绘制架构图**）
   - 损失函数（LaTeX 还原，含所有权重系数和辅助项）
   - 评估方式（指标、评估协议、负采样策略）
   - 正则化技术清单（逐项标注 PyTorch 实现注意事项）
   - 初始化方式（逐项标注论文出处段落）
   - **架构图**：如果 Markdown 中 `<!-- image-->` 位置是关键架构图，使用步骤一的补漏图片做视觉确认
4. **填充模块三：对比实验汇总**
   - baseline 清单、对比维度、对比结果数值表（标注 Table/Figure 编号）
   - 用 mermaid 绘制对比结果概览图
5. **填充模块四：实现约束**
   - 关键超参数（lr、batch_size、epochs、optimizer 等，逐项标注来源等级）
   - 训练设置（硬件、训练时长、特殊技巧如 warmup/early stopping）
6. **自查清单**：逐项核对 paper-understanding §第三步的 checklist

### 输出

```
g:/Idea/papers/{paper_name}/
├── {paper_name}.pdf
├── {paper_name}.md
├── {paper_name}_understanding.md   ← 结构化理解文档（最终产物）
└── page-*.jpg                       ← 补漏图片（如有）
```

理解文档开头插入提取信息：

```markdown
## 0. 提取信息
- 提取工具链: paper-pipeline
- 提取方式: MinerU flash-extract
- 提取时间: {timestamp}
- 源文件: {pdf_path_or_url}
- 补漏页面: {页码列表，无则为"无"}
```

### 验证

- 四模块字段均已填充，无空章节
- 每条数值型信息已标注来源等级
- 模型结构 mermaid 图与论文架构图逐层对应
- 自查清单全部通过
- 文档末尾汇总所有【AI推断补全】和【论文未提及】项

### 交付提示

完成步骤二后，明确告知用户：

> 这份文档将作为后续代码实现的唯一依据。请重点核查标注为【AI推断补全】的部分（已在文末汇总），以及模型结构和损失函数部分是否与论文一致。确认无误后请明确告知"理解文档确认通过"，我会基于此继续生成代码骨架。

---

## 使用方式

### 方式 A：全自动串行

用户说"分析这篇论文 https://arxiv.org/pdf/2511.05845" 时，自动执行步骤一 → 步骤二。

### 方式 B：只跑步骤一

用户说"提取这篇论文为 Markdown"时，只执行步骤一，不进入步骤二。输出为 `{paper}.md`。

### 方式 C：只跑步骤二

用户已有 Markdown 文件时，说"理解这篇 Markdown"或直接指定 md 文件路径，跳过步骤一，从步骤二开始。

---

## 参考文件

- `../mineru-document-extractor-0.1.29/SKILL.md` — MinerU 完整使用说明
- `../paper-understanding/SKILL.md` — 论文理解方法论（模板/自查清单）
- `../paper-understanding/references/template.md` — 四模块结构化文档模板
- `../paper-understanding/references/common-pitfalls.md` — 常见理解错误案例
