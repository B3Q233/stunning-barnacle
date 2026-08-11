# MinerU 默认输出路径调整设计文档

- 日期：2026-08-11
- 状态：设计已确认，待实现
- 范围：修改 mineru-document-extractor 技能的默认输出路径规则，并将当前
  PGD 提取产物按新约定归位
- 流程技能：brainstorming → writing-plans → executing-plans

## 1. 背景与目标

mineru-document-extractor 技能的默认输出规则是
`~/MinerU-Skill/<name>_<hash>/`（hash = 源路径 MD5 前 6 位），与仓库既有的
论文存放约定不一致：

- `paper-pipeline` 技能固定输出到 `g:/Idea/papers/{paper_name}/`，md 与
  原始 PDF 同目录；
- `papers/indirect_ad/` 已经是"文件夹 = 论文名，内含 md + pdf"的结构；
- 而 `papers/PGD.pdf` 与 `MinerU-Skill/PGD_07b060/PGD.md` 是散落的例外。

目标：把 mineru 提取的默认保存路径改为 `papers/<源文件名（不含扩展名）>/`，
提取出的 md 与原文件（PDF）同目录存放，并将当前 PGD 提取产物按新约定归位。

## 2. 变更点

### 2.1 技能默认路径规则（两处副本）

修改以下文件的 Agent rules 中默认输出目录一条：

- `G:\Idea\.codex\skills\mineru-document-extractor\SKILL.md`
- `G:\Idea\.claude\skills\mineru-document-extractor-0.1.29\SKILL.md`

旧规则：

> When user does NOT specify `-o`, generate output directory:
> `~/MinerU-Skill/<name>_<hash>/` where `<hash>` = first 6 chars of MD5 of
> the source path

新规则：

> When user does NOT specify `-o`, default to the project `papers/` directory
> (e.g. `g:/Idea/papers/`): create a folder named after the source file
> (without extension), and save the extracted Markdown together with the
> original source file (e.g. the PDF) in that folder — matching the
> paper-pipeline `{paper_name}/` convention. Run
> `mkdir -p papers/<name>/` before extraction, then pass `-o papers/<name>/`.

### 2.2 当前 PGD 产物归位（移动，非复制）

- 新建 `g:/Idea/papers/PGD/`
- `g:/Idea/papers/PGD.pdf` → `g:/Idea/papers/PGD/PGD.pdf`
- `g:/Idea/MinerU-Skill/PGD_07b060/PGD.md` → `g:/Idea/papers/PGD/PGD.md`
- 旧 `MinerU-Skill/PGD_07b060/` 目录保留其余 KPV 测试文件
  （html/json/mathjax），不删除。

### 2.3 历史文档

08-08 KPV spec/plan 中引用 `PGD_07b060/PGD.md` 的位置属于历史记录，不改动。

## 3. 成功标准

- 两处技能副本的默认输出规则均指向 `papers/<name>/`，且说明 md 与原文件
  同目录。
- `g:/Idea/papers/PGD/` 下存在 `PGD.md` 与 `PGD.pdf`。
- 旧目录 `MinerU-Skill/PGD_07b060/` 除 PGD.md 外其余文件未受影响。
- 未改动任何历史文档与代码。

## 4. 验收方式

- 重读两处 SKILL.md，确认规则文本与旧规则完全替换。
- 检查 `papers/PGD/` 目录内容（md + pdf 齐全）。
- 检查 `MinerU-Skill/PGD_07b060/` 其余文件仍在。

## 5. 范围外（不做）

- 不改动 paper-pipeline 技能（其输出约定已是目标结构）。
- 不更新历史 KPV spec/plan 文档。
- 不清理旧 `MinerU-Skill/PGD_07b060/` 目录。
- 不修改 .gitignore（papers/ 与 MinerU-Skill/ 均已忽略）。
