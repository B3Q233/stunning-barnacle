# -*- coding: utf-8 -*-
"""入库文本文件不得含绝对路径（AGENTS.md「路径与可迁移性」守护测试）。

扫描范围：``git ls-files`` 的跟踪文件（与"入库"定义一致），跳过二进制扩展名与
无法按 UTF-8 解码的文件。命中三类形态：

1. Windows 盘符路径（盘符 + 冒号 + 分隔符）；
2. POSIX 用户/系统绝对路径（/home、/Users、/root、/mnt、/usr）；
3. UNC 路径（双反斜杠 + 主机名 + 共享名）。

两类不判红（否则守护测试会被噪声淹没）：

- 脚本 shebang 行（`#!` 开头）里的解释器路径：POSIX 下依赖 PATH 解析，
  属于可移植写法；
- LaTeX 的换行/矩阵语法（连续反斜杠 + 单词），与 UNC 形态易混淆，故 UNC 规则
  要求主机名以字母开头且不紧跟 `{`。

为什么要有反向用例：守护测试最容易退化成恒真断言（扫描逻辑写错就永远通过），
所以用**运行时拼接**构造一个绝对路径样例，断言检测器能命中；拼接写法保证
本文件自身不含字面绝对路径。
"""
from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

ABS_RE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]")
POSIX_RE = re.compile(r"(?<![A-Za-z0-9.])/(?:home|Users|root|mnt|usr)/")
# UNC：主机名以字母开头、共享名以字母数字开头且后面不是词字符/`{`
# （这两条边界用于排除 LaTeX 的 \\small\. 、\\x_2\end{bmatrix} 等形态）
UNC_RE = re.compile(
    r"\\\\[A-Za-z][A-Za-z0-9-]{1,63}\\[A-Za-z0-9][A-Za-z0-9$_.-]{0,63}(?![\w{])")

SKIP_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".woff", ".woff2",
    ".ttf", ".map", ".js", ".pkl", ".pt", ".pth", ".npy", ".npz",
    ".bin", ".exe", ".dll", ".so", ".zip", ".gz",
}


def iter_tracked_text_files() -> list:
    """git 跟踪的文本候选文件（排除二进制扩展名）。"""
    out = subprocess.run(["git", "ls-files"], cwd=REPO_ROOT,
                         capture_output=True, text=True, check=True)
    files = []
    for line in out.stdout.splitlines():
        path = REPO_ROOT / line
        if path.suffix.lower() in SKIP_SUFFIXES or not path.is_file():
            continue
        files.append(path)
    return files


def find_violations(text: str) -> list:
    """返回 ``(行号, 命中片段)`` 列表（按出现顺序，重复项去重）。

    带行号是为了让失败信息可直接定位（``文件:行号: 命中内容``）。
    """
    found: list = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if line.startswith("#!"):
            continue                     # shebang：解释器路径依赖 PATH，属可移植写法
        for pattern in (ABS_RE, POSIX_RE, UNC_RE):
            for match in pattern.finditer(line):
                tail = line[match.start():match.start() + 60]
                value = tail.split()[0] if tail.split() else tail
                if (lineno, value) not in found:
                    found.append((lineno, value))
    return found


class NoAbsolutePathTest(unittest.TestCase):

    def test_tracked_text_files_have_no_absolute_paths(self):
        offenders: list = []
        for path in iter_tracked_text_files():
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            violations = find_violations(text)
            if violations:
                rel = path.relative_to(REPO_ROOT).as_posix()
                offenders += [f"{rel}:{lineno}: {value}"
                              for lineno, value in violations[:3]]
        self.assertEqual(
            [], offenders,
            "以下入库文件含绝对路径，违反 AGENTS.md「路径与可迁移性」：\n"
            + "\n".join(offenders[:40]))

    def test_detector_catches_synthetic_absolute_path(self):
        """反向用例：运行时拼接样例，证明检测器不是恒真。"""
        sample = "C" + ":" + chr(92) + "Users" + chr(92) + "someone"
        self.assertTrue(find_violations(sample))

    def test_detector_catches_posix_and_unc(self):
        self.assertTrue(find_violations("/" + "home" + "/user/x"))
        self.assertTrue(find_violations(chr(92) * 2 + "server" + chr(92) + "share"))


if __name__ == "__main__":
    unittest.main()
