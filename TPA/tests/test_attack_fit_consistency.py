"""攻击 fit.py main() 命名一致性回归测试。

背景：save_report(report, out_dir, name=attack_name) 需要 main() 定义
attack_name；tpa/random 有、pgd/bandwagon 曾缺失导致运行时 NameError。
本测试静态解析每个攻击 fit.py（含技能模板），保证 main() 在调用
save_report 前已赋值 attack_name。
"""
import ast
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ATTACK_DIRS = [
    PROJECT_ROOT / "attacks" / "tpa",
    PROJECT_ROOT / "attacks" / "pgd",
    PROJECT_ROOT / "attacks" / "bandwagon",
    PROJECT_ROOT / "attacks" / "random",
]


def main_assigns_attack_name(fit_path: Path) -> bool:
    tree = ast.parse(fit_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            assigned: set[str] = set()
            for sub in ast.walk(node):
                if isinstance(sub, ast.Assign):
                    for target in sub.targets:
                        if isinstance(target, ast.Name):
                            assigned.add(target.id)
            return "attack_name" in assigned
    return False


class AttackFitConsistencyTest(unittest.TestCase):
    def test_each_attack_main_defines_attack_name(self):
        for attack_dir in ATTACK_DIRS:
            fit_path = attack_dir / "fit.py"
            self.assertTrue(
                fit_path.exists(), f"missing {fit_path}")
            self.assertTrue(
                main_assigns_attack_name(fit_path),
                f"{fit_path} 的 main() 未定义 attack_name"
                "（save_report(report, out_dir, name=attack_name) 会 NameError）",
            )

    def test_template_fit_main_defines_attack_name(self):
        fit_path = PROJECT_ROOT.parent / ".codex" / "skills" / \
            "paper-code-implementation" / "assets" / \
            "attack-imp-direct-poison" / "fit.py"
        if not fit_path.exists():
            self.skipTest("技能模板不在本地")
        self.assertTrue(main_assigns_attack_name(fit_path))


if __name__ == "__main__":
    unittest.main()
