import importlib
import inspect
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2] / "up_platform" / "ops"


class ContractTest(unittest.TestCase):
    def test_templates_match_each_base_signature(self):
        for source in ROOT.glob("*/base.py"):
            base_module = importlib.import_module(f"up_platform.ops.{source.parent.name}.base")
            base = next(value for name, value in vars(base_module).items() if name.endswith("Ops"))
            for vendor in ("ascend", "cambricon"):
                module = importlib.import_module(f"up_platform.ops.{source.parent.name}.{vendor}")
                cls = getattr(module, vendor.capitalize() + base.__name__)
                self.assertFalse(inspect.isabstract(cls))
                for name in base.__abstractmethods__:
                    self.assertIn(name, cls.__dict__)
                    self.assertEqual(inspect.signature(getattr(base, name)), inspect.signature(getattr(cls, name)))

    def test_platform_does_not_import_up(self):
        import ast
        for source in ROOT.parent.rglob("*.py"):
            for node in ast.walk(ast.parse(source.read_text())):
                if isinstance(node, ast.ImportFrom):
                    self.assertNotEqual((node.module or "").split(".")[0], "up", str(source))
                elif isinstance(node, ast.Import):
                    self.assertFalse(any(a.name.split(".")[0] == "up" for a in node.names), str(source))


if __name__ == "__main__":
    unittest.main()
