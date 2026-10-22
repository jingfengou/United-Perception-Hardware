import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location("check_project", Path(__file__).with_name("check_project.py"))
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


class ProjectChecks(unittest.TestCase):
    def test_algorithm_mutation_and_new_business_file_are_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "up").mkdir()
            source = root / "up/model.py"
            source.write_bytes(b"value = 1\r\n")
            manifest = root / tool.MANIFEST.relative_to(tool.ROOT)
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps({"files": {"up/model.py": tool.digest("up/model.py", b"value = 1\n")}}))
            self.assertEqual(tool.check_protected(root), 1)
            source.write_bytes(b"value = 2\n")
            with self.assertRaisesRegex(ValueError, "Protected algorithm"):
                tool.check_protected(root)
            source.write_bytes(b"value = 1\n")
            (root / "up/new_loss.py").write_text("pass\n")
            with self.assertRaisesRegex(ValueError, "new_loss"):
                tool.check_protected(root)

    def test_archived_fixtures_match_original_bytes(self):
        self.assertEqual(tool.check_fixtures(), 6)
