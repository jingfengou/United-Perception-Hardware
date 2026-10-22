"""Stdlib tests for the read-only upstream audit tool."""
import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location(
    "audit_upstream", Path(__file__).with_name("audit_upstream.py"))
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class AuditTests(unittest.TestCase):
    def test_status_parser_preserves_paths_and_renames(self):
        raw = b"A\0new file\0M\0changed\0R100\0old name\0new name\0D\0gone\0"
        self.assertEqual(list(audit.changes(raw)), [
            {"status": "A", "old_path": "", "path": "new file"},
            {"status": "M", "old_path": "changed", "path": "changed"},
            {"status": "R100", "old_path": "old name", "path": "new name"},
            {"status": "D", "old_path": "gone", "path": ""},
        ])

    def test_evidence_is_not_counted_as_product_documentation(self):
        self.assertEqual(audit.area("docs/platform/validation/run/source.py"), "frozen_evidence")
        self.assertEqual(audit.area("docs/platform/INTERFACES.md"), "documentation")

    def test_classification_distinguishes_migration_and_experiment(self):
        root = "packages/up-platform/src/up_platform/"
        self.assertEqual(audit.purpose(root + "backends/nvidia/csrc/nms.cu", "R100")[0], "nvidia_ops")
        self.assertEqual(audit.purpose(root + "experimental/numerics/conv.py", "A")[0], "experimental")
        self.assertEqual(audit.purpose(root + "backends/metax/determinism.py", "A")[0], "metax_fix")

    def test_unknown_change_requires_manual_review(self):
        with self.assertRaisesRegex(ValueError, "Manual purpose review"):
            audit.purpose("up/new_algorithm.py", "M", "+change_loss_formula()")


if __name__ == "__main__":
    unittest.main()
