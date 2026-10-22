import importlib.util
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location("archive_evidence", Path(__file__).with_name("archive_evidence.py"))
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def tar_content(name, kind=tarfile.REGTYPE):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        item = tarfile.TarInfo(name)
        item.type = kind
        item.size = 3 if kind == tarfile.REGTYPE else 0
        archive.addfile(item, io.BytesIO(b"raw"))
    return buffer.getvalue()


class ArchiveTests(unittest.TestCase):
    def test_create_checks_git_blobs_and_refuses_overwrite(self):
        name = "docs/platform/validation/example.json"
        raw = tar_content(name)
        with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
            blob = tool.members(archive)[name]["git_blob"]
        tree = ("100644 blob " + blob + "\t" + name + "\0").encode()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.object(tool.subprocess, "check_output", side_effect=[b"commit\n", raw, tree]) as git:
                index = tool.create("ref", output)
                self.assertIn("core.autocrlf=false", git.call_args_list[1].args[0])
            archive = output / index["archive"]["name"]
            manifest = output / "evidence-pre-consolidation-20261022.manifest.json"
            self.assertEqual(tool.verify(archive, manifest)["verified_files"], 1)
            archive.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                tool.verify(archive, manifest)
            with patch.object(tool.subprocess, "check_output", return_value=b"commit\n"):
                with self.assertRaisesRegex(ValueError, "Never overwrite"):
                    tool.create("ref", output)

    def test_rejects_traversal_symlink_and_unrelated_files(self):
        for name, kind in (("docs/platform/validation/../escape", tarfile.REGTYPE),
                           ("docs/platform/validation/link", tarfile.SYMTYPE),
                           ("up/model.py", tarfile.REGTYPE)):
            with self.subTest(name=name), tarfile.open(fileobj=io.BytesIO(tar_content(name, kind))) as archive:
                with self.assertRaisesRegex(ValueError, "Unsafe"):
                    tool.members(archive)
