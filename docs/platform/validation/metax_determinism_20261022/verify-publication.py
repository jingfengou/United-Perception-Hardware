"""Check immutable evidence and staged runtime against the executed manifest."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tarfile
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
index = json.loads((HERE / "evidence-index.json").read_text())
audit = json.loads((HERE / "audit.json").read_text())
expected = {str((HERE / name).relative_to(ROOT)).replace("\\", "/"): (value["sha256"], False)
            for name, value in index["text_files"].items()}
expected.update({name: (value, True) for name, value in audit["source_manifests"]["changed"]["normalized_files"].items()})
entries = subprocess.check_output(["git", "ls-files", "--stage", "-z"], cwd=ROOT).split(b"\0")
staged = {}
for entry in entries:
    if entry:
        metadata, name = entry.split(b"\t", 1)
        mode, blob, stage = metadata.split()
        assert stage == b"0", "Unmerged index"
        staged[name.decode()] = blob
names = sorted(expected)
payload = subprocess.check_output(["git", "cat-file", "--batch"], cwd=ROOT,
                                  input=b"\n".join(staged[name] for name in names) + b"\n")
position = 0
for name in names:
    end = payload.index(b"\n", position)
    _, kind, size = payload[position:end].split()
    assert kind == b"blob"
    start = end + 1
    data = payload[start:start + int(size)]
    position = start + int(size) + 1
    target, normalize = expected[name]
    assert hashlib.sha256(data.replace(b"\r\n", b"\n") if normalize else data).hexdigest() == target, name
    working = (ROOT / name).read_bytes()
    assert hashlib.sha256(working.replace(b"\r\n", b"\n") if normalize else working).hexdigest() == target, name
archive = HERE / "metax-determinism-evidence.tar.gz"
outer = json.loads((HERE / (archive.name + ".index.json")).read_text())
assert hashlib.sha256(archive.read_bytes()).hexdigest() == outer["archive"]["sha256"]
with tarfile.open(archive) as packed:
    assert packed.extractfile("evidence-index.json").read() == (HERE / "evidence-index.json").read_bytes()
for path in [ROOT / "README.md", HERE / "README.md", *[ROOT / "docs/platform" / name for name in (
        "METAX_DETERMINISM.md", "CAPABILITIES.md", "INTERFACES.md", "VALIDATION.md", "REAL_HARDWARE.md")]]:
    text = path.read_text(encoding="utf-8")
    for link in re.findall(r"\]\(([^)]+)\)", text):
        if "://" not in link and not link.startswith("#"):
            assert (path.parent / link.split("#")[0]).exists(), (path, link)
    if path.name == "METAX_DETERMINISM.md":
        for block in re.findall(r"```python\n(.*?)```", text, re.S):
            ast.parse(block)
suites = ET.parse(HERE / "cpu-junit.xml").findall(".//testsuite")
assert sum(int(suite.get("tests")) for suite in suites) == 277
assert all(int(suite.get(key, "0")) == 0 for suite in suites for key in ("errors", "failures", "skipped"))
result = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "raw_text_blobs_byte_identical": len(index["text_files"]),
          "normalized_runtime_blobs_match_executed": len(audit["source_manifests"]["changed"]["normalized_files"]),
          "runtime_manifest_sha256": audit["source_manifests"]["changed"]["sha256"], "archive": outer["archive"],
          "cpu_tests_passed": 277, "cpu_failed_skipped": 0, "links_and_python_snippets": "passed"}
(HERE / "publication-audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, indent=2))
