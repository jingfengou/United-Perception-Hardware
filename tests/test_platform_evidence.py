import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("platform_evidence", ROOT / "tools/collect_platform_evidence.py")
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def test_runtime_verifier_detects_modified_source(tmp_path):
    sources = {}
    for label in ("baseline", "changed"):
        directory = tmp_path / label / "up/models"
        directory.mkdir(parents=True)
        (directory / "model.py").write_bytes(b"pass\r\n")
        files = {"up/models/model.py": hashlib.sha256(b"pass\n").hexdigest()}
        sources[label] = {"normalized_files": files, "sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True).encode()).hexdigest()}
    (tmp_path / "run-03").mkdir()
    (tmp_path / "run-03/summary.json").write_text(json.dumps({"sources": sources}))
    assert tool.verify_runtime(tmp_path)["algorithms"]["differences"] == []
    (tmp_path / "changed/up/models/model.py").write_bytes(b"changed\n")
    with pytest.raises(ValueError, match="Runtime source changed"):
        tool.verify_runtime(tmp_path)


def test_archive_indexes_large_binaries_but_preserves_raw_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(tool, "verify_runtime", lambda root: {"verified": True})
    monkeypatch.setattr(tool, "verify_selected_data", lambda root: {"verified": True})
    folder = tmp_path / "run-01"
    folder.mkdir()
    (folder / "report.json").write_text('{"passed": false}')
    (folder / "checkpoint.pt").write_bytes(b"binary")
    output = tmp_path / "evidence.tar.gz"
    result = tool.collect(tmp_path, output, "metax")
    with tarfile.open(output) as archive:
        assert "run-01/report.json" in archive.getnames()
        assert "run-01/checkpoint.pt" not in archive.getnames()
        index = json.load(archive.extractfile("evidence-index.json"))
        assert index["retained_binary_files"]["run-01/checkpoint.pt"]["sha256"] == tool.sha256(folder / "checkpoint.pt")
        assert json.load(archive.extractfile("run-01/report.json")) == {"passed": False}
    assert result["archive"]["sha256"] == tool.sha256(output)
    with pytest.raises(ValueError, match="never overwrite"):
        tool.collect(tmp_path, output, "metax")
