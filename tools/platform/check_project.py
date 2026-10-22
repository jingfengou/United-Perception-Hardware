"""Check consolidation invariants, archived fixtures and maintained Markdown links."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[2]
BASE = "a3e2752f454c72d6b46e0b64d0debeef0d7105a3"
MANIFEST = ROOT / "docs/architecture/consolidation-protected.json"
PREFIXES = ("up/", "configs/", "packages/up-platform/src/up_platform/backends/nvidia/csrc/",
            "packages/up-platform/src/up_platform/experimental/numerics/")
SINGLE = "packages/up-platform/src/up_platform/backends/metax/determinism.py"
TEXT = (".py", ".yaml", ".json", ".cpp", ".cu", ".h", ".hpp", ".cuh", ".md")


def digest(path, content):
    if path.endswith(TEXT):
        content = content.replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def write_manifest():
    if MANIFEST.exists():
        raise ValueError("Do not replace the protected baseline")
    raw = subprocess.check_output(["git", "ls-tree", "-r", "-z", BASE], cwd=ROOT)
    objects = []
    for row in raw.split(b"\0"):
        if not row:
            continue
        metadata, path = row.split(b"\t", 1)
        path = path.decode()
        if path.startswith(PREFIXES) or path == SINGLE:
            objects.append((path, metadata.split()[2]))
    data = subprocess.check_output(["git", "cat-file", "--batch"],
                                   input=b"\n".join(oid for _, oid in objects) + b"\n", cwd=ROOT)
    position, files = 0, {}
    for path, _ in objects:
        end = data.index(b"\n", position)
        size = int(data[position:end].split()[-1])
        content = data[end + 1:end + 1 + size]
        files[path] = digest(path, content)
        position = end + size + 2
    result = {"source_commit": BASE, "normalization": "CRLF to LF only for declared source text suffixes",
              "scope": "All UP/config files and existing native, MetaX and experimental mathematical implementations",
              "files": files}
    MANIFEST.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def check_protected(root=ROOT):
    expected = json.loads((root / MANIFEST.relative_to(ROOT)).read_text(encoding="utf-8"))["files"]
    changed = [name for name, value in expected.items()
               if not (root / name).is_file() or digest(name, (root / name).read_bytes()) != value]
    # New business recipes/algorithms also require an explicit decision, not a hidden cleanup change.
    extra = [p.relative_to(root).as_posix() for folder in ("up", "configs") for p in (root / folder).rglob("*")
             if p.is_file() and p.suffix in TEXT and p.relative_to(root).as_posix() not in expected]
    if changed or extra:
        raise ValueError("Protected algorithm/config changes: " + repr(changed + extra))
    return len(expected)


def check_fixtures(root=ROOT):
    index = json.loads((root / "docs/reports/evidence.manifest.json").read_text(encoding="utf-8"))
    count = 0
    for path in (root / "tests/fixtures/protocols").rglob("*.json"):
        original = "docs/hardware/validation/" + path.relative_to(root / "tests/fixtures/protocols").as_posix()
        if hashlib.sha256(path.read_bytes()).hexdigest() != index["files"][original]["sha256"]:
            raise ValueError("Archived fixture changed: " + str(path))
        count += 1
    if count != 6 or index["file_count"] != 1505:
        raise ValueError("Incomplete fixtures or original archive index")
    return count


def check_links(root=ROOT):
    paths = [root / "README.md", root / "benchmark/hardware/README.md", root / "tools/platform/README.md"]
    paths += [p for folder in ("docs/platform", "docs/hardware", "docs/reports", "docs/architecture")
              for p in (root / folder).rglob("*.md")]
    count, missing = 0, []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for value in re.findall(r"\]\(([^)]+)\)", text):
            url = urlsplit(value)
            if url.scheme or not url.path or " " in value:
                continue
            target = path.parent / unquote(url.path)
            count += 1
            if not target.exists():
                missing.append(path.relative_to(root).as_posix() + ": " + value)
    if missing:
        raise ValueError("Broken maintained links:\n" + "\n".join(missing))
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-protected", action="store_true")
    args = parser.parse_args()
    if args.write_protected:
        write_manifest()
    print(json.dumps({"protected_files": check_protected(), "archived_fixtures": check_fixtures(),
                      "local_markdown_links": check_links()}, indent=2))


if __name__ == "__main__":
    main()
