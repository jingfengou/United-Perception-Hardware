"""Preserve exact Python/config sources used by a hardware accuracy phase."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def sha(value):
    return hashlib.sha256(value).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    index_path = args.output.with_name(args.output.name + ".index.json")
    if args.output.exists() or index_path.exists():
        raise ValueError("Use fresh snapshot paths; preserve previous phases")
    root = args.source.resolve()
    files = {}
    normalized = {}
    for folder in ("up", "packages/up-platform/src/up_platform", "configs", "benchmark/hardware", "tools", "tests"):
        for path in sorted((root / folder).rglob("*")):
            if not path.is_file() or path.suffix not in (".py", ".yaml", ".json", ".cpp", ".cu", ".h", ".hpp", ".cuh"):
                continue
            if path.is_symlink():
                raise ValueError("Source snapshot cannot contain symlinks")
            relative = path.relative_to(root).as_posix()
            content = path.read_bytes()
            files[relative] = sha(content)
            if relative.startswith("up/") and path.suffix == ".py":
                normalized[relative] = sha(content.replace(b"\r\n", b"\n"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, "w:gz") as stream:
        for relative in files:
            stream.add(root / relative, arcname=relative, recursive=False)
    index = {"source_layout": "up-platform-v1", "archive_sha256": sha(args.output.read_bytes()), "files": files,
             "normalized_up_files": normalized,
             "normalized_up_manifest_sha256": sha(json.dumps(normalized, sort_keys=True).encode()),
             "native_binary_hashes": {p.relative_to(root).as_posix(): sha(p.read_bytes())
                                      for folder in ("up", "packages/up-platform/src/up_platform")
                                      for p in sorted((root / folder).rglob("*.so"))},
             "scope": "Collected source snapshot; training report hashes identify the actually executed critical files",
             "exclusions": ["SDK binaries", "native extension binaries", "weights", "data images", "credentials"]}
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"source_files": len(files), "up_files": len(normalized),
                      "normalized_up_manifest": index["normalized_up_manifest_sha256"],
                      "archive_sha256": index["archive_sha256"]}))


if __name__ == "__main__":
    main()
