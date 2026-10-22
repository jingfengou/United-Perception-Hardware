"""Archive committed evidence byte-for-byte; never edit or overwrite a snapshot."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile


ROOT = Path(__file__).resolve().parents[2]
PREFIXES = ("docs/hardware/validation/", "docs/platform/validation/")


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def members(archive):
    records = {}
    for member in archive:
        if member.isdir():
            continue
        path = PurePosixPath(member.name)
        if (not member.isfile() or path.is_absolute() or ".." in path.parts
                or "\\" in member.name or not member.name.startswith(PREFIXES)):
            raise ValueError("Unsafe evidence member: " + member.name)
        if member.name in records:
            raise ValueError("Duplicate evidence member: " + member.name)
        content = archive.extractfile(member).read()
        blob = b"blob " + str(len(content)).encode() + b"\0" + content
        records[member.name] = {"bytes": len(content), "sha256": sha256(content),
                               "git_blob": hashlib.sha1(blob).hexdigest()}
    return records


def create(source, output):
    commit = subprocess.check_output(
        ["git", "rev-parse", source + "^{commit}"], cwd=ROOT).decode().strip()
    archive_path = output / "evidence-pre-consolidation-20261022.tar.gz"
    manifest_path = output / "evidence-pre-consolidation-20261022.manifest.json"
    if archive_path.exists() or manifest_path.exists():
        raise ValueError("Never overwrite an evidence archive or manifest")
    paths = [p.rstrip("/") for p in PREFIXES]
    raw = subprocess.check_output(
        ["git", "-c", "core.autocrlf=false", "archive", "--format=tar", commit, *paths], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        files = members(archive)
    tree = subprocess.check_output(["git", "ls-tree", "-r", "-z", commit, "--", *paths], cwd=ROOT)
    blobs = {path.decode(): metadata.split()[2].decode() for row in tree.split(b"\0") if row
             for metadata, path in [row.split(b"\t", 1)]}
    if {name: value["git_blob"] for name, value in files.items()} != blobs:
        raise ValueError("Archive bytes differ from committed Git blobs")
    # Do not inherit working-tree CRLF conversion, filesystem mtimes or gzip filenames.
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0) as stream:
        stream.write(raw)
    content = buffer.getvalue()
    manifest = {"schema": 1, "source_commit": commit, "prefixes": list(PREFIXES),
                "archive": {"name": archive_path.name, "bytes": len(content), "sha256": sha256(content)},
                "file_count": len(files), "uncompressed_file_bytes": sum(f["bytes"] for f in files.values()),
                "files": dict(sorted(files.items()))}
    output.mkdir(parents=True, exist_ok=True)
    archive_path.write_bytes(content)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    verify(archive_path, manifest_path)
    return manifest


def verify(archive_path, manifest_path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    content = archive_path.read_bytes()
    expected = manifest["archive"]
    if len(content) != expected["bytes"] or sha256(content) != expected["sha256"]:
        raise ValueError("Evidence archive size or SHA256 mismatch")
    with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as archive:
        files = members(archive)
    if files != manifest["files"] or len(files) != manifest["file_count"]:
        raise ValueError("Evidence member set or content mismatch")
    if sum(f["bytes"] for f in files.values()) != manifest["uncompressed_file_bytes"]:
        raise ValueError("Evidence uncompressed byte count mismatch")
    return {"verified_files": len(files), "archive_sha256": expected["sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("create")
    build.add_argument("--source", required=True)
    build.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("verify")
    check.add_argument("--archive", type=Path, required=True)
    check.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    result = create(args.source, args.output) if args.command == "create" else verify(args.archive, args.manifest)
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
