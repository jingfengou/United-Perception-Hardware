"""Package platform split regression evidence without loading a device or UP."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tarfile


def sha256(path, normalize=False):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block.replace(b"\r\n", b"\n") if normalize else block)
    return digest.hexdigest()


def verify_runtime(root):
    summary = json.loads((root / "run-03/summary.json").read_text())
    result = {}
    for label in ("baseline", "changed"):
        expected = summary["sources"][label]
        actual = {name: sha256(root / label / name, normalize=True) for name in expected["normalized_files"]}
        if actual != expected["normalized_files"]:
            raise ValueError("Runtime source changed after execution: " + label)
        result[label] = {"source_manifest_sha256": expected["sha256"], "files": len(actual), "verified": True}
    a, b = [summary["sources"][label]["normalized_files"] for label in ("baseline", "changed")]
    old_csrc = {name.removeprefix("up/extensions/csrc/"): value for name, value in a.items()
                if name.startswith("up/extensions/csrc/")}
    new_csrc = {name.removeprefix("packages/up-platform/src/up_platform/backends/nvidia/csrc/"): value
                for name, value in b.items()
                if name.startswith("packages/up-platform/src/up_platform/backends/nvidia/csrc/")}
    result["native_sources"] = {"files": len(old_csrc), "unchanged": old_csrc == new_csrc}
    names = {name for name in a if name.startswith("up/models/") or (
        name.startswith("up/tasks/") and "/models/" in name)}
    result["algorithms"] = {"files": len(names), "differences": sorted(name for name in names if a[name] != b.get(name))}
    result["task_runner_changes"] = sorted(name for name in a if name.startswith("up/tasks/")
                                          and "/runner/" in name and a[name] != b.get(name))
    return result


def verify_selected_data(root):
    folder = root / "run-01/data"
    data = json.loads((folder / "data.json").read_text())
    selection = json.loads((folder / "selection.json").read_text())
    result = {}
    inherited = False
    for dataset in ("cifar10", "pennfudan"):
        for split, count in (("train", 8), ("test", 4)):
            key = dataset + "/" + split
            entry = data[dataset][split]
            meta = Path(entry["meta"])
            if dataset == "cifar10":
                lines = [line.rsplit(" ", 1) for line in meta.read_text().splitlines()]
                images = [name for name, _ in lines]
            else:
                images = [item["file_name"] for item in json.loads(meta.read_text())["images"]]
            hashes = {name: sha256(Path(entry["image_dir"]) / name) for name in images}
            expected = selection["subsets"][key]
            if len(images) != count or hashes != expected["images"] or sha256(meta) != expected["metadata_sha256"]:
                raise ValueError("Selected data changed: " + key)
            if dataset == "cifar10":
                records = [[name, hashes[name], int(label)] for name, label in lines]
            else:
                records = [[name, hashes[name], sha256(Path(entry["mask_dir"]) / (Path(name).stem + "_mask.png"))]
                           for name in images]
            digest = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
            parent_fields = len(entry.get("indices", [])) != count or entry["image_manifest_sha256"] != digest
            inherited |= parent_fields
            result[key] = {"count": len(images), "metadata_sha256": sha256(meta), "images": hashes, "verified": True,
                           "image_manifest_sha256": digest, "parent_summary_fields_retained": parent_fields}
    result["initial_driver_metadata_limit"] = (
        "The original subset manifest retains parent indices/image_manifest_sha256 fields. "
        "Actual selected metadata, image bytes and loader batches were checked separately. "
        "The published driver now recomputes these summary fields; original evidence is unchanged.") if inherited else None
    return result


def collect(root, output, backend):
    if output.exists() or Path(str(output) + ".index.json").exists():
        raise ValueError("Use a fresh archive path; never overwrite evidence")
    index = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "backend": backend,
             "runtime": verify_runtime(root), "data": verify_selected_data(root),
             "text_files": {}, "retained_binary_files": {}}
    included = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or not (folder.name.startswith(("run-", "repeat-", "same-weights")) or folder.name == "tools"):
            continue
        for directory, subdirs, names in os.walk(folder, followlinks=False):
            subdirs[:] = [name for name in subdirs if name != "__pycache__"]
            for name in sorted(names):
                path = Path(directory) / name
                relative = path.relative_to(root).as_posix()
                if path.suffix in (".pt", ".pth"):
                    index["retained_binary_files"][relative] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
                elif path.suffix in (".json", ".log", ".txt", ".py", ".sh"):
                    index["text_files"][relative] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
                    included.append(path)
    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix in (".json", ".log", ".sh", ".py"):
            index["text_files"][path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
            included.append(path)
    for source in (root / "baseline/up/extensions", root / "changed/packages/up-platform/src/up_platform/backends/nvidia"):
        for path in sorted(source.glob("_C*.so")):
            index["retained_binary_files"][path.relative_to(root).as_posix()] = {
                "bytes": path.stat().st_size, "sha256": sha256(path)}
    payload = json.dumps(index, indent=2, sort_keys=True).encode()
    import io
    with tarfile.open(output, "w:gz") as archive:
        for path in sorted(set(included)):
            archive.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
        info = tarfile.TarInfo("evidence-index.json")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    index["archive"] = {"filename": output.name, "bytes": output.stat().st_size, "sha256": sha256(output)}
    Path(str(output) + ".index.json").write_text(json.dumps(index, indent=2, sort_keys=True))
    return {"archive": index["archive"], "text_files": len(index["text_files"]),
            "retained_binary_files": len(index["retained_binary_files"]), "runtime": index["runtime"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("nvidia", "metax"), required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.root.resolve(), args.output.resolve(), args.backend), indent=2, sort_keys=True))
