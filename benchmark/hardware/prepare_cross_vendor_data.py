"""Copy and verify the frozen small real-data accuracy cohort, without resampling."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_verified(source, destination, expected=None):
    source, destination = Path(source), Path(destination)
    actual = sha256(source)
    if expected is not None and actual != expected:
        raise ValueError("Source hash mismatch: " + str(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(destination) != actual:
            raise ValueError("Refusing to replace different data: " + str(destination))
    else:
        shutil.copyfile(source, destination)
    return actual


def verify_data(manifest):
    evidence = {}
    for name in ("cifar10", "pennfudan"):
        for split in ("train", "test"):
            entry = manifest[name][split]
            if sha256(entry["meta"]) != entry["meta_sha256"]:
                raise ValueError("Changed metadata: " + name + "/" + split)
            if name == "cifar10":
                records = [line.rsplit(" ", 1) for line in Path(entry["meta"]).read_text().splitlines()]
                hashes = [[filename, sha256(Path(entry["image_dir"]) / filename), int(label)]
                          for filename, label in records]
            else:
                records = json.loads(Path(entry["meta"]).read_text())["images"]
                hashes = [[image["file_name"], sha256(Path(entry["image_dir"]) / image["file_name"]),
                           sha256(Path(entry["mask_dir"]) / (Path(image["file_name"]).stem + "_mask.png"))]
                          for image in records]
            if len(hashes) != entry["count"] or hashlib.sha256(
                    json.dumps(hashes, sort_keys=True).encode()).hexdigest() != entry["image_manifest_sha256"]:
                raise ValueError("Changed image/label cohort: " + name + "/" + split)
            evidence[name + "/" + split] = {"count": len(hashes), "verified": True,
                                              "image_manifest_sha256": entry["image_manifest_sha256"]}
    for name, key in (("yolov5_anchors", "path"), ("detr", "checkpoint"), ("resnet50_weights", "path")):
        expected = manifest[name]["checkpoint_sha256" if name == "detr" else "sha256"]
        if sha256(manifest[name][key]) != expected:
            raise ValueError("Changed asset: " + name)
        evidence[name] = expected
    root = Path(manifest["detr"]["source"])
    files = {str(path.relative_to(root)): sha256(path) for path in sorted(root.rglob("*.py"))}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    if digest != manifest["detr"]["source_sha256"]:
        raise ValueError("Changed official DETR source")
    evidence["detr_source"] = digest
    return evidence


def prepare(source_manifest, output):
    old = json.loads(source_manifest.read_text())
    root = output.resolve()
    result = {key: old[key] for key in ("split_seed", "cifar10", "pennfudan", "detr", "yolov5_anchors", "resnet50_weights")}
    for name in ("cifar10", "pennfudan"):
        for split in ("train", "test"):
            entry = result[name][split]
            old_images = Path(entry["image_dir"])
            meta = root / name / (split + (".txt" if name == "cifar10" else ".json"))
            copy_verified(entry["meta"], meta, entry["meta_sha256"])
            images = root / name / (split if name == "cifar10" else "images")
            filenames = ([line.rsplit(" ", 1)[0] for line in meta.read_text().splitlines()] if name == "cifar10"
                         else [record["file_name"] for record in json.loads(meta.read_text())["images"]])
            for filename in filenames:
                copy_verified(old_images / filename, images / filename)
                if name == "pennfudan":
                    mask = Path(filename).stem + "_mask.png"
                    copy_verified(old_images.parent / "PedMasks" / mask, root / name / "masks" / mask)
            entry.update(meta=str(meta), image_dir=str(images))
            if name == "pennfudan":
                entry["mask_dir"] = str(root / name / "masks")
    for name, key, filename in (("yolov5_anchors", "path", "yolov5-anchors.json"),
                               ("resnet50_weights", "path", "resnet50-state-dict.pth"),
                               ("detr", "checkpoint", "detr-r50-tensors.pt")):
        entry = result[name]
        expected = entry["checkpoint_sha256" if name == "detr" else "sha256"]
        destination = root / "weights" / filename
        copy_verified(entry[key], destination, expected)
        entry[key] = str(destination)
    source = Path(result["detr"]["source"])
    target = root / "external/detr"
    for path in sorted(source.rglob("*")):
        if path.is_file() and not any(part in (".git", "__pycache__") for part in path.relative_to(source).parts):
            copy_verified(path, target / path.relative_to(source))
    result["detr"]["source"] = str(target)
    evidence = verify_data(result)
    root.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, indent=2, sort_keys=True)
    destination = root / "data.json"
    if destination.exists() and destination.read_text() != payload:
        raise ValueError("Refusing to replace another cohort manifest")
    destination.write_text(payload)
    (root / "integrity.json").write_text(json.dumps(evidence, indent=2, sort_keys=True))
    return {"manifest": str(destination), "sha256": sha256(destination), "integrity": evidence}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        result = verify_data(json.loads(args.manifest.read_text()))
    else:
        if args.output is None:
            parser.error("--output is required to prepare a copy")
        result = prepare(args.manifest, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
