"""Prepare real, disjoint datasets for an upstream/routed accuracy regression."""
import argparse
import hashlib
import io
import json
from pathlib import Path

PARQUET_SHA256 = {
    "train": "8428b53a88a11ac374111006708df51469e315a22ac6d66470afd9c78d2ae883",
    "test": "841389e6f2d64f28bf17310e430aebac20ec3ba611a3c5e231dc93c645ce84de",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def convert_trusted_weights(source, destination):
    """Normalize a verified official legacy artifact, preserving every tensor."""
    import torch

    state = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("Expected a tensor-only official state dictionary")
    torch.save(state, destination)
    restored = torch.load(destination, map_location="cpu", weights_only=True)
    if state.keys() != restored.keys() or not all(torch.equal(value, restored[key]) for key, value in state.items()):
        raise ValueError("Weight serialization conversion changed tensor values")


def prepare_cifar(root, train_size, test_size, seed, source):
    import numpy as np
    from torchvision.datasets import CIFAR10

    result = {"dataset": "CIFAR-10", "source": "https://www.cs.toronto.edu/~kriz/cifar.html",
              "transport": source}
    for split, count in (("train", train_size), ("test", test_size)):
        if source == "original":
            dataset = CIFAR10(str(root / "downloads"), train=split == "train", download=True)
            length = len(dataset)
            get_item = dataset.__getitem__
        else:
            import pyarrow.parquet as pq
            from PIL import Image
            from torchvision.datasets.utils import download_url
            revision = "0b2714987fa478483af9968de7c934580d0bb9a2"
            filename = split + "-00000-of-00001.parquet"
            url = "https://hf-mirror.com/datasets/uoft-cs/cifar10/resolve/{}/plain_text/{}".format(
                revision, filename)
            download_url(url, str(root / "downloads"), filename=filename)
            archive = root / "downloads" / filename
            if sha256(archive) != PARQUET_SHA256[split]:
                raise ValueError("CIFAR-10 Parquet checksum differs from the pinned upstream artifact")
            table = pq.read_table(archive)
            length = len(table)
            if length != (50000 if split == "train" else 10000):
                raise ValueError("Unexpected CIFAR-10 partition length")
            def get_item(index):
                row = table.slice(index, 1).to_pylist()[0]
                image = Image.open(io.BytesIO(row["img"]["bytes"])).convert("RGB")
                return image, row["label"]
            result[split + "_archive"] = {"url": url, "revision": revision, "sha256": sha256(archive)}
        if not 0 < count <= length:
            raise ValueError("Invalid {} size: {}".format(split, count))
        indices = np.random.RandomState(seed).permutation(length)[:count].tolist()
        directory = root / "cifar10" / split
        directory.mkdir(parents=True, exist_ok=True)
        records, file_hashes = [], []
        for index in indices:
            image, label = get_item(index)
            path = directory / "{:05d}.png".format(index)
            image.save(path)
            records.append("{} {}\n".format(path.name, label))
            file_hashes.append([path.name, sha256(path), label])
        meta = root / "cifar10" / (split + ".txt")
        meta.write_text("".join(records), encoding="utf-8")
        result[split] = {"count": count, "indices": indices, "meta": str(meta),
                         "image_dir": str(directory), "meta_sha256": sha256(meta),
                         "image_manifest_sha256": hashlib.sha256(
                             json.dumps(file_hashes, sort_keys=True).encode()).hexdigest()}
    result["split_policy"] = "Separate original train/test partitions; fixed seeded subset, no overlap."
    return result


def prepare_pennfudan(root, seed, url):
    import numpy as np
    from PIL import Image
    from torchvision.datasets.utils import download_and_extract_archive

    downloads = root / "downloads"
    directory = downloads / "PennFudanPed"
    if not directory.is_dir():
        download_and_extract_archive(url, str(downloads), filename="PennFudanPed.zip")
    images = sorted((directory / "PNGImages").glob("*.png"))
    if len(images) != 170:
        raise ValueError("Expected 170 PennFudan images, found {}".format(len(images)))
    indices = np.random.RandomState(seed).permutation(len(images)).tolist()
    partitions = {"train": indices[:-50], "test": indices[-50:]}
    result = {"dataset": "PennFudan", "source": "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip",
              "transport_url": url,
              "archive_sha256": sha256(downloads / "PennFudanPed.zip"),
              "split_policy": "Fixed image-level 120/50 holdout; no training on validation images."}
    output = root / "pennfudan"
    output.mkdir(parents=True, exist_ok=True)
    for split, selection in partitions.items():
        data = {"info": {"description": "PennFudan image-level holdout"}, "licenses": [],
                "images": [], "annotations": [],
                "categories": [{"id": 1, "name": "person", "supercategory": "person"}]}
        hashes = []
        for index in selection:
            path = images[index]
            with Image.open(path) as image:
                width, height = image.size
            mask_path = directory / "PedMasks" / (path.stem + "_mask.png")
            with Image.open(mask_path) as image:
                mask = np.asarray(image)
            data["images"].append({"id": index + 1, "file_name": path.name,
                                   "width": width, "height": height})
            for instance in np.unique(mask):
                if instance == 0:
                    continue
                ys, xs = np.where(mask == instance)
                x0, y0 = int(xs.min()), int(ys.min())
                box_width, box_height = int(xs.max()) - x0 + 1, int(ys.max()) - y0 + 1
                data["annotations"].append({
                    "id": len(data["annotations"]) + 1, "image_id": index + 1, "category_id": 1,
                    "bbox": [x0, y0, box_width, box_height], "area": box_width * box_height,
                    "iscrowd": 0, "segmentation": [],
                })
            hashes.append([path.name, sha256(path), sha256(mask_path)])
        annotation = output / (split + ".json")
        annotation.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
        result[split] = {"count": len(selection), "indices": selection,
                         "meta": str(annotation), "image_dir": str(directory / "PNGImages"),
                         "meta_sha256": sha256(annotation),
                         "image_manifest_sha256": hashlib.sha256(
                             json.dumps(hashes, sort_keys=True).encode()).hexdigest()}
    assert not set(partitions["train"]) & set(partitions["test"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--split-seed", type=int, default=20261021)
    parser.add_argument("--dataset", choices=("cifar10", "pennfudan", "all"), default="all")
    parser.add_argument("--cifar-source", choices=("original", "hf-mirror"), default="original")
    parser.add_argument("--pennfudan-url", default="https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip")
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / "data.json"
    report = json.loads(path.read_text()) if path.exists() else {}
    report["split_seed"] = args.split_seed
    if args.dataset in ("cifar10", "all"):
        report["cifar10"] = prepare_cifar(root, args.train_size, args.test_size, args.split_seed, args.cifar_source)
        (root / "data-cifar10.json").write_text(json.dumps(
            {"split_seed": args.split_seed, "cifar10": report["cifar10"]},
            indent=2, sort_keys=True), encoding="utf-8")
    if args.dataset in ("pennfudan", "all"):
        report["pennfudan"] = prepare_pennfudan(root, args.split_seed, args.pennfudan_url)
        import torch
        weights = root / "weights" / "resnet50-19c8e357.pth"
        weights.parent.mkdir(parents=True, exist_ok=True)
        if not weights.exists():
            torch.hub.download_url_to_file(
                "https://download.pytorch.org/models/resnet50-19c8e357.pth",
                str(weights), hash_prefix="19c8e357")
        if not sha256(weights).startswith("19c8e357"):
            raise ValueError("Unexpected pretrained weight checksum")
        normalized = weights.with_name("resnet50-19c8e357-state-dict.pth")
        convert_trusted_weights(weights, normalized)
        report["resnet50_weights"] = {"path": str(normalized), "sha256": sha256(normalized),
                                      "original_path": str(weights), "original_sha256": sha256(weights),
                                      "conversion": "legacy archive to tensor-only modern serialization; values checked identical"}
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"manifest": str(path), "sha256": sha256(path)}, indent=2))


if __name__ == "__main__":
    main()
