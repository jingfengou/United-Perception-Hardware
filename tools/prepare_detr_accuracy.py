"""Prepare a trusted, pinned official DETR checkpoint for the UP validation adapter."""
import argparse
import json
from pathlib import Path
import subprocess

from accuracy_detr_adapter import DETR_REVISION, file_hash, source_hash


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if commit != DETR_REVISION or subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip():
        raise ValueError("Use the clean, pinned official DETR revision " + DETR_REVISION)
    args.weights_dir.mkdir(parents=True, exist_ok=True)
    url = "https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth"
    original = args.weights_dir / "detr-r50-e632da11.pth"
    if not original.exists():
        torch.hub.download_url_to_file(url, str(original), hash_prefix="e632da11")
    if not file_hash(original).startswith("e632da11"):
        raise ValueError("Unexpected official DETR checksum")
    # Only this explicitly trusted, checksum-verified official file may use legacy loading.
    value = torch.load(original, map_location="cpu", weights_only=False)["model"]
    if not all(isinstance(item, torch.Tensor) for item in value.values()):
        raise ValueError("Expected an official tensor-only model state")
    normalized = args.weights_dir / "detr-r50-tensors.pt"
    torch.save(value, normalized)
    restored = torch.load(normalized, map_location="cpu", weights_only=True)
    if value.keys() != restored.keys() or not all(torch.equal(item, restored[key]) for key, item in value.items()):
        raise ValueError("Serialization conversion changed DETR weights")
    data = json.loads(args.data.read_text())
    data["base_manifest_sha256"] = file_hash(args.data)
    data["detr"] = {"source": str(source), "source_url": "https://github.com/facebookresearch/detr",
                    "revision": commit, "source_sha256": source_hash(source),
                    "checkpoint": str(normalized.resolve()), "checkpoint_sha256": file_hash(normalized),
                    "original_checkpoint": str(original.resolve()), "original_checkpoint_sha256": file_hash(original),
                    "checkpoint_url": url, "conversion": "tensor-only serialization; every tensor checked identical",
                    "adapter": "Same external adapter in both UP trees, single-device only, not an upstream UP model"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, sort_keys=True))
    print(json.dumps(data["detr"], indent=2))


if __name__ == "__main__":
    main()
