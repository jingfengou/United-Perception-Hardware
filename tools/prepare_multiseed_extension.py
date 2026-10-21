"""Add the native UP YOLO anchor asset to an existing accuracy-data manifest."""
import argparse
import hashlib
import json
from pathlib import Path


def add_anchors(source, data_path, output):
    data = json.loads(data_path.read_text())
    payload = (source / "configs/det/yolov5/anchors.json").read_bytes()
    json.loads(payload)
    target = output.resolve().parent / "yolov5-anchors.json"
    if target.exists() and target.read_bytes() != payload:
        raise ValueError("Refusing to replace a different anchor asset")
    data["yolov5_anchors"] = {"path": str(target), "sha256": hashlib.sha256(payload).hexdigest()}
    value = json.dumps(data, indent=2, sort_keys=True)
    if output.exists() and output.read_text() != value:
        raise ValueError("Use a new manifest path; existing data manifest differs")
    output.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    output.write_text(value)
    return data["yolov5_anchors"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(add_anchors(args.source, args.data, args.output), indent=2))


if __name__ == "__main__":
    main()
