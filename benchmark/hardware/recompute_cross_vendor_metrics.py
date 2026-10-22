"""Recompute metrics from final predictions and held-out labels, without GPU inference."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def classification_metrics(predictions, meta):
    import torch

    labels = {name: int(label) for name, label in
              (line.rsplit(" ", 1) for line in meta.read_text().splitlines())}
    names = [row["filename"] for row in predictions]
    if len(names) != len(labels) or len(set(names)) != len(names) or set(names) != set(labels):
        raise ValueError("Missing, duplicate or extra held-out classification images")
    correct = {1: 0, 5: 0}
    for row in predictions:
        target = labels[row["filename"]]
        if row["label"] != target or row["prediction"] != row["topk_idx"][0]:
            raise ValueError("Saved predictions do not match held-out labels or their top1 index")
        if len(row["topk_idx"]) != 5 or len(set(row["topk_idx"])) != 5:
            raise ValueError("Invalid top5 prediction contract")
        for k in correct:
            correct[k] += int(target in row["topk_idx"][:k])
    # Match the original UP evaluator's CPU float32 count * (100 / N) exactly.
    metrics = {"top" + str(k): torch.tensor([count], dtype=torch.float32).mul_(100. / len(labels)).item()
               for k, count in correct.items()}
    return metrics, {"validation_images": len(labels), "correct": correct}


def detection_metrics(predictions, meta):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    truth = COCO(str(meta))
    for row in predictions:
        if row["image_id"] not in truth.imgs or row["category_id"] not in truth.cats:
            raise ValueError("Detection references an image/category outside the held-out set")
        if len(row["bbox"]) != 4 or not all(math.isfinite(float(value)) for value in [*row["bbox"], row["score"]]):
            raise ValueError("Non-finite or malformed detection")
    detected = truth.loadRes(predictions)
    evaluator = COCOeval(truth, detected, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    names = ("AP", "AP.5", "AP.75", "AP_S", "AP_M", "AP_L",
             "AR_1", "AR_10", "AR", "AR_S", "AR_M", "AR_L")
    return {"bbox." + name: float(value) for name, value in zip(names, evaluator.stats)}, {
        "validation_images": len(truth.imgs), "detections": len(predictions),
        "max_detections": evaluator.params.maxDets, "iou_thresholds": evaluator.params.iouThrs.tolist()}


def reevaluate(directory, data_root=None):
    report_path = directory / "report.json"
    report = json.loads(report_path.read_text())
    if report["status"] != "completed":
        raise ValueError("Only completed training runs are eligible")
    classification = report["model"] in ("resnet18", "vit_b16")
    meta = Path(report["data"]["test"]["meta"])
    if data_root is not None:
        meta = data_root / ("cifar10/test.txt" if classification else "pennfudan/test.json")
    if sha256(meta) != report["data"]["test"]["meta_sha256"]:
        raise ValueError("Held-out labels changed")
    prediction_path = directory / "predictions/results.txt"
    predictions = [json.loads(line) for line in prediction_path.read_text().splitlines()]
    metrics, details = (classification_metrics if classification else detection_metrics)(predictions, meta)
    result = {"status": "passed" if metrics == report["final_metrics"] else "failed",
              "report_sha256": sha256(report_path), "predictions_sha256": sha256(prediction_path),
              "ground_truth_sha256": sha256(meta), "reevaluator_sha256": sha256(__file__),
              "reevaluated_metrics": metrics, "reported_metrics": report["final_metrics"], "details": details}
    if details["validation_images"] != report["data"]["test"]["count"]:
        raise ValueError("Changed validation image count")
    (directory / "metric-reevaluation.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "passed":
        raise ValueError("Recomputed metrics differ from the Runner report: " + str(directory))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {str(path): reevaluate(path, args.data_root) for path in args.runs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps({"status": "passed", "runs": len(result), "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
