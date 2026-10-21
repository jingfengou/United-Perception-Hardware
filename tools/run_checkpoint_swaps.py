"""Export our trusted model states, or evaluate identical states across vendors."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


def helpers():
    spec = importlib.util.spec_from_file_location("accuracy_helpers", Path(__file__).with_name("validate_training_accuracy.py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("nvidia", "metax"), required=True)
    parser.add_argument("--models", nargs="+", choices=("yolov5s", "detr", "vit_b16"), default=["yolov5s", "detr", "vit_b16"])
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--export", action="store_true", help="Only our own trusted full checkpoints, loaded on CPU")
    parser.add_argument("--results", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--numerical-profile", choices=("native", "portable_v1"), default="native")
    parser.add_argument("--origins", nargs="+", choices=("nvidia", "metax"), default=["nvidia", "metax"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    driver = helpers()
    if args.export:
        if not args.results:
            parser.error("Export requires --results")
        import torch
        args.weights.mkdir(parents=True, exist_ok=True)
        index = {}
        for model in args.models:
            run = args.results / (model + "-seed" + str(args.seed))
            report = json.loads((run / "report.json").read_text())
            if report["status"] != "completed" or report["backend"] != args.backend or report["model"] != model:
                raise ValueError("Only complete, correctly identified training states can be exported")
            checkpoint = torch.load(run / "comparison_state.pt", map_location="cpu", weights_only=False)
            if driver.state_hash(checkpoint["model"]) != report["final_state_sha256"]:
                raise ValueError("Training checkpoint changed")
            target = args.weights / (args.backend + "-" + model + ".pt")
            if target.exists():
                if driver.state_hash(torch.load(target, map_location="cpu", weights_only=True)) != report["final_state_sha256"]:
                    raise ValueError("Refusing to replace different exported weights")
            else:
                torch.save(checkpoint["model"], target)
            index[model] = {"model_state_sha256": report["final_state_sha256"],
                            "report_sha256": driver.file_hash(run / "report.json"),
                            "weights_sha256": driver.file_hash(target), "seed": args.seed}
        (args.weights / (args.backend + "-weights-index.json")).write_text(json.dumps(index, indent=2, sort_keys=True))
        return
    if not all((args.source, args.data, args.output)) or args.device < 0:
        parser.error("Inference requires source, data, output and a nonnegative physical device")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a new inference output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.device), PYTHONHASHSEED="0",
                       OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", DEFAULT_TASKS="cls:det",
                       PYTORCH_DISABLE_CUDA_CUDNN_TF32="1", TORCH_ALLOW_TF32_CUBLAS_OVERRIDE="0")
    result = {"scope": "diagnostic_only; frozen seed checkpoint inference, not independent training seeds", "cases": {}}
    for origin in args.origins:
        index = json.loads((args.weights / (origin + "-weights-index.json")).read_text())
        for model in args.models:
            name = origin + "-" + model + "-on-" + args.backend
            checkpoint = args.weights / (origin + "-" + model + ".pt")
            if driver.file_hash(checkpoint) != index[model]["weights_sha256"]:
                raise ValueError("Model-only weight transfer changed")
            command = [sys.executable, str(Path(__file__).with_name("probe_cross_vendor_numerics.py")),
                       "--kind", "inference", "--source", str(args.source.resolve()), "--data", str(args.data.resolve()),
                       "--model", model, "--backend", args.backend, "--checkpoint", str(checkpoint.resolve()),
                       "--numerical-profile", args.numerical_profile,
                       "--output", str((args.output / name).resolve())]
            with (args.output / (name + ".log")).open("w") as log:
                status = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
            if status.returncode:
                raise RuntimeError("Inference failed; partial evidence retained: " + name)
            report = json.loads((args.output / name / "report.json").read_text())
            if report["model_state_sha256"] != index[model]["model_state_sha256"]:
                raise ValueError("Evaluated model differs from its frozen training state")
            result["cases"][name] = report
            print(json.dumps({"completed_inference": name, "metrics": report["metrics"]}), flush=True)
    result["status"] = "completed"
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
