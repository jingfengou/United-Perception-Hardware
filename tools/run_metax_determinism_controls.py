"""Sequential, bounded C500 diagnosis. Never starts the paused accuracy cohort."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from validate_platform_equivalence import file_hash, prepare_subset, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("replay", "candidate-replay", "dnn-controls", "fixed-controls"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    tool = Path(__file__).with_name("diagnose_metax_determinism.py")
    summary = {"stage": args.stage, "jobs": [], "tool_sha256": file_hash(tool)}

    def execute(command, output):
        if output.exists():
            raise FileExistsError("Preserve existing evidence: " + str(output))
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.with_suffix(".log").open("w") as log:
            code = subprocess.run([sys.executable, str(tool), *command, "--output", str(output)],
                                  stdout=log, stderr=subprocess.STDOUT, cwd=args.source).returncode
        summary["jobs"].append({"command": command, "output": str(output), "exit_code": code})
        write_json(args.root / (args.stage + "-controller.json"), summary)
        print(output, "PASS" if not code else "FAIL", flush=True)
        if code:
            raise RuntimeError("Child failed; retained output/log: " + str(output))

    if args.stage in ("replay", "candidate-replay"):
        for model in ("resnet18", "yolov5s", "detr"):
            folder = args.root / "capture" / model
            for case in sorted(folder.glob("case-*.pt")):
                command = ["replay", "--case", str(case)]
                if args.stage == "candidate-replay":
                    command += ["--source", str(args.source), "--profile", "metax_deterministic_v1"]
                    from argparse import Namespace
                    from diagnose_metax_determinism import replay
                    output = args.root / args.stage / model / case.stem
                    output.mkdir(parents=True, exist_ok=False)
                    report = replay(Namespace(case=case, source=args.source, repeats=10, profile="metax_deterministic_v1"))
                    report["tool_sha256"] = file_hash(tool)
                    write_json(output / "report.json", report)
                    summary["jobs"].append({"case": str(case), "output": str(output), "exit_code": 0})
                    write_json(args.root / (args.stage + "-controller.json"), summary)
                else:
                    execute(command, args.root / args.stage / model / case.stem)
    elif args.stage == "dnn-controls":
        data = prepare_subset(args.data, args.root / "data")
        for model in ("resnet18", "yolov5s", "detr"):
            for repeat in (0, 1):
                execute(["training", "--source", str(args.source), "--data", str(data),
                         "--model", model, "--disable-dnn"], args.root / "dnn-controls" / model / str(repeat))
            execute(["compare", "--first", str(args.root / "dnn-controls" / model / "0"),
                     "--second", str(args.root / "dnn-controls" / model / "1")],
                    args.root / "dnn-controls" / model / "comparison")
    else:
        if args.baseline is None:
            parser.error("fixed-controls requires --baseline")
        data = prepare_subset(args.data, args.root / "fixed-v1/data")
        for model in ("resnet18", "yolov5s", "detr"):
            entries = [("baseline", args.baseline), ("changed-0", args.source), ("changed-1", args.source)]
            for label, source in entries:
                training = args.root / "fixed-v1" / model / label / "training"
                common = ["--source", str(source), "--data", str(data), "--model", model,
                          "--profile", "metax_deterministic_v1", "--adaptation-source", str(args.source)]
                execute(["training", *common], training)
                execute(["inference", *common, "--training", str(training)], training.parent / "inference")
            for label in ("changed-0", "changed-1"):
                execute(["compare", "--first", str(args.root / "fixed-v1" / model / "baseline/training"),
                         "--second", str(args.root / "fixed-v1" / model / label / "training")],
                        args.root / "fixed-v1" / model / ("compare-" + label))
    summary["completed"] = True
    write_json(args.root / (args.stage + "-controller.json"), summary)


if __name__ == "__main__":
    main()
