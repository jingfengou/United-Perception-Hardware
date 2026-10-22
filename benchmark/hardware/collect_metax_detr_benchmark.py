"""CPU-only integrity, metric recomputation and evidence packaging for native/v2."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile

from benchmark_metax_detr import ORDER, summarize
from diagnose_detr_origin import digest
from prepare_cross_vendor_data import verify_data
from recompute_cross_vendor_metrics import detection_metrics
from validate_platform_equivalence import file_hash, source_manifest, write_json


def audit(root, cohort):
    import torch
    execution = json.loads((root / cohort / "execution.json").read_text())
    assert execution["status"] == "completed"
    assert [job["name"] for job in execution["jobs"]] == list(ORDER)
    assert all(job["exit_code"] == 0 for job in execution["jobs"])
    assert source_manifest(root / "source") == execution["source"]
    data_path = Path(execution["jobs"][0]["command"][execution["jobs"][0]["command"].index("--data") + 1])
    assert file_hash(data_path) == execution["data_sha256"]
    data = json.loads(data_path.read_text())
    verify_data(data)
    summary = summarize(root / cohort)
    audits = {}
    for name in ORDER:
        folder = root / cohort / name
        report = summary["runs"][name]
        checkpoint = torch.load(folder / "final.pt", map_location="cpu", weights_only=False)
        assert digest(checkpoint["model"]) == report["final_model_sha256"]
        assert digest(checkpoint["optimizer"]) == report["final_optimizer_sha256"]

        def finite(value):
            if isinstance(value, torch.Tensor):
                assert torch.isfinite(value).all()
            elif isinstance(value, dict):
                for item in value.values():
                    finite(item)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    finite(item)

        finite(checkpoint)
        predictions_path = folder / "predictions/results.txt"
        predictions = [json.loads(line) for line in predictions_path.read_text().splitlines()]
        meta = Path(data["pennfudan"]["test"]["meta"])
        assert file_hash(meta) == data["pennfudan"]["test"]["meta_sha256"]
        metrics, details = detection_metrics(predictions, meta)
        deltas = {key: abs(value - report["final_metrics"][key]) for key, value in metrics.items()}
        assert details["validation_images"] == 50 and max(deltas.values()) <= 1e-6
        audits[name] = {"checkpoint_verified": True, "checkpoint_finite": True,
                        "detr_core_sha256": digest({key.removeprefix("detr.detector."): value
                                                    for key, value in checkpoint["model"].items()
                                                    if key.startswith("detr.detector.")}),
                        "final_predictions_sha256": file_hash(predictions_path), "metric_recomputation": metrics,
                        "maximum_metric_absolute_difference": max(deltas.values()), "details": details}
    result = {"integrity_verified": True, "source_manifest_sha256": execution["source"]["sha256"],
              "data_sha256": execution["data_sha256"], "metric_atol": 1e-6, "runs": audits}
    write_json(root / ("cpu-audit-" + cohort + ".json"), result)
    return result


def package(root, output):
    index_path = Path(str(output) + ".index.json")
    if output.exists() or index_path.exists():
        raise FileExistsError("Preserve the previous evidence archive")
    verification = {cohort: audit(root, cohort) for cohort in ("results", "warm-results")}
    shutil.copyfile(root / "source/benchmark/hardware/benchmark_metax_detr.py", root / "executed-driver.py")
    index = {"integrity_verified": all(value["integrity_verified"] for value in verification.values()),
             "archived_files": {}, "retained_checkpoints": {}}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any("triton-cache" in part or part == "__pycache__" for part in path.parts):
            continue
        name = path.relative_to(root).as_posix()
        if name.startswith("source/") or name in ("collect.log", "postprocess.log", "postprocess-status.json"):
            continue
        if path.suffix in (".pt", ".pth"):
            category = "retained_checkpoints"
        elif path.suffix in (".json", ".jsonl", ".log", ".txt", ".py", ".tar"):
            category = "archived_files"
        else:
            continue
        index[category][name] = {"bytes": path.stat().st_size, "sha256": file_hash(path)}
    payload = json.dumps(index, indent=2, sort_keys=True).encode()
    with tarfile.open(output, "x:gz") as archive:
        for name in index["archived_files"]:
            archive.add(root / name, arcname=name, recursive=False)
        member = tarfile.TarInfo("evidence-index.json")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    with tarfile.open(output) as archive:
        assert len(archive.getmembers()) == len(index["archived_files"]) + 1
        for member in archive.getmembers():
            contents = archive.extractfile(member).read()
            if member.name == "evidence-index.json":
                assert contents == payload
            else:
                record = index["archived_files"][member.name]
                assert len(contents) == record["bytes"] and hashlib.sha256(contents).hexdigest() == record["sha256"]
    index["archive"] = {"name": output.name, "bytes": output.stat().st_size, "sha256": file_hash(output)}
    write_json(index_path, index)
    return index["archive"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.root.resolve(), args.output.resolve()), indent=2))
