"""CPU-only audit, paired summaries and an indexed DETR path evidence archive."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import tarfile

from benchmark_detr_paths import MODES, ORDER, UPDATES, compare
from diagnose_detr_origin import digest, write_json
from validate_platform_equivalence import file_hash


def validate_reports(reports):
    expected = {phase + "-" + mode + "-" + str(repeat)
                for phase in ("inference", "training") for mode, repeat in ORDER}
    if reports.keys() != expected:
        raise ValueError("All 18 fresh-process runs are required")
    reference = next(iter(reports.values()))
    invariants = ("seed", "fixture_sha256", "source", "initial_model", "initial_optimizer",
                  "torch", "torchvision", "python", "gpu", "visible_devices", "driver_sha256", "rtol", "atol")
    for name, row in reports.items():
        assert row["status"] == "completed" and row["criterion"] == "aligned", name
        assert name.startswith(row["phase"] + "-" + row["mode"] + "-")
        assert all(row[key] == reference[key] for key in invariants), name
        assert row["mode"] in MODES and row["phase"] in ("inference", "training")
        assert row["seed"] == 131 and row["rtol"] == 1e-4 and row["atol"] == 1e-5
        assert row["deterministic_algorithms"] and not row["warn_only"]
        assert row["up_imported"] == row["platform_imported"] == (row["mode"] != "official")
        assert row["timing"]["wall_ms"]["count"] == (UPDATES if row["phase"] == "training" else 150)
        assert row["timing"]["cache_unchanged"], name
        for key in ("wall_ms", "stream_ms"):
            assert all(math.isfinite(row["timing"][key][value]) and row["timing"][key][value] > 0
                       for value in ("mean", "median", "p95", "sum"))
        assert row["timing"]["peak_reserved_bytes"] >= row["timing"]["peak_allocated_bytes"] > 0
        assert len(row["losses"]) == (UPDATES if row["phase"] == "training" else 0)
        assert all(math.isfinite(value) for value in row["losses"])
        assert len(row["audits"]) == 3 and all(case["repeats"] == 10 for case in row["audits"])
        phase_reference = reports[row["phase"] + "-official-0"]
        assert [case["fixture_index"] for case in row["audits"]] == [
            case["fixture_index"] for case in phase_reference["audits"]]
        if row["mode"] == "up_v2":
            assert row["numerics"]["profile"] == "metax_deterministic_v2"
            assert row["numerics"]["operations"]["aten.mm.default"] > 0
            if row["phase"] == "training":
                assert row["numerics"]["operations"]["aten.convolution_backward.weight"] > 0
        else:
            assert row["numerics"]["profile"] == "native" and not row["numerics"]["operations"]
    return {key: reference[key] for key in invariants}


def performance(reports):
    phases = {}
    for phase in ("inference", "training"):
        modes = {}
        for mode in MODES:
            selected = [reports[phase + "-" + mode + "-" + str(i)] for i in range(3)]
            metrics = {name: [row["timing"][name] for row in selected]
                       for name in ("images_per_second", "peak_allocated_bytes", "peak_reserved_bytes",
                                    "baseline_allocated_bytes", "baseline_reserved_bytes", "warmup_seconds")}
            metrics["wall_ms"] = [row["timing"]["wall_ms"]["mean"] for row in selected]
            metrics["stream_ms"] = [row["timing"]["stream_ms"]["mean"] for row in selected]
            modes[mode] = {name: {"values": values, "mean": statistics.mean(values),
                                  "min": min(values), "max": max(values)} for name, values in metrics.items()}
        phases[phase] = {"modes": modes, "ratios": {}}
        for before, after in (("official", "up_native"), ("up_native", "up_v2")):
            phases[phase]["ratios"][after + "_over_" + before] = {
                key: modes[after][key]["mean"] / modes[before][key]["mean"]
                for key in ("wall_ms", "stream_ms", "images_per_second", "peak_allocated_bytes", "peak_reserved_bytes")}
    return phases


def collect(args):
    import torch
    assert not torch.cuda.is_available(), "Run this audit with CUDA_VISIBLE_DEVICES empty"
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.root
    results = root / "results"
    reports = {phase + "-" + mode + "-" + str(repeat): json.loads(
        (results / (phase + "-" + mode + "-" + str(repeat)) / "report.json").read_text())
        for phase in ("inference", "training") for mode, repeat in ORDER}
    launch = json.loads((root / "launch.json").read_text())
    execution = json.loads((results / "execution.json").read_text())
    assert launch["status"] == execution["status"] == "completed"
    assert len(execution["jobs"]) == 18 and all(job["exit_code"] == 0 for job in execution["jobs"])
    invariants = validate_reports(reports)
    manifest = json.loads((root / "fixtures/manifest.json").read_text())
    assert file_hash(root / "fixtures/manifest.json") == invariants["fixture_sha256"]
    assert manifest["source"]["sha256"] == invariants["source"]
    assert file_hash(root / "source.tar") == launch["source_archive_sha256"]
    checks = {}
    checkpoint_index = {}
    for name, report in reports.items():
        directory = results / name
        files = [directory / "final.pt"] + [directory / ("audit-" + str(i) + ".pt") for i in range(3)]
        for path in files:
            checkpoint_index[path.relative_to(root).as_posix()] = {"bytes": path.stat().st_size, "sha256": file_hash(path)}
        state = torch.load(directory / "final.pt", map_location="cpu", weights_only=False)
        assert digest(state["model"]) == report["final_model"]
        assert digest(state["optimizer"]) == report["final_optimizer"]
        for case in range(3):
            observation = torch.load(directory / ("audit-" + str(case) + ".pt"), map_location="cpu", weights_only=False)
            assert digest(observation) == report["audits"][case]["reference_sha256"]
            assert compare(observation, observation)["exact"]
    for phase in ("inference", "training"):
        comparisons = [("official-0", mode + "-" + str(i)) for mode in MODES for i in range(3)]
        comparisons += [(mode + "-0", mode + "-" + str(i)) for mode in MODES for i in (1, 2)]
        for before, after in comparisons:
            key = phase + ":" + before + ":" + after
            left, right = results / (phase + "-" + before), results / (phase + "-" + after)
            rows = []
            for case in range(3):
                a = torch.load(left / ("audit-" + str(case) + ".pt"), map_location="cpu", weights_only=False)
                b = torch.load(right / ("audit-" + str(case) + ".pt"), map_location="cpu", weights_only=False)
                rows.append({name: compare(value, b[name]) for name, value in a.items()})
            final_left = torch.load(left / "final.pt", map_location="cpu", weights_only=False)
            final_right = torch.load(right / "final.pt", map_location="cpu", weights_only=False)
            tensor_states = [{"model": state["model"],
                              "optimizer": state["optimizer"]["state"] if state["optimizer"] else None}
                             for state in (final_left, final_right)]
            groups = [state["optimizer"]["param_groups"] if state["optimizer"] else None
                      for state in (final_left, final_right)]
            parts = {name: compare(tensor_states[0][name], tensor_states[1][name]) for name in tensor_states[0]}
            final_comparison = {"exact": all(part["exact"] for part in parts.values()),
                                "outside_tolerance": sum(part["outside_tolerance"] for part in parts.values()),
                                "max_abs": max(part["max_abs"] for part in parts.values()), **parts}
            final_comparison["parameter_groups_exact"] = digest(groups[0]) == digest(groups[1])
            final_comparison["exact"] &= final_comparison["parameter_groups_exact"]
            checks[key] = {"fixed_weight_cases": rows, "final_state": final_comparison}
            if phase == "training":
                checks[key]["losses"] = compare(torch.tensor(reports[phase + "-" + before]["losses"], dtype=torch.float64),
                                               torch.tensor(reports[phase + "-" + after]["losses"], dtype=torch.float64))
    control_log = (root / "official-loss-control.log").read_text()
    control = {"exit_code": launch["official_loss_control_exit_code"],
               "deterministic_nll_rejected": "does not have a deterministic implementation" in control_log}
    result = {"scope": "one C500, seed 131; 3 processes/path/phase; fixed-weight probes and 120 updates, not full convergence",
              "invariants": invariants, "performance": performance(reports), "numerical_comparisons": checks,
              "official_unmodified_loss_control": control, "runs": reports,
              "checkpoint_index": checkpoint_index, "collector_sha256": file_hash(Path(__file__)),
              "audited_at_utc": datetime.now(timezone.utc).isoformat()}
    write_json(args.output / "summary.json", result)
    members = [root / "source.tar", Path(__file__).resolve(), root / "launch.py", root / "launch.json", root / "fixtures.log",
               root / "fixtures/manifest.json", root / "official-loss-control.log", results / "execution.json"]
    members += sorted(results.glob("*.log")) + sorted(results.glob("*/report.json"))
    index = {path.relative_to(root).as_posix(): {"bytes": path.stat().st_size, "sha256": file_hash(path)} for path in members}
    write_json(args.output / "checkpoint-index.json", checkpoint_index)
    write_json(args.output / "member-index.json", index)
    archive = args.output / "detr-paths-evidence.tar.gz"
    with tarfile.open(archive, "x:gz") as stream:
        for path in members:
            stream.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
        for name in ("summary.json", "checkpoint-index.json", "member-index.json"):
            stream.add(args.output / name, arcname="audit/" + name, recursive=False)
    write_json(args.output / "archive.json", {"path": archive.name, "bytes": archive.stat().st_size,
               "sha256": file_hash(archive), "members": len(members) + 3,
               "tensors_retained_on_server_not_in_archive": True})
    print(json.dumps({"status": "completed", "archive": file_hash(archive), "performance": result["performance"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    collect(parser.parse_args())


if __name__ == "__main__":
    main()
