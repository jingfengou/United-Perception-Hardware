"""CPU tests for fail-closed timing and memory accounting; never start training."""
import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def benchmark(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmark/hardware"))
    return importlib.import_module("benchmark_metax_detr")


def test_positive_timing_distribution_and_nearest_rank_p95(benchmark):
    result = benchmark.distribution(list(range(1, 21)))
    assert result == {"count": 20, "mean": 10.5, "median": 10.5, "p95": 19, "sum": 210}
    for values in ([], [0.], [-1.], [float("nan")], [float("inf")]):
        with pytest.raises(ValueError):
            benchmark.distribution(values)


def test_evaluation_and_checkpoint_excluded_memory_reset(benchmark, monkeypatch):
    class Cuda:
        allocated = 100
        reserved = 200
        resets = 0

        def synchronize(self):
            pass

        def max_memory_allocated(self):
            return self.allocated

        def max_memory_reserved(self):
            return self.reserved

        def reset_peak_memory_stats(self):
            self.resets += 1
            self.allocated = 20
            self.reserved = 200

    now = [0.]
    monkeypatch.setattr(benchmark, "clock", lambda: now[0])
    cuda = Cuda()
    meter = benchmark.TrainingMeter(cuda)
    assert meter.outside_training("evaluation", lambda: "initial") == "initial"
    meter.begin()
    cuda.allocated = 150
    now[0] = 5.

    def evaluate():
        cuda.allocated = 999
        now[0] += 10.
        return {"AP": .5}

    assert meter.outside_training("evaluation", evaluate) == {"AP": .5}
    assert meter.peak_allocated == 150
    assert meter.excluded["evaluation"] == 10.
    assert cuda.resets == 2
    now[0] += 2.
    meter.finish()
    assert meter.training_seconds == 7.
    assert meter.peak_allocated == 150
    assert meter.outside_training("checkpoint", lambda: "after") == "after"
    assert meter.excluded["checkpoint"] == 0


def test_incomplete_budget_is_not_a_benchmark_result(benchmark):
    meter = benchmark.TrainingMeter(None)
    with pytest.raises(ValueError, match="720-update"):
        meter.report()


def test_schedule_is_fixed_abba_single_seed(benchmark):
    assert benchmark.ORDER == ("native-0", "v2-0", "v2-1", "native-1")
    assert (benchmark.STEPS, benchmark.WARMUP, benchmark.SEED, benchmark.EPOCHS) == (720, 60, 131, 12)


@pytest.fixture
def cohort(benchmark, tmp_path):
    for name in benchmark.ORDER:
        directory = tmp_path / name
        directory.mkdir()
        steps = [{"step": i + 1, "images": 2, "loss": 1., "lr": [1e-4]} for i in range(720)]
        (directory / "steps.json").write_text(json.dumps(steps))
        (directory / "final.pt").write_bytes(b"synthetic checkpoint for CPU accounting tests only")
        timing = {key: 1. for key in ("steady_wall_ms_per_update", "steady_images_per_second",
                                      "training_seconds_excluding_eval_save", "warmup_training_seconds")}
        timing.update(steady_steps=660, training_peak_allocated_bytes=128, training_peak_reserved_bytes=256)
        native = name.startswith("native")
        report = {"status": "completed", "seed": 131, "epochs": 12, "training_steps": 720,
                  "source_manifest_sha256": "source", "data_sha256": "data",
                  "algorithm_config_sha256": "config", "initial_model_sha256": "model",
                  "initial_optimizer_sha256": "optimizer", "initial_rng_sha256": "rng",
                  "first_batch_sha256": "input", "initial_metrics": {"bbox.AP": .5},
                  "driver_sha256": "driver", "final_model_sha256": "final",
                  "profile": benchmark.PROFILES[0 if native else 1],
                  "final_metrics": {"bbox.AP": .7 if native else .71},
                  "environment": {"deterministic_algorithms": True, "deterministic_warn_only": False,
                                  "visible_devices": "2", "triton_cache_dir": name},
                  "timing": timing,
                  "steps_sha256": benchmark.file_hash(directory / "steps.json"),
                  "checkpoint_sha256": benchmark.file_hash(directory / "final.pt"),
                  "numerics": {"profile": "native" if native else benchmark.PROFILES[1],
                               "operations": {} if native else {"aten.mm.default": 1,
                                                                 "aten.convolution_backward.weight": 1}}}
        (directory / "report.json").write_text(json.dumps(report))
    return tmp_path


def test_cohort_summary_preserves_results_without_declaring_equivalence(benchmark, cohort):
    path = cohort / "v2-0/report.json"
    report = json.loads(path.read_text())
    report["initial_metrics"] = {"bbox.AP": .50001}
    path.write_text(json.dumps(report))
    result = benchmark.summarize(cohort)
    assert result["modes"]["native"]["final_ap"] == [.7, .7]
    assert result["comparison"]["mean_ap_difference_v2_minus_native"] == pytest.approx(.01)
    assert result["metric_atol"] == 1e-6
    assert "passed" not in result


def test_cache_manifest_is_content_bound(benchmark, tmp_path):
    (tmp_path / "kernel").mkdir()
    (tmp_path / "kernel/metadata.json").write_text('{"compiled": true}')
    first = benchmark.cache_manifest(tmp_path)
    assert list(first) == ["kernel/metadata.json"]
    (tmp_path / "kernel/metadata.json").write_text('{"compiled": false}')
    assert benchmark.cache_manifest(tmp_path) != first


@pytest.mark.parametrize("mutation", ["config", "steps", "warn_only", "reserved", "environment", "missing_op"])
def test_cohort_rejects_incomparable_or_incomplete_evidence(benchmark, cohort, mutation):
    path = cohort / "v2-1/report.json"
    report = json.loads(path.read_text())
    if mutation == "config":
        report["algorithm_config_sha256"] = "different"
    elif mutation == "steps":
        report["training_steps"] = 719
    elif mutation == "warn_only":
        report["environment"]["deterministic_warn_only"] = True
    elif mutation == "reserved":
        report["timing"]["training_peak_reserved_bytes"] = 0
    elif mutation == "environment":
        report["environment"]["torch"] = "different"
    else:
        report["numerics"]["operations"]["aten.mm.default"] = 0
    path.write_text(json.dumps(report))
    with pytest.raises(AssertionError):
        benchmark.summarize(cohort)
