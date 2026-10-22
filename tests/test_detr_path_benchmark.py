"""CPU-only checks for the three-path performance/numerics protocol."""
import ast
import copy
import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def benchmark(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmark/hardware"))
    return importlib.import_module("benchmark_detr_paths")


def test_balanced_order_and_single_seed(benchmark):
    assert benchmark.ORDER == (("official", 0), ("up_native", 0), ("up_v2", 0),
                               ("up_native", 1), ("up_v2", 1), ("official", 1),
                               ("up_v2", 2), ("official", 2), ("up_native", 2))
    assert (benchmark.SEED, benchmark.WARMUP, benchmark.UPDATES, benchmark.REPEATS) == (131, 60, 120, 10)


def test_comparison_keeps_exact_and_tolerance_separate(benchmark):
    torch = pytest.importorskip("torch")
    first = {"logits": torch.tensor([1., 2.]), "aux": [torch.tensor([]), None]}
    second = {"logits": torch.tensor([1.00001, 2.]), "aux": [torch.tensor([]), None]}
    assert benchmark.compare(first, first) == {"exact": True, "outside_tolerance": 0, "max_abs": 0.}
    result = benchmark.compare(first, second)
    assert not result["exact"] and result["outside_tolerance"] == 0 and result["max_abs"] > 0
    second["logits"][0] = 1.1
    assert benchmark.compare(first, second)["outside_tolerance"] == 1


@pytest.mark.parametrize("failure", ["shape", "dtype", "nan", "keys", "length"])
def test_comparison_rejects_invalid_evidence(benchmark, failure):
    torch = pytest.importorskip("torch")
    if failure == "shape":
        first, second = torch.zeros(1), torch.zeros(2)
    elif failure == "dtype":
        first, second = torch.zeros(1), torch.zeros(1, dtype=torch.float64)
    elif failure == "nan":
        first, second = torch.zeros(1), torch.tensor([float("nan")])
    elif failure == "keys":
        first, second = {"a": None}, {"b": None}
    else:
        first, second = [None], []
    with pytest.raises(ValueError):
        benchmark.compare(first, second)


def test_measured_path_does_not_restore_rng_inside_timers(benchmark):
    tree = ast.parse(Path(benchmark.__file__).read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "execute"]
    measured = [node for node in calls if any(keyword.arg == "restore" and isinstance(keyword.value, ast.Constant)
                                             and keyword.value.value is False for keyword in node.keywords)]
    assert len(measured) == 1
    assert "BlockUP(platform=True)" in Path(benchmark.__file__).read_text()


def test_outputs_refuse_overwriting_existing_directories(benchmark):
    text = Path(benchmark.__file__).read_text()
    assert text.count("args.output.mkdir(parents=True, exist_ok=False)") == 3
    assert 'warn_only=True' not in text
    assert '"rtol": 1e-4, "atol": 1e-5' in text


@pytest.fixture
def collector(benchmark):
    return importlib.import_module("collect_detr_paths")


@pytest.fixture
def reports(benchmark):
    result = {}
    for phase in ("inference", "training"):
        for mode, repeat in benchmark.ORDER:
            native = mode != "up_v2"
            row = {"status": "completed", "criterion": "aligned", "seed": 131, "fixture_sha256": "fixture",
                   "source": "source", "initial_model": "model", "initial_optimizer": "optimizer", "torch": "torch",
                   "torchvision": "torchvision", "python": "python", "gpu": "C500", "visible_devices": "2",
                   "driver_sha256": "driver", "rtol": 1e-4, "atol": 1e-5, "mode": mode, "phase": phase,
                   "deterministic_algorithms": True, "warn_only": False,
                   "up_imported": mode != "official", "platform_imported": mode != "official",
                   "timing": {"cache_unchanged": True, "wall_ms": {"count": 120 if phase == "training" else 150,
                              "mean": 2 + repeat, "median": 2 + repeat, "p95": 4., "sum": 400.},
                              "stream_ms": {"mean": 2., "median": 2., "p95": 4., "sum": 400.},
                              "peak_allocated_bytes": 100, "peak_reserved_bytes": 200, "baseline_allocated_bytes": 50,
                              "baseline_reserved_bytes": 200, "warmup_seconds": 1., "images_per_second": 10.},
                   "losses": [1.] * 120 if phase == "training" else [],
                   "audits": [{"repeats": 10, "fixture_index": i} for i in range(3)],
                   "numerics": {"profile": "native" if native else "metax_deterministic_v2",
                                "operations": {} if native else {"aten.mm.default": 1, "aten.convolution_backward.weight": 1}}}
            result[phase + "-" + mode + "-" + str(repeat)] = row
    return result


def test_complete_cohort_and_mean_not_best_performance(collector, reports):
    assert collector.validate_reports(reports)["visible_devices"] == "2"
    result = collector.performance(reports)
    assert result["training"]["modes"]["official"]["wall_ms"] == {"values": [2, 3, 4], "mean": 3., "min": 2, "max": 4}
    assert result["inference"]["ratios"]["up_v2_over_up_native"]["wall_ms"] == 1.
    assert "passed" not in result


@pytest.mark.parametrize("failure", ["missing", "profile", "source", "criterion", "seed", "warn_only", "cache", "losses", "memory", "up_imported", "fixture"])
def test_summary_rejects_incomplete_or_drifted_cohort(collector, reports, failure):
    reports = copy.deepcopy(reports)
    row = reports["training-up_v2-1"]
    if failure == "missing":
        reports.pop("inference-official-0")
    elif failure == "profile":
        row["numerics"]["operations"]["aten.mm.default"] = 0
    elif failure == "cache":
        row["timing"]["cache_unchanged"] = False
    elif failure == "losses":
        row["losses"] = [float("nan")] * 120
    elif failure == "memory":
        row["timing"]["peak_reserved_bytes"] = 1
    elif failure == "fixture":
        row["audits"][1]["fixture_index"] = 999
    elif failure in ("warn_only", "up_imported"):
        row[failure] = failure == "warn_only"
    else:
        row[failure] = "changed"
    with pytest.raises((ValueError, AssertionError)):
        collector.validate_reports(reports)


def test_published_report_preserves_failed_equivalence_and_scope():
    root = Path(__file__).resolve().parents[1]
    row = json.loads((root / "docs/reports/detr-paths-20261023.json").read_text())
    assert len(row["runs"]) == 18
    assert row["acceptance"]["official_vs_up_native_inference_exact"]
    assert row["acceptance"]["v2_training_independent_processes_exact"]
    assert not row["acceptance"]["native_vs_v2_inference_within_original_tolerance"]
    assert not row["acceptance"]["native_training_independent_processes_exact"]
    assert not row["acceptance"]["full_accuracy_equivalence_verified"]
    assert row["official_unmodified_loss_control"] == {"exit_code": 1, "deterministic_nll_rejected": True}
    assert row["invariants"]["rtol"] == 1e-4 and row["invariants"]["atol"] == 1e-5
