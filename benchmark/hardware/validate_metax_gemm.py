"""Validate the C500 fixed FP32 GEMM using the actual DETR case and autograd."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from diagnose_detr_origin import (ATOL, RTOL, cpu, digest, new_output, restore_layout,
                                  setup, tensor_comparison, write_json)


def run(args):
    import torch
    sys.path.insert(0, str(args.source.resolve() / "packages/up-platform/src"))
    from up_platform.backends.metax.determinism_v2 import DeterministicGemm
    setup()
    output = new_output(args.output)
    case = torch.load(args.case, map_location="cpu", weights_only=False)
    if case["operation"] != "aten.mm.default" or case["mutable"]:
        raise ValueError("Expected the immutable, actual DETR mm case")
    operands, kwargs = restore_layout((case["args"], case["kwargs"]), case["layout"])
    assert digest((operands, kwargs)) == case["input_sha256"]
    a, b = operands
    reference = (a.cpu().double() @ b.cpu().double()).float()
    first, rows = None, []
    started = time.monotonic()
    mode = DeterministicGemm()
    with mode:
        for _ in range(args.repeats):
            result = torch.ops.aten.mm.default(*operands, **kwargs).cpu()
            if first is None:
                first = result
                torch.save(first, output / "first-result.pt")
            rows.append(tensor_comparison(first, result))
    case_report = {"rows": rows, "reference": tensor_comparison(reference, first),
                   "exact": all(row["exact"] for row in rows), "repeats": args.repeats,
                   "seconds": time.monotonic() - started, "operations": mode.counts}
    assert digest((operands, kwargs)) == case["input_sha256"]
    assert case_report["exact"] and case_report["reference"]["outside_tolerance"] == 0
    contracts = []
    generator = torch.Generator().manual_seed(131)
    # Include rectangular, transposed, sliced, zero-stride and empty reductions.
    for m, k, n in ((4, 600, 256), (17, 33, 65), (256, 128, 64), (0, 7, 5), (3, 0, 4), (3, 4, 0)):
        for layout in ("contiguous", "transposed", "sliced", "expanded"):
            left = torch.randn(m, k, generator=generator) * .1
            right = torch.randn(k, n, generator=generator) * .1
            if layout == "transposed":
                left, right = left.t().contiguous().t(), right.t().contiguous().t()
            elif layout == "sliced":
                left = torch.nn.functional.pad(left, (0, k))[:, ::2]
                right = torch.nn.functional.pad(right, (0, n))[:, ::2]
            elif layout == "expanded" and m and k and n:
                left = left[:1].expand(m, k)
                right = right[:, :1].expand(k, n)
            # Tensor.to may densify slices/expands; replay the real strides instead.
            def transfer(value):
                metadata = {"device": "cuda", "shape": list(value.shape), "stride": list(value.stride())}
                actual = restore_layout(value, metadata).requires_grad_()
                assert actual.stride() == value.stride()
                return actual
            x, y = transfer(left), transfer(right)
            expected_x, expected_y = left.double().requires_grad_(), right.double().requires_grad_()
            gradient = torch.randn(m, n, generator=generator) * .1
            reference_output = expected_x @ expected_y
            expected = torch.autograd.grad(reference_output, (expected_x, expected_y), gradient.double())
            with DeterministicGemm():
                actual = torch.mm(x, y)
                gradients = torch.autograd.grad(actual, (x, y), gradient.to("cuda"))
            comparisons = [tensor_comparison(ref.float(), value.detach().cpu())
                           for ref, value in zip((reference_output, *expected), (actual, *gradients), strict=True)]
            assert all(row["outside_tolerance"] == 0 for row in comparisons)
            contracts.append({"shape": [m, k, n], "layout": layout, "comparisons": comparisons,
                              "input_strides": [list(x.stride()), list(y.stride())], "strides_preserved": True})
    assert not any(name == "up" or name.startswith("up.") for name in sys.modules)
    report = {"passed": True, "case": case_report, "contracts": contracts,
              "input_sha256": case["input_sha256"], "rtol": RTOL, "atol": ATOL,
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
              "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "precision": "GPU FP32, IEEE dot; CPU FP64 is operator audit reference only"}
    write_json(output / "report.json", report)
    print(json.dumps({key: report[key] for key in ("passed", "torch", "gpu", "input_sha256")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("At least two repetitions are required")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    run(args)


if __name__ == "__main__":
    main()
