"""Check shared GPU FP64 kernels and full normalization autograd contracts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--backend", choices=("nvidia", "metax"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new evidence directory")
    args.output.mkdir(parents=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    sys.path.insert(0, str(args.source.resolve()))
    import torch
    import torch.nn.functional as F
    from up.accelerator import configure_accelerator
    from up.accelerator.backends.portable_math import matrix, sum_rows
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.enabled = False
    torch.backends.cuda.matmul.allow_tf32 = False
    generator = torch.Generator().manual_seed(131)

    def fixture(shape):
        return torch.randn(shape, generator=generator).double().cuda()

    rows = {}
    for shape in ((3, 257), (32, 1638400)):
        x = fixture(shape)
        output = sum_rows(x)
        reference = x.sum(dim=1)
        torch.testing.assert_close(output, reference, rtol=1e-12, atol=1e-10)
        assert torch.equal(output, sum_rows(x))
        rows["sum/" + str(shape)] = {"inputs": [digest(x)], "outputs": [digest(output)]}
    a, b = fixture((37, 65)), fixture((65, 97))
    output = matrix(a, b)
    torch.testing.assert_close(output, a @ b, rtol=1e-12, atol=1e-10)
    rows["matrix"] = {"inputs": [digest(a), digest(b)], "outputs": [digest(output)]}

    cases = [
        ("batch_norm", [fixture((4, 8, 5, 7)), fixture((8,)), fixture((8,))],
         lambda x, w, b: F.batch_norm(x, None, None, w, b, training=True, eps=1e-3)),
        ("layer_norm", [fixture((4, 17, 32)), fixture((32,)), fixture((32,))],
         lambda x, w, b: F.layer_norm(x, (32,), w, b, eps=1e-5)),
        ("softmax", [fixture((2, 8, 100, 100))], lambda x: F.softmax(x, dim=-1)),
        ("linear", [fixture((3, 17, 32)), fixture((96, 32)), fixture((96,))], F.linear),
        ("conv2d", [fixture((2, 4, 9, 11)), fixture((6, 4, 3, 3)), fixture((6,))],
         lambda x,w,b:F.conv2d(x,w,b,stride=2,padding=1)),
        ("grouped_conv2d", [fixture((2, 4, 9, 11)), fixture((6, 2, 3, 3)), fixture((6,))],
         lambda x,w,b:F.conv2d(x,w,b,padding=1,groups=2)),
        ("dilated_conv2d", [fixture((2, 4, 9, 11)), fixture((6, 4, 2, 2))],
         lambda x,w:F.conv2d(x,w,padding=1,dilation=2)),
        ("l1_distance", [fixture((3, 4)), fixture((7, 4))], lambda a,b:torch.cdist(a,b,p=1)),
        ("batched_l1_distance", [fixture((2, 3, 4)), fixture((1, 7, 4))], lambda a,b:torch.cdist(a,b,p=1)),
        ("l2_norm", [fixture((3, 4))], lambda x:torch.linalg.vector_norm(x)),
    ]
    saved = {}
    for name, values, operation in cases:
        reference_inputs = [value.clone().requires_grad_() for value in values]
        reference = operation(*reference_inputs)
        gradient = fixture(reference.shape)
        reference_grads = torch.autograd.grad(reference, reference_inputs, gradient)
        provider = configure_accelerator({"runtime": {"accelerator": {
            "backend": args.backend, "numerics": "portable_v1"}}}, reset=True)
        inputs = [value.clone().requires_grad_() for value in values]
        output = operation(*inputs)
        gradients = torch.autograd.grad(output, inputs, gradient)
        for actual, expected in zip((output, *gradients), (reference, *reference_grads)):
            assert actual.device.type == "cuda" and actual.dtype == torch.float64
            torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
        rows[name] = {"inputs": [digest(value) for value in (*values, gradient)],
                      "outputs": [digest(value) for value in (output, *gradients)],
                      "fp32_outputs": [digest(value.float()) for value in (output, *gradients)],
                      "max_abs_vs_native": [float((a - b).abs().max())
                                            for a, b in zip((output, *gradients), (reference, *reference_grads))]}
        saved.update({name + "/" + str(i): value.detach().cpu()
                      for i, value in enumerate((output, *gradients))})
        configure_accelerator(backend=args.backend, reset=True)
    source = fixture((1, 17, 768)).float().requires_grad_()
    upstream = fixture((64, 17, 768)).float()
    configure_accelerator({"runtime": {"accelerator": {
        "backend": args.backend, "numerics": "portable_v1"}}}, reset=True)
    gradient = torch.autograd.grad(source.expand_as(upstream), source, upstream)[0]
    from up.accelerator.backends.portable_math import reduce_sum
    expected = reduce_sum(upstream.double(), (0,), True).float()
    assert torch.equal(gradient, expected)
    oracle = upstream.detach().cpu().double().sum(dim=0, keepdim=True).float()
    assert torch.equal(gradient.cpu(), oracle)
    rows["fp32_expand_backward"] = {"inputs": [digest(source), digest(upstream)],
                                    "outputs": [digest(gradient)], "fp32_outputs": [digest(gradient)],
                                    "cpu_fp64_oracle_equal": True}
    norms = torch._foreach_norm([upstream, source.detach()], 2)
    norm_oracles = [value.detach().cpu().double().square().sum().sqrt().float()
                   for value in (upstream, source)]
    assert all(torch.equal(actual.cpu(), oracle) for actual, oracle in zip(norms, norm_oracles))
    rows["fp32_foreach_norm"] = {"inputs": [digest(upstream), digest(source)],
                                 "outputs": [digest(value) for value in norms],
                                 "fp32_outputs": [digest(value) for value in norms],
                                 "cpu_fp64_oracle_equal": True}
    configure_accelerator(backend=args.backend, reset=True)
    report = {"status": "passed", "backend": args.backend, "torch": torch.__version__,
              "gpu": torch.cuda.get_device_name(), "cases": rows,
              "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    torch.save(saved, args.output / "tensors.pt")
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
