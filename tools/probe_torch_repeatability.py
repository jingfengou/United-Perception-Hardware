"""Pure-PyTorch forward/backward repeatability controls; no UP or vendor routing."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--disable-dnn", action="store_true", help="Diagnostic same-GPU convolution-provider ablation")
    parser.add_argument("--cases", nargs="+", choices=("conv2d", "batch_norm", "layer_norm", "linear", "softmax", "gelu"))
    args = parser.parse_args()
    if args.repeats < 2 or args.output.exists():
        parser.error("Need at least two repetitions and a new output file")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if args.disable_dnn:
        torch.backends.cudnn.enabled = False
    generator = torch.Generator(device="cpu").manual_seed(131)

    def normal(*shape):
        return torch.randn(shape, generator=generator).cuda()

    def digest(t):
        return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

    specs = [
        ("conv2d", [normal(2, 64, 64, 96), normal(128, 64, 3, 3)],
         lambda x, w: F.conv2d(x, w, padding=1)),
        ("batch_norm", [normal(64, 32, 40, 40), normal(32), normal(32)],
         lambda x, w, b: F.batch_norm(x, None, None, w, b, training=True)),
        ("layer_norm", [normal(64, 17, 768), normal(768), normal(768)],
         lambda x, w, b: F.layer_norm(x, (768,), w, b)),
        ("linear", [normal(64, 17, 768), normal(3072, 768)], F.linear),
        ("softmax", [normal(64, 12, 17, 17)], lambda x: F.softmax(x, dim=-1)),
        ("gelu", [normal(64, 17, 3072)], F.gelu),
    ]
    report = {"scope": "diagnostic_only; fixed tensors, repeated fresh autograd graphs, no optimizer or UP",
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
              "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "repetitions": args.repeats, "tf32": False, "deterministic_requested": True,
              "convolution_library_enabled": torch.backends.cudnn.enabled,
              "convolution_deterministic": torch.backends.cudnn.deterministic,
              "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "cases": {}}
    for name, tensors, operation in specs:
        if args.cases and name not in args.cases:
            continue
        tensors = [t.requires_grad_(True) for t in tensors]
        with torch.no_grad():
            sample = operation(*tensors)
            upstream = normal(*sample.shape)
        reference = None
        rows = []
        for _ in range(args.repeats):
            y = operation(*tensors)
            gradients = torch.autograd.grad(y, tensors, upstream)
            observed = [y.detach().cpu(), *[g.detach().cpu() for g in gradients]]
            if reference is None:
                reference = [t.clone() for t in observed]
            rows.append({"hashes": [digest(t) for t in observed],
                         "max_abs_vs_first": [float((a.double() - b.double()).abs().max())
                                              for a, b in zip(observed, reference, strict=True)]})
        report["cases"][name] = {"input_hashes": [digest(t) for t in tensors],
                                 "upstream_hash": digest(upstream), "rows": rows,
                                 "bitwise_repeatable": all(row["hashes"] == rows[0]["hashes"] for row in rows)}
        print(json.dumps({"case": name, "repeatable": report["cases"][name]["bitwise_repeatable"]}), flush=True)
    report["status"] = "completed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
