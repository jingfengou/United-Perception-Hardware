"""Isolate vendor dropout scaling from model algorithms and random masks."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


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
    sys.path.insert(0, str(args.source.resolve()))
    import torch
    from up.accelerator import configure_accelerator
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(131)
    rows, tensors = [], {}
    for shape in ((100, 2, 256), (64, 17, 3072)):
        gradient = torch.randn(shape, generator=generator).cuda()
        for probability in (.1, .5, .9):
            mask = (torch.rand(shape, generator=generator) >= probability).cuda()
            scale = 1. / (1. - probability)
            configure_accelerator(backend=args.backend, reset=True)
            native = torch.ops.aten.native_dropout_backward.default(gradient, mask, scale)
            expected = (gradient.double() * mask * scale).float()
            configure_accelerator({"runtime": {"accelerator": {
                "backend": args.backend, "numerics": "portable_v1"}}}, reset=True)
            shared = torch.ops.aten.native_dropout_backward.default(gradient, mask, scale)
            assert torch.equal(shared, expected)
            name = str(shape) + "-" + str(probability)
            rows.append({"shape": shape, "probability": probability, "scale": scale,
                         "gradient_sha256": digest(gradient), "mask_sha256": digest(mask),
                         "native_sha256": digest(native), "shared_sha256": digest(shared),
                         "shared_equals_widened_gpu_reference": True,
                         "native_shared_max_abs": float((native - shared).abs().max()),
                         "native_shared_unequal": int((native != shared).sum())})
            tensors[name + "/native"] = native.cpu()
            tensors[name + "/shared"] = shared.cpu()
    report = {"status": "passed", "scope": "Rounding diagnostic, not a vendor correctness or accuracy verdict",
              "backend": args.backend, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
              "cases": rows, "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    torch.save(tensors, args.output / "tensors.pt")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
