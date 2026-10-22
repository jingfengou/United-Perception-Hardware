"""GPU-only cross-vendor numerical-profile contract and primitive comparison."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--backend", choices=("nvidia", "metax"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory")
    args.output.mkdir(parents=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    sys.path.insert(0, str(args.source.resolve()))
    import torch
    import torch.nn.functional as F
    from up.accelerator import configure_accelerator
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    accelerator = configure_accelerator({"runtime": {"accelerator": {
        "backend": args.backend, "numerics": "portable_v1"}}}, reset=True)
    accelerator.manual_seed(131)
    tensors, cases = {}, {}
    state = accelerator.get_rng_state()
    a = torch.rand(64, 17, 3072, device="cuda")
    accelerator.set_rng_state(state)
    assert torch.equal(a, torch.rand_like(a))
    tensors["wide_uniform"] = a.cpu()
    state = accelerator.get_rng_state()
    x = torch.ones(64, 17, 3072, device="cuda", requires_grad=True)
    y = F.dropout(x, .1, True)
    y.sum().backward()
    accelerator.set_rng_state(state)
    assert torch.equal(y, F.dropout(x, .1, True))
    assert torch.allclose(x.grad, (y != 0).float() / .9)
    tensors["dropout"] = y.detach().cpu()
    tensors["dropout_gradient"] = x.grad.cpu()
    cases["rng_restore_and_dropout_gradient"] = "passed"
    g = torch.Generator(device="cpu").manual_seed(131)
    def normal(*shape):
        return torch.randn(shape, generator=g).cuda().requires_grad_()
    specs = [
        ("conv2d", [normal(2,64,64,96), normal(128,64,3,3)], lambda x,w: F.conv2d(x,w,padding=1)),
        ("linear", [normal(64,17,768), normal(3072,768)], F.linear),
        ("batch_norm", [normal(64,32,40,40), normal(32), normal(32)], lambda x,w,b: F.batch_norm(x,None,None,w,b,training=True)),
        ("layer_norm", [normal(64,17,768), normal(768), normal(768)],lambda x,w,b:F.layer_norm(x,(768,),w,b)),
        ("softmax",[normal(64,12,17,17)],lambda x:F.softmax(x,dim=-1)),
        ("gelu",[normal(64,17,3072)],F.gelu),
    ]
    for name, values, operation in specs:
        with torch.no_grad():
            upstream = torch.randn(operation(*values).shape, generator=g).cuda()
        hashes = []
        for repeat in range(3):
            output = operation(*values)
            gradients = torch.autograd.grad(output, values, upstream)
            assert output.dtype == torch.float32 and all(t.dtype == torch.float32 for t in gradients)
            hashes.append([digest(t) for t in (output, *gradients)])
            if repeat == 0:
                tensors.update({name + "/" + str(i): t.detach().cpu() for i,t in enumerate((output, *gradients))})
        assert all(row == hashes[0] for row in hashes), name
        cases[name] = {"same_card_repeatable": True, "hashes": hashes[0]}
    profile = accelerator.numerics_state()
    torch.save(tensors, args.output / "tensors.pt")
    configure_accelerator(backend=args.backend, reset=True)
    assert configure_accelerator(backend=args.backend).numerics_state()["profile"] == "native"
    report = {"status": "passed", "backend": args.backend, "torch": torch.__version__,
              "gpu": torch.cuda.get_device_name(), "numerics": profile, "cases": cases,
              "tensor_hashes": {key: digest(value) for key, value in tensors.items()},
              "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
