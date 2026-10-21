"""Validate the GPU Philox protocol against scalar reference counters and resume."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys


def float32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def reference(seed, offset):
    # Scalar test oracle for Triton's documented Philox4x32-10 protocol.
    mask = (1 << 32) - 1
    a, b, c, d = offset & mask, offset >> 32, 0, 0
    k, l = seed & mask, seed >> 32
    for _ in range(10):
        x, y = 0xCD9E8D57 * c, 0xD2511F53 * a
        a, b, c, d = ((x >> 32) ^ b ^ k) & mask, x & mask, ((y >> 32) ^ d ^ l) & mask, y & mask
        k, l = (k + 0x9E3779B9) & mask, (l + 0xBB67AE85) & mask
    signed = a if a < (1 << 31) else a - (1 << 32)
    value = signed if signed >= 0 else -signed - 1
    return float32(float32(value) * float32(4.6566127342e-10))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--backend", choices=("nvidia", "metax"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve prior controls; use a new output file")
    sys.path.insert(0, str(args.source.resolve()))
    import torch
    from up.accelerator import configure_accelerator, CapabilityError
    provider = configure_accelerator({"runtime": {"accelerator": {
        "backend": args.backend, "numerics": "portable_v1"}}}, reset=True)
    cases = []
    for seed in (131, (1 << 63) + 17):
        for offset in (0, (1 << 32) - 8, (1 << 32) + 3):
            provider.manual_seed(seed)
            provider._numerics.offset = offset
            state = provider.get_rng_state()
            a = torch.rand(19, device="cuda")
            expected = torch.tensor([reference(seed, offset + i) for i in range(19)])
            assert torch.equal(a.cpu(), expected)
            provider.set_rng_state(state)
            b = torch.cat((torch.rand(7,device="cuda"),torch.rand(12,device="cuda")))
            assert torch.equal(a,b)
            cases.append({"seed": seed, "offset": offset, "oracle_equal": True,
                          "split_calls_equal": True, "values_sha256": hashlib.sha256(a.cpu().numpy().tobytes()).hexdigest()})
    state = provider.get_rng_state()
    try:
        torch.randn(2,device="cuda")
    except CapabilityError:
        pass
    else:
        raise AssertionError("Unsupported GPU random distribution silently used a vendor stream")
    assert torch.equal(state,provider.get_rng_state())
    x = torch.arange(12,device="cuda",dtype=torch.float32).reshape(3,4).t()
    y = torch.rand_like(x)
    assert y.stride() == x.stride() and torch.all((y >= 0) & (y < 1))
    report = {"status": "passed", "backend": args.backend, "torch": torch.__version__,
              "gpu": torch.cuda.get_device_name(), "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "cases": cases, "unsupported_distribution_raises_without_rng_change": True,
              "rand_like_preserves_strides": True,
              "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,sort_keys=True))
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
