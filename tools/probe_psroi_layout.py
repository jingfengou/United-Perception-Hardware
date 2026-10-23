"""Inspect the original PSRoIPool integer-buffer transfer without UP imports."""
import json
import torch


def describe(x):
    return {"shape": list(x.shape), "stride": list(x.stride()),
            "contiguous": x.is_contiguous(), "device": str(x.device),
            "channels_last": x.is_contiguous(memory_format=torch.channels_last)}


if __name__ == "__main__":
    x = torch.IntTensor(2, 2, 2, 2).zero_()
    transferred = x.cuda()
    explicit = torch.zeros(x.shape, dtype=torch.int32, device="cuda")
    fixed = transferred.contiguous(memory_format=torch.contiguous_format)
    print(json.dumps({"torch": torch.__version__, "cpu": describe(x),
                      "cuda_copy": describe(transferred), "device_zeros": describe(explicit),
                      "contiguous_copy": describe(fixed),
                      "values_unchanged": torch.equal(transferred.cpu(), fixed.cpu())}, indent=2))
