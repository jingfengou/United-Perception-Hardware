"""Device selection and lazy built-in registration; importing does not bind hardware."""
from up_platform.errors import BackendConfigurationError
from up_platform.registry_factory import (BACKEND_PROBES, BackendSpec, register_backend,
                                          TRAINING_FACTORY, INFERENCE_FACTORY)


def resolve_backend(name, device):
    if name != "auto":
        return name
    if device is not None and str(device).split(":")[0] == "cpu":
        return "cpu"
    matches = [name for name, probe in BACKEND_PROBES.items() if probe()]
    if len(matches) > 1:
        raise BackendConfigurationError("Ambiguous accelerator probes: " + ", ".join(matches))
    if matches:
        return matches[0]
    import torch
    if hasattr(torch, "npu") or hasattr(torch, "mlu"):
        raise BackendConfigurationError("Select ascend/cambricon explicitly")
    if getattr(torch.version, "maca", None) or getattr(torch.version, "metax", None):
        raise BackendConfigurationError("Select metax explicitly; cuda is not a vendor identity")
    return "nvidia" if torch.version.cuda else "cpu"


for _name, _device, _class in (("cpu", "cpu", "CpuDevice"), ("nvidia", "cuda", "NvidiaDevice"),
                                ("ascend", "npu", "AscendDevice"), ("cambricon", "mlu", "CambriconDevice"),
                                ("metax", "cuda", "MetaXDevice")):
    register_backend(BackendSpec(
        _name, _device,
        "up_platform.base.base:DeviceBackend" if _name == "cpu" else
        "up_platform.base.{}:{}".format(_name, _class),
        TRAINING_FACTORY, INFERENCE_FACTORY))
