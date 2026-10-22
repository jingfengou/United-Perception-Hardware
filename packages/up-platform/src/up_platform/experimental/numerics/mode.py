"""Opt-in GPU numerical implementations, without changing model modules."""
import contextlib
import struct
import threading

import torch
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import TorchDispatchMode

from up_platform.errors import BackendConfigurationError, CapabilityError


_MAGIC = b"UP-PHILOX-1\n"


class _RandomDispatch(TorchDispatchMode):
    """Route below composite attention/dropout functions as well as direct calls."""

    def __init__(self, owner):
        super().__init__()
        self.owner = owner

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        try:
            return self._dispatch(function, types, args, kwargs)
        except (CapabilityError, ImportError, OSError) as exc:
            raise self.owner.capability_error(str(function), str(exc)) from exc

    def _dispatch(self, function, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = str(function)
        if (name in ("aten.sum.default", "aten.sum.dim_IntList") and args
                and args[0].device.type == "cuda" and self.owner.device_type == "cuda"
                and args[0].dtype == torch.float32
                and kwargs.get("dtype") in (None, torch.float32, torch.float64)):
            from .portable_math import reduce_sum
            dims = args[1] if len(args) > 1 else kwargs.get("dim")
            keepdim = args[2] if len(args) > 2 else kwargs.get("keepdim", False)
            self.owner.counts["gpu_fixed_fp32_sum"] = self.owner.counts.get("gpu_fixed_fp32_sum", 0) + 1
            return reduce_sum(args[0].double(), dims, keepdim).to(kwargs.get("dtype") or args[0].dtype)
        if name == "aten._foreach_norm.Scalar" and self.owner.device_type == "cuda" and args[0]:
            order = args[1] if len(args) > 1 else kwargs.get("ord", 2)
            values = args[0]
            if order == 2 and all(value.device.type == "cuda" and value.dtype in (torch.float32, torch.float64)
                                  for value in values):
                from .portable_distance import vector_norm
                self.owner.counts["gpu_fixed_grad_norm"] = self.owner.counts.get("gpu_fixed_grad_norm", 0) + 1
                return [vector_norm(value.double()).to(value.dtype) for value in values]
        if (args and isinstance(args[0], torch.Tensor)
                and args[0].device.type == self.owner.device_type
                and args[0].dtype == torch.float64
                and self.owner.device_type == "cuda"):
            if name in ("aten.mm.default", "aten.bmm.default", "aten.addmm.default", "aten.baddbmm.default"):
                from .portable_math import matrix
                self.owner.counts["gpu_fixed_matrix"] = self.owner.counts.get("gpu_fixed_matrix", 0) + 1
                if name in ("aten.mm.default", "aten.bmm.default"):
                    return matrix(*args)
                base, a, b = args[:3]
                beta, alpha = kwargs.get("beta", 1), kwargs.get("alpha", 1)
                product = matrix(a, b) * alpha
                return product if beta == 0 else product + base * beta
            if name == "aten.linalg_vector_norm.default":
                order = args[1] if len(args) > 1 else kwargs.get("ord", 2)
                if order == 2:
                    from .portable_distance import vector_norm
                    return vector_norm(*args, **kwargs)
            if name in ("aten._cdist_forward.default", "aten._cdist_backward.default"):
                order_index = 2 if name == "aten._cdist_forward.default" else 3
                order = args[order_index] if len(args) > order_index else kwargs.get("p", 2)
                if order == 1:
                    from . import portable_distance
                    self.owner.counts["gpu_fixed_l1_distance"] = self.owner.counts.get("gpu_fixed_l1_distance", 0) + 1
                    implementation = (portable_distance.l1_distance if name == "aten._cdist_forward.default"
                                      else portable_distance.l1_backward)
                    return implementation(*args, **kwargs)
            if name in ("aten.sum.default", "aten.sum.dim_IntList") and kwargs.get("dtype") in (None, torch.float64):
                from .portable_math import reduce_sum
                dims = args[1] if len(args) > 1 else kwargs.get("dim")
                keepdim = args[2] if len(args) > 2 else kwargs.get("keepdim", False)
                return reduce_sum(args[0], dims, keepdim)
            normalizations = {
                "aten.native_batch_norm.default": "batch_norm",
                "aten.native_batch_norm_backward.default": "batch_norm_backward",
                "aten.native_layer_norm.default": "layer_norm",
                "aten.native_layer_norm_backward.default": "layer_norm_backward",
                "aten._softmax.default": "softmax",
                "aten._softmax_backward_data.default": "softmax_backward",
            }
            if name in normalizations:
                from . import portable_normalization
                implementation = getattr(portable_normalization, normalizations[name])
                self.owner.counts["gpu_fixed_normalization"] = self.owner.counts.get("gpu_fixed_normalization", 0) + 1
                if name == "aten._softmax.default":
                    return implementation(args[0], args[1])
                return implementation(*args, **kwargs)
            if ((name == "aten.convolution.default" and args[0].ndim == 4 and not args[6])
                    or (name == "aten.convolution_backward.default" and args[1].ndim == 4 and not args[7])):
                from . import portable_convolution
                self.owner.counts["gpu_fixed_convolution"] = self.owner.counts.get("gpu_fixed_convolution", 0) + 1
                implementation = (portable_convolution.convolution if name == "aten.convolution.default"
                                  else portable_convolution.convolution_backward)
                return implementation(*args, **kwargs)
        if name in ("aten.rand.default", "aten.rand.generator"):
            device = kwargs.get("device") or "cpu"
            if torch.device(device).type == self.owner.device_type:
                if kwargs.get("generator") is not None:
                    raise CapabilityError("portable_v1 rand does not support explicit generators")
                return self.owner.uniform(args[0], device, kwargs.get("dtype") or torch.get_default_dtype())
        if name == "aten.rand_like.default" and args[0].device.type == self.owner.device_type:
            device = kwargs.get("device", args[0].device)
            if torch.device(device).type == self.owner.device_type:
                value = self.owner.uniform(args[0].shape, device, kwargs.get("dtype") or args[0].dtype)
                return torch.empty_like(args[0], dtype=value.dtype, device=device,
                                        memory_format=kwargs.get("memory_format", torch.preserve_format)).copy_(value)
        if name == "aten.native_dropout.default" and args[0].device.type == self.owner.device_type:
            x, p, training = args
            if x.dtype not in (torch.float32, torch.float64):
                raise CapabilityError("portable_v1 dropout requires FP32 boundaries")
            if training is False or p == 0:
                return x.clone(), torch.ones_like(x, dtype=torch.bool)
            mask = (torch.zeros_like(x, dtype=torch.bool) if p == 1 else
                    self.owner.uniform(x.shape, x.device) >= p)
            output = ((x.double() * mask * (1. / (1. - p))).to(x.dtype)
                      if p < 1 else x * 0)
            return output, mask
        if name == "aten.native_dropout_backward.default" and args[0].device.type == self.owner.device_type:
            gradient, mask, scale = args
            if gradient.dtype not in (torch.float32, torch.float64):
                raise CapabilityError("portable_v1 dropout backward requires FP32 boundaries")
            # Autograd dispatches the native backward independently of our forward.
            return (gradient.double() * mask * scale).to(gradient.dtype)
        if name in ("aten.bernoulli_.float", "aten.bernoulli.p") and args[0].device.type == self.owner.device_type:
            x = args[0]
            if kwargs.get("generator") is not None:
                raise CapabilityError("portable_v1 Bernoulli does not support explicit generators")
            p = args[1] if len(args) > 1 else .5
            if not 0 <= p <= 1:
                raise ValueError("Bernoulli probability must be between 0 and 1")
            value = (self.owner.uniform(x.shape, x.device) < p).to(x.dtype)
            return x.copy_(value) if name == "aten.bernoulli_.float" else value
        if name == "aten.bernoulli.default" and args[0].device.type == self.owner.device_type:
            x = args[0]
            if kwargs.get("generator") is not None:
                raise CapabilityError("portable_v1 Bernoulli does not support explicit generators")
            if not bool(((x >= 0) & (x <= 1)).all()):
                raise ValueError("Bernoulli probabilities must be between 0 and 1")
            return (self.owner.uniform(x.shape, x.device) < x).to(x.dtype)
        if name == "aten.uniform_.default" and args[0].device.type == self.owner.device_type:
            x = args[0]
            if kwargs.get("generator") is not None:
                raise CapabilityError("portable_v1 uniform does not support explicit generators")
            low = args[1] if len(args) > 1 else kwargs.get("from", 0.)
            high = args[2] if len(args) > 2 else kwargs.get("to", 1.)
            if low > high:
                raise ValueError("Uniform bounds must be ordered")
            return x.copy_(self.owner.uniform(x.shape, x.device, x.dtype) * (high - low) + low)
        if name.startswith(("aten.randn", "aten.randint", "aten.normal", "aten.poisson", "aten.multinomial",
                            "aten.exponential", "aten.cauchy", "aten.log_normal", "aten.geometric", "aten.rand.",
                            "aten.rand_like.", "aten.bernoulli.", "aten.bernoulli_.")):
            tensor = next((t for t in args if isinstance(t, torch.Tensor)), kwargs.get("out"))
            device = tensor.device if tensor is not None else torch.device(kwargs.get("device") or "cpu")
            if device.type == self.owner.device_type:
                raise CapabilityError("portable_v1 has no common GPU RNG implementation for " + name)
        return function(*args, **kwargs)


def _map(value, function):
    if isinstance(value, torch.Tensor):
        return function(value)
    if isinstance(value, tuple):
        return tuple(_map(item, function) for item in value)
    if isinstance(value, list):
        return [_map(item, function) for item in value]
    if isinstance(value, dict):
        return {key: _map(item, function) for key, item in value.items()}
    return value


class PortableNumerics(TorchFunctionMode):
    """FP32 boundaries, widened GPU arithmetic and a versioned GPU RNG stream.

    This is an explicit numerical policy, not a bitwise-equivalence guarantee.
    It is process/thread scoped, eager-only, and currently single-device only.
    """

    def __init__(self, device_type="cuda", widened=True, backend_name=None):
        super().__init__()
        self.device_type = device_type
        self.backend_name = backend_name or "unbound"
        self.widened = widened
        self.seed = 0
        self.offset = 0
        self.owner_thread = threading.get_ident()
        self.device_index = None
        self.counts = {}
        self.random_mode = _RandomDispatch(self)
        self.functions = {
            F.conv1d, F.conv2d, F.conv3d, F.linear, F.batch_norm, F.layer_norm,
            F.softmax, F.log_softmax, F.gelu, F.silu, F.binary_cross_entropy_with_logits,
            F.cross_entropy, F.mse_loss, F.l1_loss, F.smooth_l1_loss,
            F.multi_head_attention_forward, F.scaled_dot_product_attention,
            torch.matmul, torch.mm, torch.bmm, torch.addmm, torch.baddbmm,
            torch.softmax, torch.log_softmax, torch.sum, torch.mean, torch.var, torch.std,
            torch.erf, torch.tanh, torch.sigmoid, torch.exp, torch.log, torch.sqrt, torch.rsqrt,
            torch.pow, torch.atan, torch.atan2,
            torch.linalg.vector_norm, torch.cdist,
        }
        for name in ("matmul", "__matmul__", "mm", "bmm", "addmm", "baddbmm", "softmax",
                     "log_softmax", "sum", "mean", "var", "std", "erf", "tanh", "sigmoid",
                     "exp", "log", "sqrt", "rsqrt", "pow", "atan", "atan2"):
            self.functions.add(getattr(torch.Tensor, name))

    def capability_error(self, operation, reason):
        return CapabilityError("backend={!r}, numerics='portable_v1', operation={!r}: {}".format(
            self.backend_name, operation, reason))

    def _check_thread(self):
        if threading.get_ident() != self.owner_thread:
            raise BackendConfigurationError("Portable numerical state belongs to another thread")

    def reseed(self, seed):
        self._check_thread()
        self.seed, self.offset = int(seed) % (1 << 64), 0

    def state(self, native):
        self._check_thread()
        payload = _MAGIC + struct.pack("<QQ", self.seed, self.offset) + bytes(native.cpu().tolist())
        return torch.tensor(list(payload), dtype=torch.uint8, device="cpu")

    def restore(self, value):
        self._check_thread()
        if value.device.type != "cpu" or value.dtype != torch.uint8 or value.ndim != 1:
            raise BackendConfigurationError("Portable RNG state must be a CPU uint8 vector")
        payload = bytes(value.tolist())
        if not payload.startswith(_MAGIC) or len(payload) <= len(_MAGIC) + 16:
            raise BackendConfigurationError("Not a portable_v1 RNG state; do not mix numerical profiles")
        seed, offset = struct.unpack("<QQ", payload[len(_MAGIC):len(_MAGIC) + 16])
        if offset >= 1 << 63:
            raise BackendConfigurationError("Portable RNG state contains an invalid counter")
        native = value[len(_MAGIC) + 16:].clone()
        return seed, offset, native

    def uniform(self, shape, device, dtype=torch.float32):
        self._check_thread()
        if dtype != torch.float32:
            raise CapabilityError("portable_v1 random sampling currently supports float32 only")
        device = torch.device(device)
        index = device.index if device.index is not None else torch.cuda.current_device()
        if self.device_index is not None and index != self.device_index:
            raise CapabilityError("portable_v1 requires one accelerator device per process")
        self.device_index = index
        count = 1
        for size in shape:
            count *= size
        if self.offset + count >= 1 << 63:
            raise CapabilityError("portable_v1 random counter exhausted")
        try:
            from .portable_rng import uniform
            output = uniform(shape, device, self.seed, self.offset)
        except (ImportError, OSError) as exc:
            raise CapabilityError("portable_v1 requires a working vendor Triton implementation: " + str(exc)) from exc
        self.offset += count
        self.counts["gpu_uniform_values"] = self.counts.get("gpu_uniform_values", 0) + count
        return output

    def _wide(self, function, args, kwargs, tensor):
        if kwargs.get("out") is not None:
            raise CapabilityError("portable_v1 does not implement out= for " + function.__name__)
        if tensor.dtype != torch.float32:
            if tensor.dtype in (torch.float16, torch.bfloat16):
                raise CapabilityError("portable_v1 is an FP32-boundary profile, not an AMP implementation")
            return function(*args, **kwargs)
        if torch.is_autocast_enabled():
            raise CapabilityError("portable_v1 cannot be combined with autocast")
        if kwargs.get("dtype") not in (None, torch.float32):
            return function(*args, **kwargs)
        promote = lambda t: t.double() if t.is_floating_point() and t.dtype == torch.float32 else t
        wide_args, wide_kwargs = _map(args, promote), _map(kwargs, promote)
        if "dtype" in wide_kwargs:
            wide_kwargs["dtype"] = torch.float64
        running = []
        if function is F.batch_norm:
            names = ("input", "running_mean", "running_var", "weight", "bias", "training", "momentum", "eps")
            original = dict(zip(names, args))
            original.update(kwargs)
            promoted = dict(zip(names, wide_args))
            promoted.update(wide_kwargs)
            running = [(original.get(k), promoted.get(k)) for k in ("running_mean", "running_var")]
        inplace = bool(kwargs.get("inplace", args[1] if function is F.silu and len(args) > 1 else False))
        if function is F.silu:
            wide_args = wide_args[:1]
            wide_kwargs = dict(wide_kwargs, inplace=False)
        context = torch.backends.cudnn.flags(enabled=False) if function in (F.conv1d, F.conv2d, F.conv3d, F.batch_norm) else contextlib.nullcontext()
        if function in (F.multi_head_attention_forward, F.scaled_dot_product_attention):
            from torch.nn.attention import SDPBackend, sdpa_kernel
            context = sdpa_kernel(SDPBackend.MATH)
        with context:
            output = function(*wide_args, **wide_kwargs)
        for original, promoted in running:
            if original is not None:
                with torch.no_grad():
                    original.copy_(promoted)
        result = _map(output, lambda t: t.float() if t.dtype == torch.float64 else t)
        return tensor.copy_(result) if function is F.silu and inplace else result

    def __torch_function__(self, function, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if function not in self.functions:
            return function(*args, **kwargs)
        tensors = []
        _map((args, kwargs), lambda t: tensors.append(t) or t)
        tensor = next((t for t in tensors if t.device.type == self.device_type), None)
        if tensor is None:
            return function(*args, **kwargs)
        name = getattr(function, "__name__", str(function))
        self.counts[name] = self.counts.get(name, 0) + 1
        try:
            return self._wide(function, args, kwargs, tensor) if self.widened else function(*args, **kwargs)
        except (CapabilityError, ImportError, OSError) as exc:
            raise self.capability_error(name, str(exc)) from exc
