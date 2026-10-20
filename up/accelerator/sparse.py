"""Small sparse-layer interface. Providers own native tensors and parameter layout."""
import torch


def _call(name, *args, **kwargs):
    from . import ops
    return ops.call("sparse." + name, *args, **kwargs)


class SparseModule(torch.nn.Module):
    pass


class SparseSequential(torch.nn.Sequential, SparseModule):
    def forward(self, value):
        return _call("sequential_forward", self, value)


def _sequential_forward(sequence, value):
    from . import get_accelerator
    native = get_accelerator().dependency("sparse")
    for module in sequence:
        if isinstance(module, (SparseModule, native.SparseModule)):
            value = module(value)
        elif hasattr(value, "features"):
            value = replace_feature(value, module(features(value)))
        else:
            value = module(value)
    return value


def _factory(name, *args, **kwargs):
    from . import get_accelerator
    return getattr(get_accelerator().dependency("sparse"), name)(*args, **kwargs)


def SubMConv3d(*args, **kwargs):
    return _call("SubMConv3d", *args, **kwargs)


def SparseConv3d(*args, **kwargs):
    return _call("SparseConv3d", *args, **kwargs)


def SparseInverseConv3d(*args, **kwargs):
    return _call("SparseInverseConv3d", *args, **kwargs)


def SparseConvTensor(*args, **kwargs):
    return _call("SparseConvTensor", *args, **kwargs)


def features(value):
    return _call("features", value)


def replace_feature(value, new_features):
    return _call("replace_feature", value, new_features)


def to_dense(value):
    return _call("to_dense", value)


def is_sparse_convolution(module):
    return _call("is_sparse_convolution", module)


def _features(value):
    return value.features


def _replace_feature(value, new_features):
    if hasattr(value, "replace_feature"):
        return value.replace_feature(new_features)
    value.features = new_features
    return value


def _to_dense(value):
    return value.dense()


def _is_sparse_convolution(module):
    from . import get_accelerator
    return isinstance(module, get_accelerator().dependency("sparse").conv.SparseConvolution)


class _Conv:
    @property
    def SparseConvolution(self):
        from . import get_accelerator
        return get_accelerator().dependency("sparse").conv.SparseConvolution


conv = _Conv()
