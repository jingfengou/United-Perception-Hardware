from up_platform.dependencies import dependency
from functools import lru_cache

_module_types = {}


def voxelize_cpu(generator, points):
    tv = dependency('cumm.tensorview')
    output = generator.point_to_voxel(tv.from_numpy(points))
    # numpy_view() would alias generator scratch storage; upstream uses numpy().
    return tuple(value.numpy() for value in output)


voxelize_cpu.dependencies = ('cumm.tensorview',)

@lru_cache()
def spconv():
    package = dependency('spconv')
    major = int(package.__version__.split('.')[0])
    return dependency('spconv.pytorch') if major >= 2 else package


def SubMConv3d(in_channels, out_channels, kernel_size, **kwargs):
    return spconv().SubMConv3d(in_channels, out_channels, kernel_size, **kwargs)


def SparseConv3d(in_channels, out_channels, kernel_size, **kwargs):
    return spconv().SparseConv3d(in_channels, out_channels, kernel_size, **kwargs)


def SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key, **kwargs):
    return spconv().SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, **kwargs)


def SparseSequential(*modules):
    return spconv().SparseSequential(*modules)


def SparseConvTensor(features, indices, spatial_shape, batch_size, **kwargs):
    return spconv().SparseConvTensor(features, indices, spatial_shape, batch_size, **kwargs)


def replace_feature(tensor, features):
    if hasattr(tensor, 'replace_feature'):
        return tensor.replace_feature(features)
    tensor.features = features
    return tensor


def features(tensor):
    return tensor.features


def dense(tensor, channels_first=True):
    return tensor.dense(channels_first=channels_first)


def is_sparse_convolution(module):
    return isinstance(module, spconv().conv.SparseConvolution)


def module_instance(module_type):
    base = spconv().SparseModule
    bound = _module_types.get((module_type, base))
    if bound is None:
        bound = type(module_type.__name__, (module_type, base), {'__module__': module_type.__module__})
        _module_types[module_type, base] = bound
    return object.__new__(bound)

