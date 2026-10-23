from abc import ABC, abstractmethod
from up_platform.contracts import operation


class SparseOps(ABC):
    @staticmethod
    @operation(source="cumm.tensorview.from_numpy", output="Tuple of owned NumPy voxel, coordinate and count arrays.",
               gradients="None: original explicitly CPU data preprocessing.",
               device="CPU NumPy inputs/outputs only; this is not a GPU fallback.",
               state="The supplied Point2VoxelCPU3d generator owns scratch storage; returned arrays must own copies.",
               export="Not part of the model export graph.")
    @abstractmethod
    def voxelize_cpu(generator, points):
        """Call the original Point2VoxelCPU3d/cumm bridge and return independent arrays."""
        raise NotImplementedError


    @staticmethod
    @operation(source="sparse.SubMConv3d", output="Original sparse module and parameter layout.", gradients="Features and weights",
               state="Preserve dependency state updates, parameter identity and call order.")
    @abstractmethod
    def SubMConv3d(in_channels, out_channels, kernel_size, **kwargs):
        "Original sparse module and parameter layout."
        raise NotImplementedError

    @staticmethod
    @operation(source="sparse.SparseConv3d", output="Original sparse module; original indice mapping and weight layout.", gradients="Features and weights",
               state="Preserve dependency state updates, parameter identity and call order.")
    @abstractmethod
    def SparseConv3d(in_channels, out_channels, kernel_size, **kwargs):
        "Original sparse module; original indice mapping and weight layout."
        raise NotImplementedError

    @staticmethod
    @operation(source="sparse.SparseInverseConv3d", output="Original inverse sparse convolution; requires corresponding indice state.", gradients="Features and weights",
               state="Preserve dependency state updates, parameter identity and call order.")
    @abstractmethod
    def SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key, **kwargs):
        "Original inverse sparse convolution; requires corresponding indice state."
        raise NotImplementedError

    @staticmethod
    @operation(source="sparse.SparseSequential", output="Original sparse sequential module; no extra checkpoint prefix.", gradients="According to contained modules",
               state="Preserve dependency state updates, parameter identity and call order.")
    @abstractmethod
    def SparseSequential(*modules):
        "Original sparse sequential module; no extra checkpoint prefix."
        raise NotImplementedError

    @staticmethod
    @operation(source="sparse.SparseConvTensor", output="Vendor sparse tensor with int32 [N,4] indices and original cache semantics.", gradients="Features only",
               state="Preserve dependency state updates, parameter identity and call order.")
    @abstractmethod
    def SparseConvTensor(features, indices, spatial_shape, batch_size, **kwargs):
        "Vendor sparse tensor with int32 [N,4] indices and original cache semantics."
        raise NotImplementedError

    @staticmethod
    @operation(source="sparse.replace_feature", output="Sparse tensor preserving indices and indice dictionaries.", gradients="Features",
               state="Preserve dependency state updates, parameter identity and call order.")
    @abstractmethod
    def replace_feature(tensor, features):
        "Sparse tensor preserving indices and indice dictionaries."
        raise NotImplementedError

    @staticmethod
    @operation(source="sparse.features", output="Original feature tensor without clone/detach.", gradients="Preserve graph and aliasing",
               state="Preserve dependency state updates, parameter identity and call order.")
    @abstractmethod
    def features(tensor):
        "Original feature tensor without clone/detach."
        raise NotImplementedError

    @staticmethod
    @operation(source="sparse.dense", output="Dense tensor using original spconv layout.", gradients="Features",
               state="Preserve dependency state updates, parameter identity and call order.")
    @abstractmethod
    def dense(tensor, channels_first=True):
        "Dense tensor using original spconv layout."
        raise NotImplementedError

    @staticmethod
    @operation(source="sparse.is_sparse_convolution", output="Whether module is an actual spconv SparseConvolution.", gradients="None",
               state="Preserve dependency state updates, parameter identity and call order.")
    @abstractmethod
    def is_sparse_convolution(module):
        "Whether module is an actual spconv SparseConvolution."
        raise NotImplementedError

    @staticmethod
    @operation(source="sparse.module_instance", output="Uninitialized instance retaining UP class methods and vendor SparseModule recognition.", gradients="Not applicable",
               state="Preserve dependency state updates, parameter identity and call order.")
    @abstractmethod
    def module_instance(module_type):
        "Uninitialized instance retaining UP class methods and vendor SparseModule recognition."
        raise NotImplementedError

