"""Original UP deformable contracts, including nonstandard outputs/gradients."""
from abc import ABC, abstractmethod
from up_platform.contracts import operation


class DeformableOps(ABC):
    @staticmethod
    @operation(source="up/extensions/python/deformable_conv.py:18", output="Input NCHW, offsets[N,2*deformable_groups*Kh*Kw,Hout,Wout], weights[Cout,Cin/groups,Kh,Kw] -> output NCHW.",
               gradients="input, offsets, weights; save columns/ones buffers", dtype="Float16/32/64, matching inputs, original layout.",
               empty="Original output-size validation.", export="Original DeformConv symbol and attributes.",
               state="Original autograd buffers retained; no extra numerical policy.",
               device="Original device contract; no new CPU fallback.")
    @abstractmethod
    def deform_conv(input, offset, weight, out_channels, kernel_size, stride, pad, dilations, groups, deformable_groups):
        """Input NCHW, offsets[N,2*deformable_groups*Kh*Kw,Hout,Wout], weights[Cout,Cin/groups,Kh,Kw] -> output NCHW."""
        raise NotImplementedError
