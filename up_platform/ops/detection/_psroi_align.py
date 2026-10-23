"""Relocated from UP 52aaeb9ed27a; arithmetic and autograd state retained."""
import logging
import numpy as np
import torch
from torch.autograd import Function
from up_platform.ops._native.loader import load_native
logger = logging.getLogger(__name__)


class PSRoIAlignFunction(Function):

    @staticmethod
    def symbolic(g, features, rois, group_size, spatial_scale, output_dim, sampling_ratio, vendor=None):
        return g.op('PSRoiAlign', features, rois, output_dim_i=output_dim, group_size_i=group_size, spatial_scale_f=spatial_scale, sample_num_i=sampling_ratio)

    @staticmethod
    def forward(self, features, rois, group_size, spatial_scale, output_dim, sampling_ratio, vendor=None):
        self._psroi_align = load_native(vendor).psroi_align
        self.save_for_backward(features, rois)
        self.group_size = group_size
        self.spatial_scale = spatial_scale
        self.output_dim = output_dim
        self.sampling_ratio = sampling_ratio
        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.shape[0]
        output = features.new(num_rois, self.output_dim, self.group_size, self.group_size).zero_()
        mapping_channel = torch.cuda.IntTensor(num_rois, self.output_dim, self.group_size, self.group_size).zero_()
        forward_fn = self._psroi_align.forward_cuda
        if not features.is_cuda:
            logger.warning('---CPU version of PSRoIAligning is a dummpy function, which is used to support tocaffe')
            forward_fn = self._psroi_align.forward_cpu
        forward_fn(self.group_size, self.group_size, self.output_dim, self.spatial_scale, self.sampling_ratio, features, rois, output, mapping_channel)
        self.mapping_channel = mapping_channel
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_output = grad_output.data
        feature, rois = self.saved_tensors
        assert grad_output.is_cuda
        batch_size, num_channels, data_height, data_width = feature.shape
        grad_input = grad_output.new(batch_size, num_channels, data_height, data_width).zero_()
        self._psroi_align.backward_cuda(self.group_size, self.group_size, self.output_dim, self.spatial_scale, self.sampling_ratio, grad_output, rois, grad_input, self.mapping_channel)
        return (grad_input, None, None, None, None, None, None)
