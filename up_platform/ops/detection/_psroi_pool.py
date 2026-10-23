"""Relocated from UP 52aaeb9ed27a; arithmetic and autograd state retained."""
import logging
import numpy as np
import torch
from torch.autograd import Function
from up_platform.ops._native.loader import load_native
logger = logging.getLogger(__name__)


class PSRoIPoolFunction(Function):

    @staticmethod
    def symbolic(g, features, rois, group_size, spatial_scale, output_dim, vendor=None):
        return g.op('PSRoiPool', features, rois, output_dim_i=output_dim, group_size_i=group_size, spatial_scale_f=spatial_scale)

    @staticmethod
    def forward(self, features, rois, group_size, spatial_scale, output_dim, vendor=None):
        self._psroi_pooling = load_native(vendor).psroi_pooling
        self.save_for_backward(features, rois)
        self.group_size = group_size
        self.spatial_scale = spatial_scale
        self.output_dim = output_dim
        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.shape[0]
        output = features.new(num_rois, self.output_dim, self.group_size, self.group_size).zero_()
        mapping_channel = torch.IntTensor(num_rois, self.output_dim, self.group_size, self.group_size).zero_()
        forward_fn = self._psroi_pooling.forward_cuda
        if not features.is_cuda:
            logger.warning('---CPU version of PSRoIPooling is a dummpy function, which is used to support tocaffe')
            forward_fn = self._psroi_pooling.forward_cpu
        else:
            mapping_channel = mapping_channel.cuda()
            if vendor == 'metax':
                # MetaX .cuda() converts 4-D CPU buffers to channels-last.
                # The original kernel indexes this integer workspace as NCHW.
                mapping_channel = mapping_channel.contiguous()
        forward_fn(self.group_size, self.group_size, self.output_dim, self.spatial_scale, features, rois, output, mapping_channel)
        self.mapping_channel = mapping_channel
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_output = grad_output.data
        feature, rois = self.saved_tensors
        assert grad_output.is_cuda
        batch_size, num_channels, data_height, data_width = feature.shape
        grad_input = grad_output.new(batch_size, num_channels, data_height, data_width).zero_()
        self._psroi_pooling.backward_cuda(self.group_size, self.group_size, self.output_dim, self.spatial_scale, grad_output, rois, grad_input, self.mapping_channel)
        return (grad_input, None, None, None, None, None)
