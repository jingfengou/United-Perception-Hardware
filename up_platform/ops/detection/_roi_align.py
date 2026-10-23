"""Relocated from UP 52aaeb9ed27a; arithmetic and autograd state retained."""
import logging
import numpy as np
import torch
from torch.autograd import Function
from up_platform.ops._native.loader import load_native
logger = logging.getLogger(__name__)


class RoIAlignFunction(Function):

    @staticmethod
    def symbolic(g, features, rois, pooled_h, pooled_w, spatial_scale, sampling_ratio, pooled_mode, aligned=False, vendor=None):
        return g.op('RoiAlign', features, rois, spatial_scale_f=spatial_scale, pooled_width_i=pooled_w, pooled_height_i=pooled_h, pooled_mode_s=pooled_mode, sample_num_i=sampling_ratio)

    @staticmethod
    def forward(self, features, rois, pooled_h, pooled_w, spatial_scale, sampling_ratio, pooled_mode, aligned=False, vendor=None):
        self._roi_align = load_native(vendor).roi_align
        self.aligned = aligned
        self.save_for_backward(features, rois)
        self.pooled_h = pooled_h
        self.pooled_w = pooled_w
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.pooled_mode = pooled_mode
        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size(0)
        output = features.new(num_rois, num_channels, pooled_h, pooled_w).zero_()
        assert features.is_contiguous() and rois.is_contiguous()
        if output.numel() == 0:
            return output
        forward_fn = self._roi_align.forward_avg_cuda
        if pooled_mode == 'MAX':
            forward_fn = self._roi_align.forward_max_cuda
        if not features.is_cuda:
            logger.warning('---CPU version of RoIAlignPooling is a dummpy function, which is used to support tocaffe')
            forward_fn = self._roi_align.forward_cpu
        forward_fn(self.aligned, pooled_h, pooled_w, spatial_scale, sampling_ratio, features, rois, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_output = grad_output.data
        feature, rois = self.saved_tensors
        batch_size, num_channels, data_height, data_width = feature.shape
        grad_input = feature.new(batch_size, num_channels, data_height, data_width).zero_()
        if grad_output.data.numel() == 0:
            return (grad_input, None, None, None, None, None, None, None, None)
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        assert grad_output.is_contiguous()
        assert rois.is_contiguous()
        if self.pooled_mode == 'AVG':
            backward_fn = self._roi_align.backward_avg_cuda
            backward_fn(self.aligned, self.pooled_h, self.pooled_w, self.spatial_scale, self.sampling_ratio, grad_output, rois, grad_input)
        elif self.pooled_mode == 'MAX':
            backward_fn = self._roi_align.backward_max_cuda
            backward_fn(self.aligned, self.pooled_h, self.pooled_w, self.spatial_scale, self.sampling_ratio, grad_output, rois, grad_input, feature)
        else:
            raise KeyError
        return (grad_input, None, None, None, None, None, None, None, None)
