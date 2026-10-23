"""Relocated from UP 52aaeb9ed27a; arithmetic and autograd state retained."""
import logging
import numpy as np
import torch
from torch.autograd import Function
from up_platform.ops._native.loader import load_native
logger = logging.getLogger(__name__)


def naive_nms(dets, thresh, *, offset, vendor):
    nms = load_native(vendor).naive_nms
    softer_nms_c = load_native(vendor).softer_nms
    assert dets.shape[1] == 5
    num_dets = dets.shape[0]
    if torch.is_tensor(num_dets):
        num_dets = num_dets.item()
    keep = torch.LongTensor(num_dets)
    num_out = torch.LongTensor(1)
    if dets.device.type == 'cuda':
        nms.gpu_nms(keep, num_out, dets.float(), thresh, offset)
    else:
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        areas = (x2 - x1 + offset) * (y2 - y1 + offset)
        order = torch.from_numpy(np.arange(dets.shape[0])).long()
        nms.cpu_nms(keep, num_out, dets.float(), order, areas, thresh, offset)
    return keep[:num_out[0]].contiguous().to(device=dets.device)


def softer_nms(dets, thresh=0.0001, sigma=0.5, iou_thresh=0.3, iou_method='linear', iou_sigma=0.02, method='var_voting', *, vendor):
    nms = load_native(vendor).naive_nms
    softer_nms_c = load_native(vendor).softer_nms
    if dets.shape[1] != 9:
        raise ValueError('Expected input tensor `dets` of 9-channels, got {}-channels'.format(dets.shape[1]))
    device = dets.device
    dets = dets.detach()
    if not hasattr(softer_nms_c.IOUMethod, iou_method.upper()):
        raise TypeError('IoU_method {} not supported'.format(iou_method.upper()))
    iou_method = getattr(softer_nms_c.IOUMethod, iou_method.upper())
    if not hasattr(softer_nms_c.NMSMethod, method.upper()):
        raise TypeError('Softer NMS method {} not supported'.format(method.upper()))
    if method.upper() == 'SOFTER':
        raise Warning('Using deprecated softer_nms (method == SOFTER)')
    method = getattr(softer_nms_c.NMSMethod, method.upper())
    dets = dets.cpu()
    N = dets.shape[0]
    inds = torch.arange(0, N, dtype=torch.long)
    softer_nms_c.cpu_softer_nms(dets, inds, sigma, iou_thresh, iou_method, iou_sigma, method)
    keep = (dets[:, 8] > thresh).nonzero().reshape(-1)
    dets = dets[keep]
    keep = inds[keep]
    return (dets.to(device=device), keep.to(device=device))
