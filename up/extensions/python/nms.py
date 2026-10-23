from up_platform import get_platform
# Import from third library
import numpy as np
import torch
from up.utils.general.global_flag import ALIGNED_FLAG


def naive_nms(dets, thresh):
    return get_platform().category('detection').naive_nms(dets, thresh, offset=ALIGNED_FLAG.offset)


def softer_nms(dets, thresh=0.0001, sigma=0.5, iou_thresh=0.3, iou_method='linear', iou_sigma=0.02, method='var_voting'):
    return get_platform().category('detection').softer_nms(dets, thresh, sigma, iou_thresh, iou_method, iou_sigma, method)
