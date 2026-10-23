from up_platform import get_platform
# Import from local


def gpu_iou_overlap(b1, b2, mode='IoU'):
    return get_platform().category('detection').iou_overlap(b1, b2, mode)
