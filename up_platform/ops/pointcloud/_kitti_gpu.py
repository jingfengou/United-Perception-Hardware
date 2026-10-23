import numpy as np
import torch

from up_platform.ops._native.loader import load_native


def rotate_iou_eval(boxes, query_boxes, criterion=-1, device_id=0, *, vendor):
    boxes = np.ascontiguousarray(boxes, dtype=np.float32)
    queries = np.ascontiguousarray(query_boxes, dtype=np.float32)
    if not len(boxes) or not len(queries):
        return np.zeros((len(boxes), len(queries)), dtype=np.float32)
    torch.cuda.set_device(device_id)
    device = torch.device('cuda', device_id)
    result = load_native(vendor).kitti_rotate_iou(torch.as_tensor(boxes, device=device),
                                                torch.as_tensor(queries, device=device), criterion)
    # The original evaluator explicitly returns host NumPy metrics.
    return result.cpu().numpy()
