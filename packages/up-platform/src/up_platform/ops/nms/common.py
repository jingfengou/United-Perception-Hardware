import torch


def naive_nms(dets, thresh, offset=0):
    """Same-device NMS for the legacy score-sorted input contract."""
    boxes = dets.float()
    order = torch.arange(len(boxes), device=boxes.device)
    return _ordered_nms(boxes, order, thresh, offset)


def torchvision_nms(boxes, scores, threshold):
    """Torchvision's score-order and exclusive-coordinate contract."""
    return _ordered_nms(boxes, scores.argsort(descending=True), threshold, 0)


def _ordered_nms(boxes, pending, threshold, offset):
    areas = (boxes[:, 2:4] - boxes[:, :2] + offset).prod(dim=1)
    kept = []
    while pending.numel():
        current = pending[0]
        kept.append(current)
        rest = pending[1:]
        lt = torch.maximum(boxes[current, :2], boxes[rest, :2])
        rb = torch.minimum(boxes[current, 2:4], boxes[rest, 2:4])
        intersection = (rb - lt + offset).clamp(min=0).prod(dim=1)
        iou = intersection / (areas[current] + areas[rest] - intersection)
        pending = rest[~(iou > threshold)]
    return torch.stack(kept).long() if kept else pending
