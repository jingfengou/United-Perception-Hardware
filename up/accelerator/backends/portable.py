import torch


def naive_nms(dets, thresh):
    """Same-device NMS for the legacy score-sorted input contract."""
    from up.flags import ALIGNED_FLAG
    boxes = dets.float()
    order = torch.arange(len(boxes), device=boxes.device)
    return _ordered_nms(boxes, order, thresh, ALIGNED_FLAG.offset)


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


def gpu_iou_overlap(b1, b2, mode="IoU"):
    # Preserve the legacy extension's coordinate convention (inclusive corners).
    if b1.numel() == 0 or b2.numel() == 0:
        return b1.new_zeros((0,))
    lt = torch.maximum(b1[:, None, :2], b2[None, :, :2])
    rb = torch.minimum(b1[:, None, 2:4], b2[None, :, 2:4])
    intersection = (rb - lt + 1).clamp(min=0).prod(dim=-1)
    area1 = (b1[:, 2:4] - b1[:, :2] + 1).prod(dim=-1)[:, None]
    area2 = (b2[:, 2:4] - b2[:, :2] + 1).prod(dim=-1)[None, :]
    divisor = {"IoU": area1 + area2 - intersection, "IoF": area1, "IoS": area2}[mode]
    return intersection / divisor.clamp(min=1)
