import torch


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
