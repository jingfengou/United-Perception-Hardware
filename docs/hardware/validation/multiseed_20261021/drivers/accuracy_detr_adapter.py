"""Single-device validation adapter for pinned official DETR, not a UP baseline."""
import hashlib
import importlib
import json
from pathlib import Path
import sys

import torch
from torch import nn

DETR_REVISION = "29901c51d7fe8712168b8d0d64351170bc0f83e0"


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hash(source):
    files = {str(path.relative_to(source)): file_hash(path) for path in sorted(source.rglob("*.py"))}
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def scale_pair(info):
    value = info[2]
    if isinstance(value, (list, tuple)):
        return float(value[0]), float(value[1])
    return float(value), float(value)


def target_boxes(boxes, info, offset):
    """Undo UP's scaled inclusive endpoint before normalizing DETR half-open boxes."""
    height, width = float(info[0]), float(info[1])
    scale_h, scale_w = scale_pair(info)
    result = boxes.clone()
    result[:, 2] += offset * scale_w
    result[:, 3] += offset * scale_h
    result[:, [0, 2]] = result[:, [0, 2]].clamp(0, width)
    result[:, [1, 3]] = result[:, [1, 3]].clamp(0, height)
    x0, y0, x1, y1 = result.unbind(-1)
    return torch.stack(((x0 + x1) / (2 * width), (y0 + y1) / (2 * height),
                        (x1 - x0) / width, (y1 - y0) / height), dim=-1)


def up_prediction_boxes(boxes, info, offset):
    scale_h, scale_w = scale_pair(info)
    result = boxes.clone()
    result[:, 2] -= offset * scale_w
    result[:, 3] -= offset * scale_h
    return result


class DetrForUP(nn.Module):
    def __init__(self, source, checkpoint, source_sha256, checkpoint_sha256):
        super().__init__()
        source = Path(source).resolve()
        if source_hash(source) != source_sha256 or file_hash(checkpoint) != checkpoint_sha256:
            raise ValueError("Pinned DETR source or checkpoint changed")
        sys.path.insert(0, str(source))
        detr = importlib.import_module("models.detr")
        backbone = importlib.import_module("models.backbone")
        transformer = importlib.import_module("models.transformer")
        position = importlib.import_module("models.position_encoding")
        matcher = importlib.import_module("models.matcher")
        misc = importlib.import_module("util.misc")
        if Path(detr.__file__).resolve().parent != source / "models":
            raise RuntimeError("Imported a different DETR package")
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            raise RuntimeError("This external validation adapter has not been adapted for distributed backends")
        from torchvision.models import resnet50

        # Avoid the official Backbone constructor's implicit network download.
        body = resnet50(weights=None, replace_stride_with_dilation=[False, False, False],
                        norm_layer=backbone.FrozenBatchNorm2d)
        features = backbone.BackboneBase(body, train_backbone=True, num_channels=2048,
                                         return_interm_layers=False)
        features = backbone.Joiner(features, position.PositionEmbeddingSine(128, normalize=True))
        features.num_channels = 2048
        self.detector = detr.DETR(features, transformer.Transformer(
            d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
            dim_feedforward=2048, dropout=.1, return_intermediate_dec=True),
            num_classes=91, num_queries=100, aux_loss=True)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.detector.load_state_dict(state, strict=True)
        head = nn.Linear(256, 2)
        with torch.no_grad():
            head.weight.copy_(self.detector.class_embed.weight[[1, -1]])
            head.bias.copy_(self.detector.class_embed.bias[[1, -1]])
        self.detector.class_embed = head
        self.weights = {"loss_ce": 1., "loss_bbox": 5., "loss_giou": 2.}
        self.weights.update({name + "_" + str(index): value
                             for index in range(5) for name, value in list(self.weights.items())})
        self.criterion = detr.SetCriterion(1, matcher.HungarianMatcher(1, 5, 2),
                                           self.weights, .1, ["labels", "boxes", "cardinality"])
        self.postprocess = detr.PostProcess()
        self.nested_tensor = misc.NestedTensor

    def forward(self, batch):
        from up.utils.general.global_flag import ALIGNED_FLAG

        images, information = batch["image"], batch["image_info"]
        mask = images.new_ones((images.shape[0], images.shape[2], images.shape[3]), dtype=torch.bool)
        for index, info in enumerate(information):
            if float(info[6]) or float(info[7]):
                raise ValueError("This pinned validation recipe expects no left/top padding")
            mask[index, :int(info[0]), :int(info[1])] = False
        predictions = self.detector(self.nested_tensor(images, mask))
        if self.training:
            targets = []
            for boxes, info in zip(batch["gt_bboxes"], information):
                selected = boxes[boxes[:, 4] > 0]
                targets.append({"boxes": target_boxes(selected[:, :4], info, ALIGNED_FLAG.offset),
                                "labels": selected.new_zeros(selected.shape[0], dtype=torch.long)})
            losses = self.criterion(predictions, targets)
            return {"DETR." + name: value * self.weights[name]
                    for name, value in losses.items() if name in self.weights}
        sizes = images.new_tensor([[float(info[0]), float(info[1])] for info in information])
        decoded = self.postprocess(predictions, sizes)
        rows = []
        for index, (result, info) in enumerate(zip(decoded, information)):
            boxes = up_prediction_boxes(result["boxes"], info, ALIGNED_FLAG.offset)
            rows.append(torch.cat((boxes.new_full((boxes.shape[0], 1), index), boxes,
                                   result["scores"][:, None], (result["labels"] + 1).to(boxes.dtype)[:, None]), dim=1))
        return {"dt_bboxes": torch.cat(rows, dim=0)}
