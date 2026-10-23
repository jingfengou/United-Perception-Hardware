"""metax: explicitly reuse the installed vendor PyTorch implementation."""
from .base import LossOps
from . import torch_impl


class MetaxLossOps(LossOps):
    dependencies = ("torch",)
    binary_cross_entropy = staticmethod(torch_impl.binary_cross_entropy)
    binary_cross_entropy_with_logits = staticmethod(torch_impl.binary_cross_entropy_with_logits)
    cross_entropy = staticmethod(torch_impl.cross_entropy)
    kl_div = staticmethod(torch_impl.kl_div)
    l1_loss = staticmethod(torch_impl.l1_loss)
    CrossEntropyLoss = staticmethod(torch_impl.CrossEntropyLoss)
    BCEWithLogitsLoss = staticmethod(torch_impl.BCEWithLogitsLoss)
    KLDivLoss = staticmethod(torch_impl.KLDivLoss)
    L1Loss = staticmethod(torch_impl.L1Loss)
    CosineSimilarity = staticmethod(torch_impl.CosineSimilarity)
