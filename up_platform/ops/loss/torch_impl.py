"""Explicit bindings to the installed vendor PyTorch; no device transfer."""
import torch
from torch.nn import functional as F

binary_cross_entropy = F.binary_cross_entropy
binary_cross_entropy_with_logits = F.binary_cross_entropy_with_logits
cross_entropy = F.cross_entropy
kl_div = F.kl_div
l1_loss = F.l1_loss
CrossEntropyLoss = torch.nn.CrossEntropyLoss
BCEWithLogitsLoss = torch.nn.BCEWithLogitsLoss
KLDivLoss = torch.nn.KLDivLoss
L1Loss = torch.nn.L1Loss
CosineSimilarity = torch.nn.CosineSimilarity
