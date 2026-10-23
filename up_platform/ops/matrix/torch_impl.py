"""Explicit bindings to the installed vendor PyTorch; no device transfer."""
import torch
from torch.nn import functional as F

linear = F.linear
matmul = torch.matmul
mm = torch.mm
bmm = torch.bmm
addmm = torch.addmm
einsum = torch.einsum
Linear = torch.nn.Linear
