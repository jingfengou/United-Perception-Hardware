"""Explicit bindings to the installed vendor PyTorch; no device transfer."""
import torch
from torch.nn import functional as F

MultiheadAttention = torch.nn.MultiheadAttention
