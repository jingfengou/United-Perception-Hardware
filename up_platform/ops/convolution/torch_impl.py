"""Explicit bindings to the installed vendor PyTorch; no device transfer."""
import torch
from torch.nn import functional as F

conv2d = F.conv2d
conv_transpose2d = F.conv_transpose2d
Conv2d = torch.nn.Conv2d
ConvTranspose2d = torch.nn.ConvTranspose2d
