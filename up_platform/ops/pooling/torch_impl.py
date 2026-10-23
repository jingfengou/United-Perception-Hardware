"""Explicit bindings to the installed vendor PyTorch; no device transfer."""
import torch
from torch.nn import functional as F

adaptive_avg_pool2d = F.adaptive_avg_pool2d
max_pool2d = F.max_pool2d
interpolate = F.interpolate
pad = F.pad
AdaptiveAvgPool2d = torch.nn.AdaptiveAvgPool2d
AvgPool2d = torch.nn.AvgPool2d
MaxPool2d = torch.nn.MaxPool2d
Upsample = torch.nn.Upsample
UpsamplingBilinear2d = torch.nn.UpsamplingBilinear2d
ZeroPad2d = torch.nn.ZeroPad2d
