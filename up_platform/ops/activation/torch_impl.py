"""Explicit bindings to the installed vendor PyTorch; no device transfer."""
import torch
Hardswish = torch.nn.Hardswish
from torch.nn import functional as F

relu = F.relu
relu6 = F.relu6
hardtanh = F.hardtanh
gelu = F.gelu
silu = F.silu
sigmoid = torch.sigmoid
tanh = torch.tanh
softmax = F.softmax
log_softmax = F.log_softmax
softplus = F.softplus
ReLU = torch.nn.ReLU
ReLU6 = torch.nn.ReLU6
GELU = torch.nn.GELU
SiLU = torch.nn.SiLU
SELU = torch.nn.SELU
LeakyReLU = torch.nn.LeakyReLU
Sigmoid = torch.nn.Sigmoid
Tanh = torch.nn.Tanh
Softmax = torch.nn.Softmax
Dropout = torch.nn.Dropout
Dropout2d = torch.nn.Dropout2d
