"""Explicit bindings to the installed vendor PyTorch; no device transfer."""
import torch
from torch.nn.modules._functions import SyncBatchNorm as sync_batch_norm_function
batch_norm_module_forward = torch.nn.BatchNorm2d.forward
sync_batch_norm_forward = torch.nn.SyncBatchNorm.forward
sync_batch_norm = sync_batch_norm_function.apply
from torch.nn import functional as F

batch_norm = F.batch_norm
layer_norm = F.layer_norm
group_norm = F.group_norm
normalize = F.normalize
BatchNorm1d = torch.nn.BatchNorm1d
BatchNorm2d = torch.nn.BatchNorm2d
LayerNorm = torch.nn.LayerNorm
