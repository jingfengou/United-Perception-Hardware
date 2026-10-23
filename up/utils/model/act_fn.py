from up_platform import get_platform
from up_platform import layers as platform_layers
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * get_platform().category('activation').sigmoid(x)


class Hardswish(nn.Module):  # export-friendly version of nn.Hardswish()
    @staticmethod
    def forward(x):
        # return x * F.hardsigmoid(x)  # for torchscript and CoreML
        return x * get_platform().category('activation').hardtanh(x + 3, 0., 6.) / 6.  # for torchscript, CoreML and ONNX


class GELU(nn.Module):
    @staticmethod
    def forward(x):
        erf = get_platform().category('activation').tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3)))
        return 0.5 * x * (1 + erf)


_act_cfg = {
    'Hardswish': ('act', platform_layers.Hardswish if hasattr(torch.nn, 'Hardswish') else Hardswish),
    'LeakyReLU': ('act', platform_layers.LeakyReLU),
    'ReLU': ('act', platform_layers.ReLU),
    'Identity': ('act', torch.nn.Identity),
    'Silu': ('act', platform_layers.SiLU if hasattr(torch.nn, 'SiLU') else SiLU),
    'GELU': ('act', platform_layers.GELU if hasattr(torch.nn, 'GELU') else GELU)
}


def build_act_fn(cfg, postfix=''):
    assert isinstance(cfg, dict) and 'type' in cfg
    cfg = cfg.copy()
    act_type = cfg.pop('type')
    kwargs = cfg.get('kwargs', {})

    if act_type not in _act_cfg:
        raise KeyError('Unrecognized act type {}'.format(act_type))
    else:
        abbr, act_layer = _act_cfg[act_type]
        if act_layer is None:
            raise NotImplementedError

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    layer = act_layer(**kwargs)
    return name, layer
