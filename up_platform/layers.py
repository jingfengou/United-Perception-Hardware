"""Lazy module constructors; return the actual vendor module, not a wrapper module."""
from .factory import get_platform


class _DeferredDefault:
    def __init__(self, factory, args, kwargs):
        self.factory, self.args, self.kwargs = factory, args, kwargs
        self.module = None

    def resolve(self):
        if self.module is None:
            self.module = self.factory(*self.args, **self.kwargs)
        return self.module


def defer_default(factory, *args, **kwargs):
    """Defer a parameter-free, RNG-free default until a vendor is selected.

    Cache the actual module to preserve Python's shared-default identity. This
    descriptor is resolved before assignment and never becomes a model layer.
    """
    return _DeferredDefault(factory, args, kwargs)


def resolve_default(value):
    return value.resolve() if isinstance(value, _DeferredDefault) else value


def Hardswish(inplace=False):
    return get_platform().category('activation').Hardswish(inplace)


def Linear(in_features, out_features, bias=True, device=None, dtype=None):
    return get_platform().category("matrix").Linear(in_features=in_features, out_features=out_features, bias=bias, device=device, dtype=dtype)


def Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None):
    return get_platform().category("convolution").Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, device=device, dtype=dtype)


def ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None):
    return get_platform().category("convolution").ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias, dilation=dilation, padding_mode=padding_mode, device=device, dtype=dtype)


def BatchNorm1d(num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None):
    return get_platform().category("normalization").BatchNorm1d(num_features=num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats, device=device, dtype=dtype)


def BatchNorm2d(num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None):
    return get_platform().category("normalization").BatchNorm2d(num_features=num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats, device=device, dtype=dtype)


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, bias=True, device=None, dtype=None):
    return get_platform().category("normalization").LayerNorm(normalized_shape=normalized_shape, eps=eps, elementwise_affine=elementwise_affine, bias=bias, device=device, dtype=dtype)


def ReLU(inplace=False):
    return get_platform().category("activation").ReLU(inplace=inplace)


def ReLU6(inplace=False):
    return get_platform().category("activation").ReLU6(inplace=inplace)


def GELU(approximate='none'):
    return get_platform().category("activation").GELU(approximate=approximate)


def SiLU(inplace=False):
    return get_platform().category("activation").SiLU(inplace=inplace)


def SELU(inplace=False):
    return get_platform().category("activation").SELU(inplace=inplace)


def LeakyReLU(negative_slope=0.01, inplace=False):
    return get_platform().category("activation").LeakyReLU(negative_slope=negative_slope, inplace=inplace)


def Sigmoid():
    return get_platform().category("activation").Sigmoid()


def Tanh():
    return get_platform().category("activation").Tanh()


def Softmax(dim=None):
    return get_platform().category("activation").Softmax(dim=dim)


def Dropout(p=0.5, inplace=False):
    return get_platform().category("activation").Dropout(p=p, inplace=inplace)


def Dropout2d(p=0.5, inplace=False):
    return get_platform().category("activation").Dropout2d(p=p, inplace=inplace)


def AdaptiveAvgPool2d(output_size):
    return get_platform().category("pooling").AdaptiveAvgPool2d(output_size=output_size)


def AvgPool2d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
    return get_platform().category("pooling").AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode, count_include_pad=count_include_pad, divisor_override=divisor_override)


def MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
    return get_platform().category("pooling").MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode)


def Upsample(size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None):
    return get_platform().category("pooling").Upsample(size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners, recompute_scale_factor=recompute_scale_factor)


def UpsamplingBilinear2d(size=None, scale_factor=None):
    return get_platform().category("pooling").UpsamplingBilinear2d(size=size, scale_factor=scale_factor)


def ZeroPad2d(padding):
    return get_platform().category("pooling").ZeroPad2d(padding=padding)


def CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', label_smoothing=0.0):
    return get_platform().category("loss").CrossEntropyLoss(weight=weight, size_average=size_average, ignore_index=ignore_index, reduce=reduce, reduction=reduction, label_smoothing=label_smoothing)


def BCEWithLogitsLoss(weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None):
    return get_platform().category("loss").BCEWithLogitsLoss(weight=weight, size_average=size_average, reduce=reduce, reduction=reduction, pos_weight=pos_weight)


def KLDivLoss(size_average=None, reduce=None, reduction='mean', log_target=False):
    return get_platform().category("loss").KLDivLoss(size_average=size_average, reduce=reduce, reduction=reduction, log_target=log_target)


def L1Loss(size_average=None, reduce=None, reduction='mean'):
    return get_platform().category("loss").L1Loss(size_average=size_average, reduce=reduce, reduction=reduction)


def CosineSimilarity(dim=1, eps=1e-8):
    return get_platform().category("loss").CosineSimilarity(dim=dim, eps=eps)


def MultiheadAttention(embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=False, device=None, dtype=None):
    return get_platform().category("attention").MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, bias=bias, add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn, kdim=kdim, vdim=vdim, batch_first=batch_first, device=device, dtype=dtype)
