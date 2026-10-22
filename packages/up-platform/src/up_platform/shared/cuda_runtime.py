"""PyTorch CUDA-shaped controls shared explicitly by compatible vendors."""
import contextlib

import torch


class CudaConvolutionControls:
    def configure_convolution(self, enabled=True):
        torch.backends.cudnn.enabled = enabled if self._numerics is None else False

    @contextlib.contextmanager
    def convolution_flags(self, enabled=True):
        with torch.backends.cudnn.flags(enabled=enabled if self._numerics is None else False):
            yield
