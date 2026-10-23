"""PyTorch CUDA-shaped controls shared explicitly by compatible vendors."""
import contextlib

import torch


class CudaConvolutionControls:
    @property
    def _convolution_library_enabled(self):
        return torch.backends.cudnn.enabled

    def configure_convolution(self, enabled=True):
        torch.backends.cudnn.enabled = self._numerical_policy.convolution_enabled(enabled)

    @contextlib.contextmanager
    def convolution_flags(self, enabled=True):
        with torch.backends.cudnn.flags(enabled=self._numerical_policy.convolution_enabled(enabled)):
            yield
