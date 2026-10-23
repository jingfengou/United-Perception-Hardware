from pathlib import Path
import subprocess
import sys

import torch


def test_generated_plugin_reuses_torch_and_overrides_one_operation(tmp_path):
    root = Path(__file__).resolve().parents[2]
    plugin = tmp_path/'sample_gpu'
    subprocess.run([sys.executable, str(root/'tools/new_platform_vendor.py'), '--vendor', 'sample_gpu',
                    '--device', 'cpu', '--output', str(plugin), '--reuse-torch'], check=True)
    # A real backend would use its registered device name, not this test-only CPU.
    with (plugin/'ops/matrix.py').open('a') as file:
        file.write('\ndef linear(input, weight, bias=None):\n    return torch_impl.linear(input, weight, bias)\nSampleGpuMatrixOps.linear = staticmethod(linear)\n')
    source = f'''import sys
sys.path.insert(0, {str(tmp_path)!r})
from up_platform import configure_platform
import torch
p = configure_platform({{"backend":"sample_gpu","plugins":["sample_gpu"]}},purpose="training")
x = torch.ones(2,3,requires_grad=True)
y = p.category("matrix").linear(x,torch.ones(4,3,requires_grad=True))
y.sum().backward()
assert y.shape == (2,4) and x.grad is not None
assert p.category("matrix").linear.__module__ == "sample_gpu.ops.matrix"
from up_platform.ops.matrix import torch_impl
assert p.category("matrix").mm is torch_impl.mm
assert not p.capabilities()["detection.naive_nms"].registered
'''
    subprocess.run([sys.executable, '-c', source], check=True, cwd=root)


def test_sdk_example_checks_full_backward_and_optional_bias():
    from up_platform.templates.sdk_linear import bind_sdk_linear
    class TestKernels:
        # CPU float64 kernels are a test double, never a hardware implementation.
        @staticmethod
        def linear_forward(x, w, b):
            return torch.nn.functional.linear(x, w, b)
        @staticmethod
        def linear_backward(grad, x, w, has_bias, needs):
            flat_grad, flat_x = grad.reshape(-1, w.shape[0]), x.reshape(-1, w.shape[1])
            return (grad @ w if needs[0] else None,
                    flat_grad.T @ flat_x if needs[1] else None,
                    flat_grad.sum(0) if has_bias and needs[2] else None)
    linear = bind_sdk_linear(TestKernels)
    x = torch.randn(2, 2, 3, dtype=torch.float64, requires_grad=True)
    w = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    b = torch.randn(4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(linear, (x, w, b))
    assert torch.autograd.gradcheck(linear, (x, w, None))
