"""metax training: explicit reuse of vendor PyTorch, not the other purpose."""
from .torch_impl import TorchTraining


class MetaxTraining(TorchTraining):
    autocast = TorchTraining.autocast
    grad_scaler = TorchTraining.grad_scaler
    wrap_ddp = TorchTraining.wrap_ddp
    validation = TorchTraining.validation
    fused_optimizer = TorchTraining.fused_optimizer
    optimizer = TorchTraining.optimizer
    clip_grad_norm_ = TorchTraining.clip_grad_norm_
