from abc import ABC, abstractmethod


class TrainingBackend(ABC):
    @abstractmethod
    def autocast(self, enabled=True, dtype=None, cache_enabled=True):
        """Return a context manager. Do not change persistent precision defaults."""

    @abstractmethod
    def grad_scaler(self, init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000, enabled=True):
        """Return a scaler supporting step/update and original state_dict keys."""

    @abstractmethod
    def wrap_ddp(self, module, **kwargs):
        """Return distributed wrapper, passing original DDP kwargs unchanged."""

    @abstractmethod
    def validation(self, module):
        """eval/no_grad context; restore every submodule's entry mode on success/error."""

    @abstractmethod
    def fused_optimizer(self, name, params, **kwargs):
        """Construct the requested vendor optimizer; no replacement by an unfused algorithm."""

    @abstractmethod
    def optimizer(self, name, **kwargs):
        """Construct the original torch.optim class with original parameter groups."""

    @abstractmethod
    def clip_grad_norm_(self, parameters, max_norm, norm_type=2.0, **kwargs):
        """Original in-place gradient clipping and returned total norm."""
