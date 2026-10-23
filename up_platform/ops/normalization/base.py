"""Complete normalization operation contracts, independent of UP and vendor SDKs."""
from abc import ABC, abstractmethod
from up_platform.contracts import operation


class NormalizationOps(ABC):
    @staticmethod
    @operation(source="torch.nn.BatchNorm2d.forward", output="Original PyTorch result.", gradients="Original input and parameter gradients.", state="Original module and autograd state updates.")
    @abstractmethod
    def batch_norm_module_forward(module, input):
        raise NotImplementedError

    @staticmethod
    @operation(source="torch.nn.SyncBatchNorm.forward", output="Original PyTorch result.", gradients="Original input and parameter gradients.", state="Original module and autograd state updates.")
    @abstractmethod
    def sync_batch_norm_forward(module, input):
        raise NotImplementedError

    @staticmethod
    @operation(source="sync_batch_norm_function.apply", output="Original PyTorch result.", gradients="Original input and parameter gradients.", state="Original module and autograd state updates.")
    @abstractmethod
    def sync_batch_norm(input, weight, bias, running_mean, running_var, eps, momentum, process_group, world_size):
        raise NotImplementedError

    @staticmethod
    @operation(source="torch.nn.functional.batch_norm", output="Tensor same shape as input; mutate running stats in training exactly as upstream.",
               gradients="input, weight, bias; not running statistics", state="Update running_mean/running_var only when training=True, in place.")
    @abstractmethod
    def batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5):
        """Tensor same shape as input; mutate running stats in training exactly as upstream."""
        raise NotImplementedError

    @staticmethod
    @operation(source="torch.nn.functional.layer_norm", output="Tensor same shape; normalize trailing normalized_shape dimensions.",
               gradients="input, weight, bias", state="Preserve original aliasing and RNG behavior; no additional persistent state.")
    @abstractmethod
    def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
        """Tensor same shape; normalize trailing normalized_shape dimensions."""
        raise NotImplementedError

    @staticmethod
    @operation(source="torch.nn.functional.group_norm", output="Tensor same NCHW shape; num_groups must divide C.",
               gradients="input, weight, bias", state="Preserve original aliasing and RNG behavior; no additional persistent state.")
    @abstractmethod
    def group_norm(input, num_groups, weight=None, bias=None, eps=1e-5):
        """Tensor same NCHW shape; num_groups must divide C."""
        raise NotImplementedError

    @staticmethod
    @operation(source="torch.nn.functional.normalize", output="Same-shape Lp normalization with original eps and dimension.",
               gradients="input unless out is used", state="Preserve original aliasing and RNG behavior; no additional persistent state.")
    @abstractmethod
    def normalize(input, p=2.0, dim=1, eps=1e-12, out=None):
        """Same-shape Lp normalization with original eps and dimension."""
        raise NotImplementedError

    @staticmethod
    @operation(source="torch.nn.BatchNorm1d", output="nn.BatchNorm1d-compatible module; preserve running_mean/var and num_batches_tracked.",
               gradients="input and parameters", state="Construct the original module with original parameter identities, initialization and RNG consumption. Running stats and RNG change only as in upstream forward.")
    @abstractmethod
    def BatchNorm1d(num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None):
        """nn.BatchNorm1d-compatible module; preserve running_mean/var and num_batches_tracked."""
        raise NotImplementedError

    @staticmethod
    @operation(source="torch.nn.BatchNorm2d", output="nn.BatchNorm2d-compatible module; preserve running_mean/var and num_batches_tracked.",
               gradients="input and parameters", state="Construct the original module with original parameter identities, initialization and RNG consumption. Running stats and RNG change only as in upstream forward.")
    @abstractmethod
    def BatchNorm2d(num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None):
        """nn.BatchNorm2d-compatible module; preserve running_mean/var and num_batches_tracked."""
        raise NotImplementedError

    @staticmethod
    @operation(source="torch.nn.LayerNorm", output="nn.LayerNorm-compatible module with original parameter names and initialization.",
               gradients="input and parameters", state="Construct the original module with original parameter identities, initialization and RNG consumption. Running stats and RNG change only as in upstream forward.")
    @abstractmethod
    def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, bias=True, device=None, dtype=None):
        """nn.LayerNorm-compatible module with original parameter names and initialization."""
        raise NotImplementedError
