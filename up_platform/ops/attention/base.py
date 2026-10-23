"""Complete attention operation contracts, independent of UP and vendor SDKs."""
from abc import ABC, abstractmethod
from up_platform.contracts import operation


class AttentionOps(ABC):
    @staticmethod
    @operation(source="torch.nn.MultiheadAttention", output="Original nn.MultiheadAttention module, weights, dropout, masks and state_dict; no implicit FlashAttention replacement.",
               gradients="query, key, value and parameters", state="Construct the original module with original parameter identities, initialization and RNG consumption. Running stats and RNG change only as in upstream forward.")
    @abstractmethod
    def MultiheadAttention(embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=False, device=None, dtype=None):
        """Original nn.MultiheadAttention module, weights, dropout, masks and state_dict; no implicit FlashAttention replacement."""
        raise NotImplementedError
