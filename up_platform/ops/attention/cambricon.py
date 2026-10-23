"""cambricon template: no operation in this category is claimed as supported."""
from .base import AttentionOps
from up_platform.contracts import UnsupportedOperation


class CambriconAttentionOps(AttentionOps):
    dependencies = ()
    @staticmethod
    def MultiheadAttention(embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=False, device=None, dtype=None):
        raise UnsupportedOperation("cambricon", "unverified-template", "attention.MultiheadAttention", "Implement and validate this contract before enabling the vendor")

    MultiheadAttention.__func__.unsupported_reason = "Vendor template; not implemented"
