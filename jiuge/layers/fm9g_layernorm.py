import torch
import torch.nn as nn
from megatron.core.transformer.module import MegatronModule
from jiuge.config.fm9g_config import FM9GConfig

@torch.jit.script
def rms_layernorm(hidden: torch.Tensor, weight: torch.Tensor, eps: float):
    """
    RMS LayerNorm computation with JIT acceleration.
    Calculated as: x * rsqrt(mean(x^2) + eps) * weight
    """
    old_dtype = hidden.dtype
    # Use float32 for variance to avoid overflow/precision issues
    # Note: Megatron-core often does this in FP32 for stability
    variance = hidden.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    hidden = (hidden * torch.rsqrt(variance + eps)).to(old_dtype)
    return hidden * weight


class LayerNorm(MegatronModule):
    """
    RMS LayerNorm (Megatron-style)
    Custom implementation aligned with Megatron-core TransformerConfig.
    """

    def __init__(
        self,
        config: FM9GConfig,
        hidden_size: int, # Traditionally 'dim_norm'
        init_var: float = 1.0,
    ):
        super().__init__(config=config)
        
        # Align with standard TransformerConfig field names
        self.eps = config.layernorm_epsilon if hasattr(config, 'layernorm_epsilon') else 1e-5
        self.hidden_size = hidden_size
        
        # In Megatron, LayerNorm weights are replicated across TP ranks.
        self.weight = nn.Parameter(
            torch.full((self.hidden_size,), init_var, dtype=config.params_dtype)
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass for RMS LayerNorm.
        """
        # Verification (Optional but good for debugging PP/TP tensor shapes)
        if x.size(-1) != self.hidden_size:
            raise ValueError(
                f"Input feature dim {x.size(-1)} != expected {self.hidden_size}"
            )
            
        return rms_layernorm(x, self.weight, self.eps)