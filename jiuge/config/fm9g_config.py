from dataclasses import dataclass

import torch.nn.functional as F
from megatron.core.transformer.transformer_config import TransformerConfig

@dataclass
class FM9GConfig(TransformerConfig):
    """
    Custom configuration for JIUGE FM9G model.
    """
    model_name: str = "fm9g"
    scale_embeddings: bool = True
    use_flash_attn: bool = False
    use_cache: bool = False  
    mask_att: bool = False
    mask_ffn: bool = False
    max_sequence_length: int = 4096
    
    # Position Embedding Strategy
    # Allowed: 'relative', 'rotary', 'chatglm_rotary'
    pos_bias_type: str = "relative"
    
    def __post_init__(self):
        super().__post_init__()

        # 1. GQA Check
        if self.num_query_groups >= self.num_attention_heads:
            raise ValueError(
                f"JIUGE {self.model_name} requires GQA. "
                f"num_query_groups ({self.num_query_groups}) < num_heads ({self.num_attention_heads})"
            )

        # 2. Activation Check
        supported_acts = [F.silu, F.gelu]
        if self.activation_func not in supported_acts:
            raise ValueError(f"Unsupported activation: {self.activation_func}")

        # 3. Position Bias Type Check
        allowed_pos_types = ['relative', 'rotary', 'chatglm_rotary']
        if self.pos_bias_type is None or self.pos_bias_type.lower() not in allowed_pos_types:
            raise ValueError(f"Invalid pos_bias_type: {self.pos_bias_type}. Must be one of {allowed_pos_types}")