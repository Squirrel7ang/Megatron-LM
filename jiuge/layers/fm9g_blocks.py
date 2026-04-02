from typing import Optional, Tuple, Union
import torch
import torch.nn as nn

# Assuming these are updated versions that accept config
from jiuge.layers.fm9g_attention import Attention
from jiuge.layers.fm9g_feedforward import FeedForward
from jiuge.layers.fm9g_layernorm import LayerNorm
from jiuge.layers.fm9g_position_embedding import RotaryEmbedding, RotaryEmbeddingESM
from jiuge.config.fm9g_config import FM9GConfig

from megatron.core.transformer.module import MegatronModule
from megatron.core.process_groups_config import ProcessGroupCollection

class SelfAttentionBlock(MegatronModule):
    """Megatron-style self-attention block with RMS LayerNorm, attention, residual connection."""

    def __init__(
        self,
        config: FM9GConfig,
        # dim_model: int,
        # num_heads: int,
        # num_kv_heads: int,
        # dim_head: int,
        # eps: float = 1e-5,
        # add_qkv_bias: bool = False,
        # use_flash_attn: bool = False,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super().__init__(config=config)

        # Use updated LayerNorm with config
        self.layernorm_before_attention = LayerNorm(
            config=config,
            hidden_size=config.hidden_size,
        )

        # Attention handles its own TP via config and pg_collection
        self.self_attention = Attention(
            config=config,
            # dim_model=dim_model,
            # num_heads=num_heads,
            # num_kv_heads=num_kv_heads,
            # dim_head=dim_head,
            # add_qkv_bias=add_qkv_bias,
            # use_flash_attn=use_flash_attn,
            pg_collection=pg_collection,
        )

        self.dropout = nn.Dropout(config.attention_dropout) if config.attention_dropout > 0 else None
        self.use_cache = config.use_cache

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[Union[torch.Tensor, "RotaryEmbedding", "RotaryEmbeddingESM"]] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        length_mask: Optional[torch.Tensor] = None,
        attention_mask_bias: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        # 1. Pre-norm
        x = self.layernorm_before_attention(hidden_states)

        # 2. Self Attention
        x = self.self_attention(
            hidden_q=x,
            hidden_kv=x,
            attention_mask=attention_mask,
            position_bias=position_bias,
            past_kv=past_key_value,
            length_mask=length_mask,
            attention_mask_bias=attention_mask_bias,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            position_ids=position_ids,
        )

        if self.use_cache:
            x, current_key_value = x
        else:
            current_key_value = None

        if self.dropout is not None:
            x = self.dropout(x)

        # 3. Residual connection
        hidden_states = hidden_states + x

        return (hidden_states, current_key_value) if self.use_cache else hidden_states


class FFNBlock(MegatronModule):
    """Megatron-style feed-forward block with RMS LayerNorm, feed-forward, residual connection."""

    def __init__(
        self,
        config: FM9GConfig,
        # dim_model: int,
        # dim_ff: int,
        # activate_fn: str = "gelu",
        # eps: float = 1e-6,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super().__init__(config=config)

        self.layernorm_before_ffn = LayerNorm(
            config=config,
            hidden_size=config.hidden_size,
        )

        self.ffn = FeedForward(
            config=config,
            # dim_model=dim_model,
            # dim_ff=dim_ff,
            # activate_fn=activate_fn,
            pg_collection=pg_collection,
        )

        self.dropout = torch.nn.Dropout(config.hidden_dropout) if config.hidden_dropout > 0 else None

    def forward(self, hidden_states: torch.Tensor):
        x = self.layernorm_before_ffn(hidden_states)
        x = self.ffn(x)

        if self.dropout is not None:
            x = self.dropout(x)

        # Residual connection
        hidden_states = hidden_states + x
        return hidden_states


class TransformerLayer(MegatronModule):
    """Transformer layer combining attention and ffn."""

    def __init__(
        self,
        config: FM9GConfig,
        # dim_model: int,
        # dim_ff: int,
        # num_heads: int,
        # num_kv_heads: int,
        # dim_head: int,
        # activate_fn: str = "gelu",
        # eps: float = 1e-6,
        # add_qkv_bias: bool = False,
        # mask_att: bool = False,
        # mask_ffn: bool = False,
        # use_flash_attn: bool = False,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super().__init__(config=config)
        self.mask_att = config.mask_att
        self.mask_ffn = config.mask_ffn
        self.use_cache = config.use_cache

        if not self.mask_att:
            self.self_att = SelfAttentionBlock(
                config=config,
                # dim_model=dim_model,
                # num_heads=num_heads,
                # num_kv_heads=num_kv_heads,
                # dim_head=dim_head,
                # eps=eps,
                # add_qkv_bias=add_qkv_bias,
                # use_flash_attn=use_flash_attn,
                pg_collection=pg_collection,
            )

        if not self.mask_ffn:
            self.ffn = FFNBlock(
                config=config,
                # dim_model=dim_model,
                # dim_ff=dim_ff,
                # activate_fn=activate_fn,
                # eps=eps,
                pg_collection=pg_collection,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[Union[torch.Tensor, "RotaryEmbedding", "RotaryEmbeddingESM"]] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        length_mask: Optional[torch.Tensor] = None,
        attention_mask_bias: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        """
        JIUGE native TransformerLayer forward.
        Explicitly passing arguments instead of using **kwargs.
        """
        # 1. Attention layer
        current_key_value = None
        if not self.mask_att:
            # Jiuge native call to SelfAttentionBlock
            res = self.self_att(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_bias=position_bias,
                past_key_value=past_key_value,
                length_mask=length_mask,
                attention_mask_bias=attention_mask_bias,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                position_ids=position_ids,
            )
            
            if self.use_cache:
                hidden_states, current_key_value = res
            else:
                hidden_states = res

        # 2. FFN layer
        if not self.mask_ffn:
            # Jiuge native call to FFNBlock
            hidden_states = self.ffn(hidden_states)

        return (hidden_states, current_key_value) if self.use_cache else hidden_states