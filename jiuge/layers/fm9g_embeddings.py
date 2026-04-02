import math
from typing import Optional

import torch
import torch.nn.functional as F
from megatron.core.transformer.module import MegatronModule
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import VocabParallelEmbedding
from megatron.core.tensor_parallel.layers import linear_with_grad_accumulation_and_async_allreduce
from megatron.core import tensor_parallel

from jiuge.layers.fm9g_position_embedding import RotaryEmbedding
from jiuge.config.fm9g_config import FM9GConfig


class Embedding(MegatronModule):
    """Standard embedding layer with optional scaling, Megatron style."""

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        dtype: torch.dtype = torch.half,
        scale: bool = True,
        init_mean: float = 0.0,
        init_std: float = 1.0,
    ):
        super().__init__(config=None)  # config not required for basic embedding
        self.dim_model = embedding_size
        self.scale = scale

        # Use a normal initialization
        self.weight = torch.nn.Parameter(
            torch.empty(vocab_size, embedding_size, dtype=dtype)
        )
        torch.nn.init.normal_(self.weight, mean=init_mean, std=init_std)

    def forward(self, ids: torch.Tensor):
        """Forward pass: embedding lookup"""
        embeds = F.embedding(ids, self.weight)
        if self.scale:
            embeds = embeds / math.sqrt(self.dim_model)
        return embeds

    def projection(self, x: torch.Tensor):
        """Project embedding back to vocab logits"""
        if self.scale:
            x_scaled = x / math.sqrt(self.dim_model)
        else:
            x_scaled = x
        return F.linear(x_scaled, self.weight)


class EmbeddingExt(MegatronModule):
    """Embedding with rotary embedding support."""

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        dtype: torch.dtype = torch.half,
        init_mean: float = 0.0,
        init_std: float = 1.0,
        distance_scale: int = 16,
    ):
        super().__init__(config=None)
        self.dim_model = embedding_size
        self.rotary_emb = RotaryEmbedding(
            dim=embedding_size, distance_scale=distance_scale, dtype=dtype
        )

        self.weight = torch.nn.Parameter(
            torch.empty(vocab_size, embedding_size, dtype=dtype)
        )
        torch.nn.init.normal_(self.weight, mean=init_mean, std=init_std)

    def forward(self, ids: torch.Tensor, ids_sub: torch.Tensor):
        """
        Args:
            ids: (batch, seq_len) token ids
            ids_sub: (batch,) sequence indices for rotary embedding
        Returns:
            (batch, seq_len, embedding_size)
        """
        embeds = F.embedding(ids, self.weight) / math.sqrt(self.dim_model)
        return self.rotary_emb(embeds, ids_sub)

    def projection(self, x: torch.Tensor, ext_table: Optional[torch.Tensor] = None):
        """
        Project embedding back to vocab logits with optional extended vocab
        """
        logits = F.linear(x / math.sqrt(self.dim_model), self.weight)
        if ext_table is not None:
            logits_ext = F.linear(x, ext_table)
            logits = torch.cat([logits, logits_ext], dim=-1)
        return logits
    

class FM9GEmbedding(MegatronModule):
    """
    Megatron-compatible vocab-parallel embedding layer.
    Fully aligned with TransformerConfig fields.
    """

    def __init__(
        self,
        config: FM9GConfig,
        vocab_size: int,
        # 'embedding_size' matches config.hidden_size
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        # 1. Properly pass pg_collection to base if supported, 
        # or it will be accessible via self.pg_collection after this call
        super().__init__(config=config)
        if pg_collection is not None:
            self.pg_collection = pg_collection

        # Align with TransformerConfig: hidden_size
        self.hidden_size = config.hidden_size
        
        # JIUGE specific scaling flag
        self.scale = getattr(config, 'scale_embeddings', True)
        
        # Get TP group from collection or parallel_state
        tp_group = self.pg_collection.tp if hasattr(self, 'pg_collection') and self.pg_collection else None

        # 2. Megatron vocab-parallel embedding
        self.word_embeddings = VocabParallelEmbedding(
            num_embeddings=vocab_size,
            embedding_dim=self.hidden_size,
            config=config,
            init_method=config.init_method,
            tp_group=tp_group,
        )

    def forward(self, input_ids: torch.Tensor):
        embeds = self.word_embeddings(input_ids)

        if self.scale:
            # FM9G specific scaling: x / sqrt(d)
            # Note: Standard Transformer usually does x * sqrt(d) or no scale
            embeds = embeds / math.sqrt(self.hidden_size)

        if self.config.sequence_parallel:
            embeds = tensor_parallel.scatter_to_sequence_parallel_region(embeds)

        return embeds

    def projection(self, hidden_states: torch.Tensor):
        """
        Project hidden states back to vocab logits using tied weights.
        Used when share_embeddings_and_output_weights is True.
        """
        if self.scale:
            hidden_states = hidden_states / math.sqrt(self.hidden_size)

        # 3. Use Megatron's optimized linear function
        # All fields below are standard in TransformerConfig
        logits = linear_with_grad_accumulation_and_async_allreduce(
            input=hidden_states,
            weight=self.word_embeddings.weight,
            bias=None,
            gradient_accumulation_fusion=self.config.gradient_accumulation_fusion,
            async_grad_allreduce=self.config.async_tensor_model_parallel_allreduce,
            sequence_parallel=self.config.sequence_parallel,
        )
        return logits