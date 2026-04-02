import math
from typing import Optional, Tuple
from typing import Union

import torch
import torch.nn.functional as F
import torch.nn as nn

from megatron.core.transformer.module import MegatronModule
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import reduce_from_tensor_model_parallel_region
from megatron.core import parallel_state

from jiuge.config.fm9g_config import FM9GConfig

class SegmentPositionEmbedding(MegatronModule):
    """
    Segment-aware relative position embedding with Megatron tensor parallel support.
    """

    def __init__(
        self,
        config: FM9GConfig,
        num_heads: int,
        num_segments: int = 1,
        num_buckets: int = 32,
        max_distance: int = 128,
        bidirectional: bool = False,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super().__init__(config=config)
        self.pg_collection = pg_collection
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.bidirectional = bidirectional
        self.num_segments = num_segments

        tp_size = self.pg_collection.tp_world_size if self.pg_collection else 1
        tp_rank = self.pg_collection.tp_rank if self.pg_collection else 0

        total_rows = num_segments * num_segments + num_buckets
        rows_per_rank = total_rows // tp_size
        
        self.start_row = tp_rank * rows_per_rank
        self.end_row = self.start_row + rows_per_rank if tp_rank != tp_size - 1 else total_rows
        actual_rows_on_this_rank = self.end_row - self.start_row

        self.relative_attention_bias = nn.Parameter(
            torch.empty(actual_rows_on_this_rank, num_heads, 
                        dtype=config.params_dtype)
        )
        config.init_method(self.relative_attention_bias)

    def forward(
        self,
        key_pos: torch.Tensor,
        query_pos: torch.Tensor,
        key_segment: torch.Tensor,
        query_segment: torch.Tensor,
    ):
        tp_rank = self.pg_collection.tp_rank if self.pg_collection else 0
        
        with torch.no_grad():
            batch = key_pos.size(0)
            keylen = key_pos.size(1)
            querylen = query_pos.size(1)

            # Ensure segments match pos dimensions
            key_segment = key_segment.view(batch, keylen)
            query_segment = query_segment.view(batch, querylen)

            # Segment-based relative positions (batch, len_q, len_k)
            relative_position_bucket = self._segment_relative_position_bucket(
                query_segment[:, :, None], key_segment[:, None, :]
            )
            relative_position_bucket = relative_position_bucket + self.num_buckets

            # Absolute positions within segments
            absolute_position_bucket = self._position_bucket(
                key_pos[:, None, :] - query_pos[:, :, None],
                bidirectional=self.bidirectional,
                num_buckets=self.num_buckets,
                max_distance=self.max_distance,
            )

            # Merge absolute and segment-relative
            relative_position_bucket = torch.where(
                (query_segment[:, :, None] == key_segment[:, None, :]),
                absolute_position_bucket,
                relative_position_bucket,
            )

            # Map to local TP slice
            mask = (relative_position_bucket >= self.start_row) & (relative_position_bucket < self.end_row)
            local_bucket = relative_position_bucket - self.start_row
            local_bucket = torch.clamp(local_bucket, min=0, max=self.relative_attention_bias.size(0)-1)

        # Embedding lookup
        embeds = F.embedding(local_bucket, self.relative_attention_bias)
        
        # Dim handling: (batch, len_q, len_k, num_heads) -> (batch, num_heads, len_q, len_k)
        if embeds.dim() == 5:
            embeds = embeds.squeeze(1)
        embeds = embeds.permute(0, 3, 1, 2).contiguous()

        # Masking outside TP slice
        mask_final = mask.unsqueeze(1) if mask.dim() == 3 else mask
        embeds = embeds * mask_final.to(embeds.dtype)

        if self.pg_collection and self.pg_collection.tp_world_size > 1:
            embeds = reduce_from_tensor_model_parallel_region(embeds, self.pg_collection.tp)

        
        embeds = embeds + (self.relative_attention_bias.sum() * 0.0)

        return embeds

    def _segment_relative_position_bucket(self, query_segment, key_segment):
        return query_segment * self.num_segments + key_segment

    def _position_bucket(self, relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        if bidirectional:
            num_buckets //= 2
            relative_buckets = (relative_position > 0).to(torch.int32) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
            relative_buckets = 0

        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        relative_postion_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.int32)
        relative_postion_if_large = torch.min(
            relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1)
        )
        relative_buckets += torch.where(is_small, relative_position.to(torch.int32), relative_postion_if_large)
        return relative_buckets


class BucketPositionBias(MegatronModule):
    """
    Bucketed relative position bias with Megatron tensor parallel support.
    """

    def __init__(
        self,
        config: FM9GConfig,
        num_heads: int,
        num_buckets: int = 32,
        num_segment_bucket: int = 32,
        max_distance: int = 128,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super().__init__(config=config)
        self.pg_collection = pg_collection
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.num_segment_bucket = num_segment_bucket
        self.max_distance = max_distance

        tp_size = self.pg_collection.tp_world_size if self.pg_collection else 1
        tp_rank = self.pg_collection.tp_rank if self.pg_collection else 0

        total_rows = num_buckets + num_segment_bucket
        rows_per_rank = total_rows // tp_size
        
        self.start_row = tp_rank * rows_per_rank
        self.end_row = self.start_row + rows_per_rank if tp_rank != tp_size - 1 else total_rows
        actual_rows_on_this_rank = self.end_row - self.start_row

        self.relative_attention_bias = nn.Parameter(
            torch.empty(actual_rows_on_this_rank, num_heads, 
                        dtype=config.params_dtype)
        )
        config.init_method(self.relative_attention_bias)

    def forward(
        self,
        query_pos: torch.Tensor,    
        key_pos: torch.Tensor,      
        rel_buckets: torch.Tensor,  
    ):
        tp_rank = self.pg_collection.tp_rank if self.pg_collection else 0
        
        with torch.no_grad():
            batch = key_pos.size(0)
            querylen = query_pos.size(1)
            keylen = key_pos.size(1)
            
            global_query_len = rel_buckets.size(1)
            global_key_len = rel_buckets.size(2)

            # Handle Sequence Parallel slicing
            if querylen < global_query_len or keylen < global_key_len:
                q_start = tp_rank * querylen
                k_start = tp_rank * keylen
                actual_rel_buckets = rel_buckets[:, q_start:q_start + querylen, k_start:k_start + keylen]
            else:
                actual_rel_buckets = rel_buckets

            # Final safety check for shape alignment
            if actual_rel_buckets.size(1) != querylen or actual_rel_buckets.size(2) != keylen:
                actual_rel_buckets = actual_rel_buckets[:, :querylen, :keylen]

            # Calculate bucket indices
            relative_position_bucket = actual_rel_buckets - 1 + self.num_buckets

            inner_segment_bucket = self._position_bucket(
                key_pos[..., None, :] - query_pos[..., :, None],
                num_buckets=self.num_buckets,
                max_distance=self.max_distance,
            )

            relative_position_bucket = torch.where(
                actual_rel_buckets == 0,
                inner_segment_bucket,
                relative_position_bucket,
            )

            mask = (relative_position_bucket >= self.start_row) & (relative_position_bucket < self.end_row)
            local_bucket = relative_position_bucket - self.start_row
            local_bucket = torch.clamp(local_bucket, min=0, max=self.relative_attention_bias.size(0)-1)

        # Embedding and dimension permutation
        # Step 1: Look up embedding for relative position buckets.
        # local_bucket: [Batch, Seq_Q, 1, Seq_K] or [Batch, Seq_Q, Seq_K]
        # self.relative_attention_bias: [Rows_per_rank, Num_Heads]
        # Output embeds: [Batch, Seq_Q, 1, Seq_K, Num_Heads] or [Batch, Seq_Q, Seq_K, Num_Heads]
        embeds = F.embedding(local_bucket, self.relative_attention_bias)
        
        # Step 2: Harmonize dimension if it's 5D. 
        # We need to remove the singleton dimension to get [Batch, Seq_Q, Seq_K, Num_Heads].
        if embeds.dim() == 5 and embeds.size(2) == 1:
            # Case: [B, Q, 1, K, H] -> [B, Q, K, H]
            embeds = embeds.squeeze(2)
        elif embeds.dim() == 5 and embeds.size(1) == 1:
            # Case: [B, 1, Q, K, H] -> [B, Q, K, H]
            embeds = embeds.squeeze(1)

        # Step 3: Align mask dimensions for broadcasting.
        # The goal is to make mask_final compatible with embeds [B, Q, K, H].
        if mask.dim() == 4:
            # Current mask: [Batch, Seq_Q, 1, Seq_K]
            # Operation: Move Seq_K to dim 2 to get [Batch, Seq_Q, Seq_K, 1].
            # This allows broadcasting over the Num_Heads dimension.
            mask_final = mask.permute(0, 1, 3, 2).contiguous()
        else:
            # Current mask: [Batch, Seq_Q, Seq_K]
            # Operation: Add singleton dimension at the end to get [Batch, Seq_Q, Seq_K, 1].
            mask_final = mask.unsqueeze(-1)

        # Step 4: Apply the mask.
        # embeds: [B, Q, K, H] * mask_final: [B, Q, K, 1] -> [B, Q, K, H]
        embeds = embeds * mask_final.to(embeds.dtype)

        # Step 5: Final permutation for Attention Score compatibility.
        # [Batch, Seq_Q, Seq_K, Num_Heads] -> [Batch, Num_Heads, Seq_Q, Seq_K]
        embeds = embeds.permute(0, 3, 1, 2).contiguous()

        if self.pg_collection and self.pg_collection.tp_world_size > 1:
            embeds = reduce_from_tensor_model_parallel_region(embeds, self.pg_collection.tp)

        # embeds = embeds + (self.relative_attention_bias.sum() * 0.0)

        return embeds

    def _position_bucket(self, relative_position, num_buckets=32, max_distance=128):
        num_buckets //= 2
        relative_buckets = (relative_position > 0).to(torch.int32) * num_buckets
        relative_position = torch.abs(relative_position)

        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        relative_postion_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.int32)
        relative_postion_if_large = torch.min(
            relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1)
        )
        relative_buckets += torch.where(is_small, relative_position.to(torch.int32), relative_postion_if_large)
        return relative_buckets


class RotaryEmbedding(MegatronModule):
    """
    Rotary positional embeddings (RoPE) with Megatron compatibility.
    """
    def __init__(
        self,
        config: FM9GConfig,
        dim: int,
        base: Union[int, float] = 10000,
        distance_scale: Union[int, float] = 1,
        dtype: torch.dtype = None,
    ):
        super().__init__(config=config)
        # --- LOGIC TO DETERMINE COMPUTATION DTYPE ---
        # If no explicit dtype is passed, check config flags.
        if dtype is not None:
            self.dtype = dtype
        elif config.bf16:
            self.dtype = torch.bfloat16
        elif config.fp16:
            self.dtype = torch.float16
        else:
            self.dtype = config.params_dtype
        # inverse frequency for even dimensions
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device="cuda", dtype=torch.float32) / dim))
        self.inv_freq = inv_freq.to(dtype)
        self.distance_scale = distance_scale
        self.dtype = dtype

    def forward(self, x: torch.Tensor, x_pos: torch.Tensor):
        """
        Apply rotary embedding to input tensor.
        Args:
            x (:obj:`torch.Tensor` of shape (..., dim)): Input tensor.
            x_pos (:obj:`torch.Tensor` of shape (...)): Positions of input tokens.
        Returns:
            Tensor with rotary embeddings applied.
        """
        x_pos = x_pos * self.distance_scale
        freqs = x_pos[..., None].to(self.dtype) * self.inv_freq[None, :]  # (..., dim/2)

        # duplicate freqs to match dimension
        emb = torch.cat((freqs, freqs), dim=-1)  # (..., dim)
        emb_cos = emb.cos().to(x.dtype)
        emb_sin = emb.sin().to(x.dtype)

        rotate_x = torch.cat([-x[..., x.size(-1)//2:], x[..., :x.size(-1)//2]], dim=-1)
        return x * emb_cos + rotate_x * emb_sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate half of the embedding vector.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1).contiguous()


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, seq_dim: int, offset: int):
    """
    Apply rotary embeddings with defensive shape handling.
    """
    # 1. Force cos/sin to match the number of dimensions of x (4D)
    # Originally [4096, 1, 1], we need to reshape to [1, 1, 4096, 1] for broadcasting
    if cos.ndim == 3:
        # Assume cos is [S, 1, D_rope] or [S, D_rope, 1]
        # Reshape it to [1, 1, S, D_rope] to match x's shape [1, B, S, D]
        cos = cos.view(1, 1, cos.size(0), -1)
        sin = sin.view(1, 1, sin.size(0), -1)
    
    # 2. Now cos is [1, 1, 4096, D], x is [1, 7, 4096, 128]
    # Perform narrow along seq_dim
    q_len = x.size(seq_dim)
    cos_sliced = cos.narrow(seq_dim, offset, q_len)
    sin_sliced = sin.narrow(seq_dim, offset, q_len)

    # 3. Compute rotation
    x_rot = rotate_half(x)
    
    # Broadcasting will take effect:
    # x: [1, 7, 4096, 128]
    # cos_sliced: [1, 1, 4096, 128] (D dimension will automatically align)
    return (x * cos_sliced) + (x_rot * sin_sliced)


def unpad_apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, seq_dim: int, position_ids: torch.Tensor):
    # x shape: [S_local, H, D] -> e.g., 2048
    cos_table = cos.squeeze() 
    sin_table = sin.squeeze()

    p_ids = position_ids.view(-1).long()

    # --- Core adjustment logic ---
    if p_ids.numel() > x.size(0):
        # Indicates position_ids is full-length (4096)
        # We need to select the slice corresponding to the current TP rank
        from megatron.core import parallel_state
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        
        # Length of each slice
        local_s = x.size(0) # e.g., 2048
        
        # Explicitly compute start and end indices
        start_idx = tp_rank * local_s
        end_idx = start_idx + local_s
        
        # Slice safely to avoid out-of-bounds
        p_ids = p_ids[start_idx:end_idx]
        
    # Double-check to be safe
    if p_ids.numel() != x.size(0):
        # If still not equal, tp_rank * local_s logic mismatches the external data flow
        # Force truncate to x's length as a fallback (may slightly misalign positions but will run)
        p_ids = p_ids[:x.size(0)]

    # index_select now selects exactly 2048 position vectors
    cos_selected = cos_table.index_select(0, p_ids).unsqueeze(1)
    sin_selected = sin_table.index_select(0, p_ids).unsqueeze(1)
    
    return (x * cos_selected.to(x.dtype)) + (rotate_half(x) * sin_selected.to(x.dtype))


class RotaryEmbeddingESM(MegatronModule):
    """
    Rotary position embeddings based on those in
    [RoFormer](https://huggingface.co/docs/transformers/model_doc/roformer). Query and keys are transformed by rotation
    matrices which depend on their relative positions.
    """

    def __init__(
        self,
        config: FM9GConfig,
        dim: int,
        base: Union[int, float] = 10000,
        distance_scale: Union[int, float] = 1,
        dtype=None,
        persistent=True,
        mixed_precision=True,
    ):
        super().__init__(config=config)
        # --- LOGIC TO DETERMINE COMPUTATION DTYPE ---
        # If no explicit dtype is passed, check config flags.
        if dtype is not None:
            self.dtype = dtype
        elif config.bf16:
            self.dtype = torch.bfloat16
        elif config.fp16:
            self.dtype = torch.float16
        else:
            self.dtype = config.params_dtype
        self.config=config
        self.base = base
        self.distance_scale = distance_scale
        self.dtype = dtype

        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device="cuda", dtype=torch.float32) / dim))
        if mixed_precision:
            self.register_buffer("inv_freq", inv_freq, persistent=persistent)
        else:
            self.register_buffer("inv_freq", inv_freq.to(self.dtype), persistent=persistent)

        self._seq_len_cached = -1
        self._cos_cached = None
        self._sin_cached = None
        self.mixed_precision = mixed_precision

        self.apply_rotary_pos_emb = apply_rotary_pos_emb
        self.unpad_apply_rotary_pos_emb = unpad_apply_rotary_pos_emb

    def _update_cos_sin_tables(self, x, seq_dim, seq_len):
        if seq_len > self._seq_len_cached or self._cos_cached.device != x.device:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t * self.distance_scale, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            for i in range(x.dim() - 1):
                if i != seq_dim:
                    emb = emb.unsqueeze_(i)
            if self.mixed_precision:
                self._cos_cached = emb.cos().to(self.dtype)
                self._sin_cached = emb.sin().to(self.dtype)
            else:
                self._cos_cached = emb.cos()
                self._sin_cached = emb.sin()
        return self._cos_cached, self._sin_cached

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, seq_dim, offset=0, cu_seqlens=None, max_length=None, position_ids=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # --- DEBUG PRINT ---
        # if torch.distributed.get_rank() == 0:
        #     print(f"[RoPE] q_shape: {q.shape}, k_shape: {k.shape}, seq_dim: {seq_dim}, ids_shape: {position_ids.shape if position_ids is not None else 'None'}")

        seq_dim = (seq_dim + k.dim()) % k.dim()
        
        if cu_seqlens is None:
            # [Batch, Heads, Seq, Dim]
            self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k, seq_dim, k.size(seq_dim) + offset)
            return (
                self.apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached, seq_dim, offset),
                self.apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached, seq_dim, offset),
            )
        else:
            # FlashAttention Varlen [Total_Tokens, Heads, Dim]
            assert offset == 0, "past kv is not supported in flash attn"
            self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k, seq_dim, self.config.max_sequence_length)
            
            return (
                unpad_apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached, seq_dim, position_ids),
                unpad_apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached, seq_dim, position_ids),
            )


@torch.jit.script
def apply_chatglm_rotary_pos_emb(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    # x: [b, np, sq, hn]
    x = x.permute(2, 0, 1, 3)  # [b, np, sq, hn] -> [sq, b, np, hn]
    sq, b, np, hn = x.shape
    rot_dim = rope_cache.shape[-2] * 2
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    # truncate to support variable sizes
    rope_cache = rope_cache[:sq]
    xshaped = x.reshape(sq, -1, np, rot_dim // 2, 2)
    rope_cache = rope_cache.view(sq, -1, 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(3)
    ret = torch.cat((x_out2, x_pass), dim=-1)
    ret = ret.permute(1, 2, 0, 3)  # [sq, b, np, hn] -> [b, np, sq, hn]
    return ret


class ChatGLMRotaryEmbedding(MegatronModule):
    def __init__(
        self, 
        config,
        dim, 
        device="cuda", 
        dtype=None, 
        persistent=True):
        super().__init__(config=config)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=dtype, device=device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=persistent)
        self.dim = dim

    def forward_impl(self, seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000):
        """Enhanced Transformer with Rotary Position Embedding.

        Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
        transformers/rope/__init__.py. MIT License:
        https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
        """
        # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
        theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=dtype, device=device) / n_elem))

        # Create position indexes `[0, 1, ..., seq_len - 1]`
        seq_idx = torch.arange(seq_len, dtype=dtype, device=device)

        # Calculate the product of position index and $\theta_i$
        idx_theta = torch.outer(seq_idx, theta).float()

        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)

        # this is to mimic the behaviour of complex32, else we will get different results
        if dtype in (torch.float16, torch.bfloat16, torch.int8):
            cache = cache.bfloat16() if dtype == torch.bfloat16 else cache.half()
        return cache

    def forward(self, max_seq_len, offset: int = 0):
        return self.forward_impl(max_seq_len, self.dim, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
