import math
from typing import Optional, Tuple

import torch

try:
    from jiuge.layers.fm9g_flash_triton import FlashAttnFunc
except:
    FlashAttnFunc = None

# Megatron tensor parallel layers
from megatron.core.transformer.module import MegatronModule
from megatron.core.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core import parallel_state

#  Retain Jiuge Custom Rotary
from jiuge.layers.fm9g_position_embedding import apply_chatglm_rotary_pos_emb
from jiuge.config.fm9g_config import FM9GConfig

# Official FlashAttention CUDA Interface
try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None


"""
# Legacy implementation from Jiuge training framework.
# Uses deprecated FlashAttention v1 custom CUDA kernels.
# Replaced by FlashAttention v2: flash_attn_varlen_func
# This block is kept only for historical reference.

class OpFlash(torch.autograd.Function):
    @staticmethod
    def forward(ctx, self, record, q, k, v, cu_seqlens, max_seqlen, dropout_p, causal):
        ctx.self = self
        ctx.cu_seqlens = cu_seqlens
        ctx.max_length = max_seqlen
        ctx.dropout_p = dropout_p
        ctx.causal = causal
        ctx.softmax_scale = q.shape[-1] ** (-0.5)
        if not record and "out" in self._layer_dict:
            out = self._layer_dict.pop("out")
            softmax_lse = self._layer_dict.pop("softmax_lse")
            rng_state = self._layer_dict.pop("rng_state")
        else:
            out, _, _, _, _, softmax_lse, _, rng_state = _flash_attn_varlen_forward(
                q,
                k,
                v,
                cu_seqlens,
                cu_seqlens,
                max_seqlen,
                max_seqlen,
                dropout_p,
                ctx.softmax_scale,
                causal=causal,
                window_size=(-1, -1),
                alibi_slopes=None,
                return_softmax=False,
            )
            if record:
                self._layer_dict["out"] = out
                self._layer_dict["softmax_lse"] = softmax_lse
                self._layer_dict["rng_state"] = rng_state

        ctx.save_for_backward(q, k, v, out, softmax_lse, rng_state)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, softmax_lse, rng_state = ctx.saved_tensors
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        _flash_attn_varlen_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            ctx.cu_seqlens,
            ctx.cu_seqlens,
            ctx.max_length,
            ctx.max_length,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            (-1,-1),
            None,
            False,
            rng_state=rng_state,
        )
        return None, None, dq, dk, dv, None, None, None, None
"""


class Attention(MegatronModule):
    """
    JIUGE Attention implementation refactored for Megatron-core.
    Aligned with TransformerConfig fields.
    """
    def __init__(
        self,
        config: FM9GConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ) -> None:
        super().__init__(config=config)
        
        self.pg_collection = pg_collection

        # Aligning with Megatron-core TransformerConfig fields
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        
        # In Megatron, num_query_groups is used for GQA (equivalent to num_kv_heads)
        self.num_query_groups = config.num_query_groups if config.num_query_groups is not None else self.num_attention_heads
        self.kv_channels = config.kv_channels
        
        # Calculate GQA groups
        self.head_groups = self.num_attention_heads // self.num_query_groups
        
        # Process Groups
        tp_group = self.pg_collection.tp if self.pg_collection else None

        self.pos_bias_type = config.pos_bias_type
        self.use_cache = config.use_cache 

        # 1. QKV Projection (Column Parallel)
        # Megatron uses (num_heads + 2 * num_query_groups) * kv_channels
        self.qkv_proj = ColumnParallelLinear(
            input_size=self.hidden_size,
            output_size=(self.num_attention_heads + 2 * self.num_query_groups) * self.kv_channels,
            config=config,
            init_method=config.init_method,
            bias=config.add_bias_linear,
            gather_output=False,
            tp_group=tp_group,
        )

        # 2. Output Projection (Row Parallel)
        self.out_proj = RowParallelLinear(
            input_size=self.num_attention_heads * self.kv_channels,
            output_size=self.hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=config.add_bias_linear,
            input_is_parallel=True,
            tp_group=tp_group,
            skip_bias_add=False,
        )

        # 3. Dropout and Softmax
        self.softmax = torch.nn.Softmax(dim=-1)
        self.attention_dropout = torch.nn.Dropout(p=config.attention_dropout) if config.attention_dropout > 0 else None
        
        # Using official 'use_flash_attn' field from config
        self.use_flash_attn = getattr(config, 'use_flash_attn', False)


    def forward(
        self,
        hidden_q: torch.Tensor,
        hidden_kv: torch.Tensor,
        attention_mask: torch.BoolTensor,
        position_bias: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        length_mask: Optional[torch.Tensor] = None,
        attention_mask_bias: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: int = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        if self.use_flash_attn:
            assert self.pos_bias_type == "rotary", (
                f"FlashAttention only supports 'rotary' pos_bias_type."
            )

        # 1. Get input dimensions
        len_q, batch_size, _ = hidden_q.size()
        len_k = hidden_kv.size(0)

        # 2. QKV projection
        qkv_res = self.qkv_proj(hidden_q)
        
        if isinstance(qkv_res, tuple):
            qkv, qkv_bias = qkv_res
            if qkv_bias is not None:
                qkv = qkv + qkv_bias  
        else:
            qkv = qkv_res

        # Identify current sequence length (could be full sequence in SP mode)
        current_seq_len = qkv.size(0)

        # --- PRECISION ENFORCEMENT ---
        compute_dtype = self.config.params_dtype
        if self.config.bf16:
            compute_dtype = torch.bfloat16
        elif self.config.fp16:
            compute_dtype = torch.float16
        
        qkv = qkv.to(compute_dtype)

        # --- HEAD DIMENSION CALCULATION ---
        actual_last_dim = qkv.size(-1)
        local_total_heads = actual_last_dim // self.kv_channels
        
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        local_num_kv_heads = self.num_query_groups // tp_size
        local_num_heads = local_total_heads - 2 * local_num_kv_heads

        # Reshape to [Seq, Batch, Heads, Dim]
        qkv = qkv.view(current_seq_len, batch_size, local_total_heads, self.kv_channels)

        # 3. Split Q, K, and V
        h_q = qkv[:, :, : local_num_heads, :]
        h_k = qkv[:, :, local_num_heads : local_num_heads + local_num_kv_heads, :]
        h_v = qkv[:, :, local_num_heads + local_num_kv_heads :, :]

        # 4. Attention Core Logic
        if not self.use_flash_attn:
            # Standard Attention expects [Batch, Heads, Seq, Dim]
            h_q = h_q.permute(1, 2, 0, 3).contiguous()
            h_k = h_k.permute(1, 2, 0, 3).contiguous()
            h_v = h_v.permute(1, 2, 0, 3).contiguous()

            # --- Position Bias / Rotary ---
            if self.pos_bias_type == "rotary":
                h_q, h_k = position_bias(h_q, h_k, -2, offset=past_kv[0].size(-2) if past_kv is not None else 0)
            elif self.pos_bias_type == "chatglm_rotary":
                h_q = apply_chatglm_rotary_pos_emb(h_q, position_bias)
                h_k = apply_chatglm_rotary_pos_emb(h_k, position_bias)

            # --- KV Caching ---
            if past_kv is not None:
                h_k = torch.cat([past_kv[0], h_k], dim=-2)
                h_v = torch.cat([past_kv[1], h_v], dim=-2)
                len_k = h_k.size(-2)
            else:
                len_k = current_seq_len

            scale = math.sqrt(math.sqrt(self.kv_channels))
            h_q = h_q / scale
            h_k = h_k / scale

            # GQA Handling
            head_groups = local_num_heads // local_num_kv_heads
            if head_groups == 1:
                score = torch.matmul(h_q, h_k.transpose(-1, -2))
            else:
                score = torch.matmul(
                    h_q.reshape(batch_size, local_num_kv_heads, head_groups, current_seq_len, self.kv_channels),
                    h_k.unsqueeze(2).transpose(-1, -2),
                ).view(batch_size, local_num_heads, current_seq_len, len_k)

            if self.pos_bias_type == "relative":
                # 1. Handle incremental decoding (inference) case where current_seq_len == 1
                pb = position_bias[:, :, -1:, :] if (current_seq_len == 1 and len(position_bias.size()) == 4) else position_bias
                
                # 2. Check for Head Dimension mismatch (Global Heads vs Local Heads)
                if pb.size(1) != score.size(1):
                    # Get the number of heads assigned to each TP rank
                    heads_per_rank = score.size(1) 
                    # Identify which slice of heads this rank is responsible for
                    tp_rank = parallel_state.get_tensor_model_parallel_rank()
                    
                    start_idx = tp_rank * heads_per_rank
                    end_idx = start_idx + heads_per_rank
                    
                    # Slice pb from [Batch, Global Heads, Q, K] to [Batch, Local Heads, Q, K]
                    pb = pb[:, start_idx:end_idx, :, :]

                # 3. Now the shapes are aligned: [B, Local Heads, Q, K] + [B, Local Heads, Q, K]
                score = score + pb

            mask_val = attention_mask.view(batch_size, 1, current_seq_len, len_k)
            score = torch.masked_fill(score, mask_val == False, float("-inf"))
            score = self.softmax(score)
            score = torch.masked_fill(score, mask_val == False, 0)

            if self.attention_dropout is not None:
                score = self.attention_dropout(score)

            score = torch.matmul(
                score.view(batch_size, local_num_kv_heads, head_groups, current_seq_len, len_k), 
                h_v.unsqueeze(2)
            ).view(batch_size, local_num_heads, current_seq_len, self.kv_channels)
            
            score = score.permute(2, 0, 1, 3).reshape(current_seq_len, batch_size, -1)

        else:
            # 7. FlashAttention branch
            h_q, h_k, h_v = h_q.to(compute_dtype), h_k.to(compute_dtype), h_v.to(compute_dtype)
            
            if attention_mask_bias is not None:
                score = FlashAttnFunc.apply(h_q, h_k, h_v, attention_mask_bias, False, None)
            else:
                if self.pos_bias_type == "chatglm_rotary":
                    raise NotImplementedError("FlashAttn not supported for ChatGLM rotary")
                
                # Reshape to [Total_Tokens, Heads, Dim]
                h_q = h_q.reshape(-1, local_num_heads, self.kv_channels)
                h_k = h_k.reshape(-1, local_num_kv_heads, self.kv_channels)
                h_v = h_v.reshape(-1, local_num_kv_heads, self.kv_channels)
                
                # Apply RoPE
                h_q, h_k = position_bias(h_q, h_k, seq_dim=0, cu_seqlens=cu_seqlens, max_length=max_seqlen, position_ids=position_ids)
                
                # Execute FlashAttention
                score = flash_attn_varlen_func(
                    h_q, h_k, h_v,
                    cu_seqlens, cu_seqlens,
                    max_seqlen, max_seqlen,
                    getattr(self.config, 'attention_dropout', 0.0),
                    causal=True,
                    deterministic=True,
                )
                
                # Reshape back to [Seq, Batch, Local_Hidden]
                score = score.view(current_seq_len, batch_size, local_num_heads * self.kv_channels)

        # 8. Output projection (RowParallelLinear handles Reduce-Scatter automatically if sequence_parallel=True)
        out_res = self.out_proj(score)

        if isinstance(out_res, tuple):
            out, bias = out_res
            if bias is not None:
                out = out + bias
        else:
            out = out_res

        if self.use_cache:
            return out, (h_k, h_v)
        else:
            return out