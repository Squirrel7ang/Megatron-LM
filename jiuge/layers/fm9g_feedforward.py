from typing import Optional, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.core.transformer.module import MegatronModule
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)

from jiuge.config.fm9g_config import FM9GConfig

class DenseGatedACT(MegatronModule):
    """
    Gated linear unit (GLU) variant.
    Maps to standard SwiGLU/GeGLU structures.
    """
    def __init__(
        self,
        config: FM9GConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super().__init__(config=config)
        self.pg_collection = pg_collection
        
        # Standardize dimension names
        input_size = config.hidden_size
        ffn_hidden_size = config.ffn_hidden_size
        tp_group = self.pg_collection.tp if self.pg_collection else None

        # w_0 (Gate) and w_1 (Input)
        self.w_0 = ColumnParallelLinear(
            input_size=input_size,
            output_size=ffn_hidden_size,
            config=config,
            init_method=config.init_method,
            bias=config.add_bias_linear,
            gather_output=False,
            tp_group=tp_group,
        )
        self.w_1 = ColumnParallelLinear(
            input_size=input_size,
            output_size=ffn_hidden_size,
            config=config,
            init_method=config.init_method,
            bias=config.add_bias_linear,
            gather_output=False,
            tp_group=tp_group,
        )

        # activation_func in config is typically a callable (e.g., F.silu)
        self.activation_func = config.activation_func

    def forward(self, x: torch.Tensor):
        # --- ROBUST HANDLING FOR ColumnParallelLinear (WITH BIAS) ---
        def handle_linear_output(res):
            if isinstance(res, tuple):
                out, bias = res
                # If bias is not None, we MUST add it to the output 
                # to keep it in the computational graph.
                return out + bias if bias is not None else out
            return res

        # standard GLU: ACT(w0(x)) * w1(x)
        # Apply the helper to both w_0 and w_1
        gate_score = self.activation_func(handle_linear_output(self.w_0(x)))
        gate_input = handle_linear_output(self.w_1(x))
        
        return gate_score * gate_input


class FeedForward(MegatronModule):
    """
    JIUGE FeedForward module fully integrated with Megatron-core.
    """
    def __init__(
        self,
        config: FM9GConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super().__init__(config=config)
        self.pg_collection = pg_collection

        input_size = config.hidden_size
        ffn_hidden_size = config.ffn_hidden_size
        tp_group = self.pg_collection.tp if self.pg_collection else None

        # Input projection + Gated Activation
        self.w_in = DenseGatedACT(config=config, pg_collection=pg_collection)

        # Dropout: Using Megatron standard field
        dropout_rate = getattr(config, 'hidden_dropout', 0.0)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

        # Output projection
        self.w_out = RowParallelLinear(
            input_size=ffn_hidden_size,
            output_size=input_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=config.add_bias_linear,
            input_is_parallel=True,
            tp_group=tp_group,
            skip_bias_add=False,
        )

    def forward(self, x: torch.Tensor):
        # 1. Input projection + Gated Activation
        x = self.w_in(x)
        
        # 2. Dropout before output projection (following original flow)
        if self.dropout is not None:
            x = self.dropout(x)
            
        # 3. Output projection (RowParallelLinear)
        out_res = self.w_out(x)
        
        # --- FIX: Resolve potential (Tensor, Bias) tuple ---
        if isinstance(out_res, tuple):
            out, bias = out_res
            if bias is not None:
                # If training with bf16/fp16, bias addition is handled by PyTorch
                out = out + bias
        else:
            out = out_res

        return out