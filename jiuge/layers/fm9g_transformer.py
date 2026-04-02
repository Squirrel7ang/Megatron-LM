import torch
from torch import Tensor
from torch.nn import ModuleList
from typing import Optional, List, Tuple

from megatron.core import parallel_state
from megatron.core.transformer.module import MegatronModule
from megatron.core.tensor_parallel.random import checkpoint
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

# Import our refactored components
from jiuge.layers.fm9g_blocks import TransformerLayer
from jiuge.layers.fm9g_layernorm import LayerNorm
from jiuge.config.fm9g_config import FM9GConfig

import torch.distributed as dist

class TransformerBlock(MegatronModule):
    """
    Megatron-style Transformer TransformerBlock (Decoder backbone) with Virtual Pipeline (VP) support.
    """

    def __init__(
        self,
        config: FM9GConfig,
        pre_process: bool = True,
        post_process: bool = True,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
    ):
        super().__init__(config=config)
        self.pg_collection = pg_collection
        self.vp_stage = vp_stage
        self.pre_process = pre_process
        self.post_process = post_process

        # 1. Layer Allocation Logic
        pp_world_size = parallel_state.get_pipeline_model_parallel_world_size()
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        vpp_size = self.config.virtual_pipeline_model_parallel_size
        

        # Using Megatron core utility to handle offset calculation
        # This is safe for both standard PP and Interleaved PP (VPP)
        self.offset = get_transformer_layer_offset(self.config, self.vp_stage, pp_rank)

        # Calculate layers for this specific chunk
        if vpp_size is not None and vpp_size > 1:
            assert self.config.num_layers % (pp_world_size * vpp_size) == 0, \
                f"Total layers ({self.config.num_layers}) must be divisible by PP*VP"
            self.num_layers_per_chunk = self.config.num_layers // (pp_world_size * vpp_size)
        else:
            self.num_layers_per_chunk = self.config.num_layers // pp_world_size

        # 2. Layer Initialization
        self.layers = ModuleList()
        for i in range(self.num_layers_per_chunk):
            # global_layer_number is useful for logging or specialized init
            # global_layer_number = self.offset + i + 1
            
            layer = TransformerLayer(
                config=self.config,
                pg_collection=self.pg_collection,
            )
            self.layers.append(layer)

        # 3. Final LayerNorm (only for the last chunk of the last PP stage)
        if self.post_process:
            self.output_layernorm = LayerNorm(
                config=self.config,
                hidden_size=self.config.hidden_size, 
            )
        else:
            self.output_layernorm = None

        # Placeholder for p2p communication
        self.input_tensor = None
        
    def set_input_tensor(self, input_tensor: Tensor) -> None:
        """
        Required by Megatron-LM core scheduler. 
        Sets the input tensor received from the previous pipeline stage.
        """
        self.input_tensor = input_tensor

    def get_input_tensor(self) -> Optional[Tensor]:
        return self.input_tensor

    def _checkpointed_forward(self, layer, *args):
        """
        Helper for gradient checkpointing (recomputation).
        """
        def custom_forward(*inputs):
            return layer(*inputs)
        
        return checkpoint(custom_forward, None, *args)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_bias: Optional[Tensor] = None,
        length_mask: Optional[Tensor] = None,
        attention_mask_bias: Optional[Tensor] = None,
        cu_seqlens: Optional[Tensor] = None,
        max_seqlen: Optional[int] = None,
        position_ids: Optional[Tensor] = None,
    ) -> Tensor:

        # print(f"DEBUG [Encoder In]: grad_fn={hidden_states.grad_fn}")
        
        # 1. P2P Input Handling
        # If this is not the first stage of the whole pipeline, 
        # we must use the tensor provided by set_input_tensor.
        if not self.pre_process:
            if self.input_tensor is not None:
                hidden_states = self.input_tensor
                # Clear to avoid memory leaks or using stale data in next microbatch
                self.input_tensor = None 

        # 2. Execution Loop
        for layer in self.layers:
            if self.config.recompute_granularity == 'full' and self.training:
                # Recompute everything for this layer
                hidden_states = self._checkpointed_forward(
                    layer,
                    hidden_states,
                    attention_mask,
                    position_bias,
                    None, # past_key_value
                    length_mask,
                    attention_mask_bias,
                    cu_seqlens,
                    max_seqlen,
                    position_ids,
                )
            else:
                # Standard forward
                hidden_states = layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_bias=position_bias,
                    past_key_value=None, 
                    length_mask=length_mask,
                    attention_mask_bias=attention_mask_bias,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    position_ids=position_ids,
                )

        # 3. Final LayerNorm
        if self.post_process and self.output_layernorm is not None:
            hidden_states = self.output_layernorm(hidden_states)

        return hidden_states