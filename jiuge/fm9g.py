import torch
from torch import Tensor
from typing import Optional, List, Tuple, Union

from megatron.core.enums import ModelType
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core import tensor_parallel
from megatron.core import parallel_state
from megatron.training import get_args

from jiuge.layers.fm9g_embeddings import FM9GEmbedding
from jiuge.layers.fm9g_transformer import TransformerBlock
from jiuge.config.fm9g_config import FM9GConfig
from jiuge.layers.fm9g_position_embedding import BucketPositionBias, RotaryEmbeddingESM, ChatGLMRotaryEmbedding

class FM9GModel(LanguageModule):
    """
    Top-level wrapper for the FM9G model.
    Handles standard and interleaved pipeline parallelism (Virtual Pipeline).
    Even though the backbone is named 'Encoder', it functions as an autoregressive Decoder.
    """
    def __init__(
        self,
        config: FM9GConfig,
        vocab_size: int,
        pre_process: bool = True,
        post_process: bool = True,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
    ) -> None:
        super().__init__(config=config, pg_collection=pg_collection)

        self.pre_process = pre_process
        self.post_process = post_process
        self.parallel_output = parallel_output
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.vp_stage = vp_stage
        self.model_type = ModelType.encoder_or_decoder
        self.config = config
        # print(f"DEBUG: share_embeddings_and_output_weights={share_embeddings_and_output_weights}")

        # 1. Position Bias / Embedding Initialization
        # Note: In VPP mode, every chunk typically holds its own RoPE object as it's stateless.
        # For learnable relative bias, further synchronization might be needed.
        if config.pos_bias_type == "relative":
            self.position_bias_module = BucketPositionBias(
                config=config,
                num_heads=config.num_attention_heads,
                pg_collection=pg_collection
            )
        elif config.pos_bias_type == "rotary":
            self.position_bias_module = RotaryEmbeddingESM(
                config=config,
                dim=config.kv_channels,
            )
        elif config.pos_bias_type == "chatglm_rotary":
            self.position_bias_module = ChatGLMRotaryEmbedding(
                config=config,
                dim=config.kv_channels
            )
        else:
            self.position_bias_module = None

        # 2. Embedding Layer (First Rank, First Chunk only)
        if self.pre_process:
            self.embedding = FM9GEmbedding(
                vocab_size=vocab_size,
                config=config,
            )

        # 3. Transformer Backbone (Specific layers for this chunk)
        self.decoder = TransformerBlock(
            config=config,
            pre_process=self.pre_process,
            post_process=self.post_process,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
        )

        # 4. Output Layer (Last Rank, Last Chunk only)
        if self.post_process:
            self.output_layer = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                vocab_size,
                config=config,
                init_method=config.init_method,
                bias=False,
                gather_output=not self.parallel_output,
                # Skip allocation if we are tying weights and this is the embedding stage
                skip_weight_param_allocation=(
                    self.share_embeddings_and_output_weights and self.pre_process
                ),
                tp_group=self.pg_collection.tp
            )

        # 5. Weight Tying Setup
        if self.pre_process or self.post_process:
            self.setup_embeddings_and_output_layer()

    def set_input_tensor(self, input_tensor: Tensor) -> None:
        """Required by Megatron P2P communication."""
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        self.decoder.set_input_tensor(input_tensor[0])

    def shared_embedding_or_output_weight(self) -> Optional[Tensor]:
        """Helper to retrieve the weight for tying across PP stages."""
        if not self.share_embeddings_and_output_weights:
            return None
        
        # Check for embedding on first stage/chunk
        if self.pre_process and hasattr(self, 'embedding'):
            return self.embedding.word_embeddings.weight
        
        # Check for output layer on last stage/chunk
        if self.post_process and hasattr(self, 'output_layer'):
            return self.output_layer.weight
            
        return None

    # def setup_embeddings_and_output_layer(self):
    #     """
    #     Synchronizes weights between the first and last stages of the pipeline.
    #     Supports both Pipeline Parallel and single-stage weight tying.
    #     """
    #     if not self.share_embeddings_and_output_weights:
    #         return

    #     # 1. Ensure we obtain the real embedding weight tensor
    #     # When PP=1, this module owns both pre_process and post_process
    #     if self.pre_process and hasattr(self, 'embedding'):
    #         shared_weight = self.embedding.word_embeddings.weight
    #     else:
    #         shared_weight = self.shared_embedding_or_output_weight()

    #     # In PP > 1 scenarios, if we are in the last stage, shared_weight might be 
    #     # uninitialized because of skip_weight_param_allocation. We must ensure 
    #     # it points to the output_layer's weight parameter object.
    #     if shared_weight is None and self.post_process and hasattr(self, 'output_layer'):
    #         # If ColumnParallelLinear skipped allocation, self.output_layer.weight is None.
    #         # We must manually create a parameter placeholder to hold the broadcasted weights.
    #         if self.output_layer.weight is None:
    #             args = get_args()
    #             tp_size = parallel_state.get_tensor_model_parallel_world_size()
                
    #             # ColumnParallelLinear weights are partitioned across the vocab dimension
    #             per_partition_vocab_size = (args.padded_vocab_size + tp_size - 1) // tp_size
                
    #             # Initialize an empty parameter on the current device
    #             self.output_layer.weight = torch.nn.Parameter(
    #                 torch.empty(
    #                     per_partition_vocab_size,
    #                     args.hidden_size,
    #                     device=torch.cuda.current_device(),
    #                     dtype=args.params_dtype
    #                 )
    #             )
    #         shared_weight = self.output_layer.weight

    #     if shared_weight is None:
    #         return

    #     # 2. Mark parameter attributes (required by Megatron DDP)
    #     shared_weight.is_embedding_or_output_parameter = True
    #     shared_weight.shared = True

    #     # 3. Handle the single-stage case (PP=1) with physical weight sharing
    #     # In this case, Embedding and Output Layer are on the same rank,
    #     # so they must point to the same underlying tensor in memory.
    #     if parallel_state.get_pipeline_model_parallel_world_size() == 1:
    #         if self.post_process and hasattr(self, 'output_layer'):
    #             # Critical: overwrite the reference to ensure both layers
    #             # use the exact same Tensor object. This guarantees that
    #             # the computation graph contains only one parameter and
    #             # DDP will build only one gradient bucket.
    #             self.output_layer.weight = shared_weight
    #         return  # Single-stage logic finished


    def _preprocess(
        self,
        input_ids: Tensor,
        decoder_input: Optional[Tensor] = None,
    ) -> Tensor:
        """Handles the transition from raw IDs/input tensors to hidden states."""
        if decoder_input is not None:
            return decoder_input
        
        if self.pre_process:
            return self.embedding(input_ids)
        
        # Middle stages retrieve from p2p buffer
        return self.decoder.get_input_tensor()

    def _postprocess(
        self,
        hidden_states: Tensor,
        labels: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Projects hidden states to logits.
        If labels are provided, computes and returns vocab-parallel cross entropy.
        """
        if not self.post_process:
            return hidden_states

        output_weight = self.shared_embedding_or_output_weight()

        # 1. Logits projection: result shape [S, B, V]
        logits, _ = self.output_layer(hidden_states, weight=output_weight)

        # 2. If labels are provided, compute the loss directly inside the model.
        # This avoids potential in-place modification errors and gradient
        # disconnections that may occur when computing the loss externally.
        if labels is not None:
            from megatron.core.tensor_parallel import vocab_parallel_cross_entropy

            if labels.dim() == 2 and labels.size(0) != logits.size(0):
                labels = labels.transpose(0, 1).contiguous()

            # Megatron's implementation expects logits to be in float32
            # The returned loss tensor has shape [S, B]
            losses = vocab_parallel_cross_entropy(logits.float(), labels)
            return losses

        # If labels are not provided (e.g., during inference),
        # return the raw logits
        return logits

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        decoder_input: Optional[Tensor] = None,
        length_mask: Optional[Tensor] = None,
        attention_mask_bias: Optional[Tensor] = None,
        cu_seqlens: Optional[Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> Tensor:
        
        # 1. Embedding / Input handling
        hidden_states = self._preprocess(
            input_ids=input_ids,
            decoder_input=decoder_input,
        )

        if attention_mask is None:
            # Get dimensions based on the GLOBAL sequence length
            # In SP, if hidden_states.size(0) is 1024 and TP=2, total_seq is 2048
            tp_size = parallel_state.get_tensor_model_parallel_world_size()
            local_seq_len = hidden_states.size(0)
            global_seq_len = local_seq_len * tp_size 
            batch_size = hidden_states.size(1)

            device = hidden_states.device
            # Create a FULL causal mask [global_seq_len, global_seq_len]
            full_seq_idx = torch.arange(global_seq_len, device=device)
            # [global_seq_len, global_seq_len]
            full_mask = (full_seq_idx.view(-1, 1) >= full_seq_idx.view(1, -1))
            
            # Final Shape: [B, 1, S_global, S_global] -> e.g., [1, 1, 2048, 2048]
            attention_mask = full_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)

        # 2. Position Bias handling
        current_pos_bias = None
        if self.position_bias_module is not None:
            if self.config.pos_bias_type == "relative":
                # Relative buckets logic (Global sequence length due to SP)
                current_pos_bias = self.position_bias_module(
                    query_pos=position_ids,
                    key_pos=position_ids,
                    rel_buckets=torch.zeros_like(attention_mask).long() 
                )
            elif self.config.pos_bias_type == "chatglm_rotary":
                # NEW: Get the actual Tensor from the module
                # current_seq_len should be global sequence length if SP is ON
                global_seq_len = attention_mask.size(-1) 
                current_pos_bias = self.position_bias_module(global_seq_len)
            else:
                # For standard RoPE or other types, pass the module/bias as is
                current_pos_bias = self.position_bias_module
        
        # 3. Layers execution
        hidden_states = self.decoder(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_bias=current_pos_bias,
            length_mask=length_mask,
            attention_mask_bias=attention_mask_bias,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            position_ids=position_ids,
        )

        # 4. Final Projection / Loss
        return self._postprocess(
            hidden_states=hidden_states,
            labels=labels
        )