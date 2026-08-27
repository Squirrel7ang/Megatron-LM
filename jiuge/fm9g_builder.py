from typing import List, Union, Optional

from megatron.training import get_args, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from jiuge.config.fm9g_config import FM9GConfig
from jiuge.fm9g import FM9GModel

def fm9g_builder(
    pre_process: bool, 
    post_process: bool, 
    vp_stage: int = None,
    config: FM9GConfig = None,
    pg_collection: Optional[ProcessGroupCollection] = None,
) -> Union[FM9GModel, List[FM9GModel]]:
    """
    Builder function for FM9G model.
    Converts Megatron args to FM9GConfig and instantiates the model.
    Handles both standard Pipeline Parallelism and Virtual Pipeline Parallelism.
    """
    args = get_args()
    print_rank_0('Building FM9G model ...')

    # 1. Generate Config
    if config is None:
        base_config = core_transformer_config_from_args(args)
        
        # Convert base config to dictionary and unpack into FM9GConfig
        config_dict = vars(base_config) if hasattr(base_config, '__dict__') else {}
        config_dict['num_query_groups'] = args.num_query_groups 
        config_dict['num_attention_heads'] = args.num_attention_heads
        config_dict['max_sequence_length'] = args.seq_length
        config = FM9GConfig(**config_dict)
        
        # Explicitly set JIUGE-specific flags
        config.use_flash_attn = getattr(args, 'use_flash_attn', False)
        config.num_query_groups = getattr(args, 'num_query_groups', None)
        raw_pos_type = getattr(args, 'pos_bias_type', None)
        if raw_pos_type is not None:
            normalized_pos_type = raw_pos_type.lower().replace("-", "_")
            config.pos_bias_type = normalized_pos_type
        else:
            config.pos_bias_type = None
        
    # 2. Retrieve parallel configurations
    vpp_size = getattr(args, 'virtual_pipeline_model_parallel_size', None)
    share_embeddings = not getattr(args, 'untie_embeddings_and_output_weights', False)
    # Prefer padded_vocab_size if available in args for Tensor Core efficiency
    vocab_size = getattr(args, 'padded_vocab_size', args.vocab_size)

    # 3. Instantiate FM9G Model(s)
    if vpp_size is not None and vpp_size > 1:
        if vp_stage is None:
            # Fallback handling: if VP is enabled but vp_stage is not provided, default to stage 0
            vp_stage = 0
            
        # Override input pre/post flags to ensure correct pipeline chunk behavior
        chunk_pre_process = pre_process and (vp_stage == 0)
        chunk_post_process = post_process and (vp_stage == vpp_size - 1)
    else:
        chunk_pre_process = pre_process
        chunk_post_process = post_process

    model = FM9GModel(
        config=config,
        vocab_size=vocab_size,
        pre_process=chunk_pre_process,
        post_process=chunk_post_process,
        parallel_output=True, 
        share_embeddings_and_output_weights=share_embeddings,
        vp_stage=vp_stage,
    )

    return model