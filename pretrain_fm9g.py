"""Pretrain and SFT fm9g."""

import sys
from typing import Optional

import torch

from functools import partial
from jiuge.utils.arguments import JIUGEArgument
from megatron.core import parallel_state
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.training import inprocess_restart
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig, MockGPTDataset
from megatron.core.enums import ModelType
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.utils import get_attr_wrapped_model, StragglerDetector
from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
from megatron.core.tensor_parallel import vocab_parallel_cross_entropy
from megatron.training import get_args, get_timers, pretrain, print_rank_0
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    get_blend_and_blend_per_split,
    is_first_or_last_pipeline_stage,
)
from megatron.training.datasets.sft_dataset import SFTDataset
from model_provider import model_provider
from jiuge.fm9g_builder import fm9g_builder

try:
    from megatron.post_training.arguments import add_modelopt_args
    from megatron.post_training.loss_func import loss_func as loss_func_modelopt

    has_nvidia_modelopt = True
except ImportError:
    has_nvidia_modelopt = False


# Detect if any node is slowing down the training process
stimer = StragglerDetector()


def get_batch(data_iterator, vp_stage=None):
    """Generate a batch with explicit JIUGE compatibility."""
    # 1. Only the first and last stages of the pipeline need to handle data (PP logic)
    if not is_first_or_last_pipeline_stage(vp_stage):
        return None, None, None, None, None

    # 2. Get data for the current parallel group.
    # This function usually handles broadcasting to ensure data consistency within the TP group.
    batch = get_batch_on_this_tp_rank(data_iterator)

    # 3. Perform sequence splitting if Context Parallel (CP) is enabled
    batch = get_batch_on_this_cp_rank(batch)

    # 4. Explicitly extract tensors to match the forward interface
    tokens = batch.get('tokens')
    labels = batch.get('labels')
    loss_mask = batch.get('loss_mask')
    attention_mask = batch.get('attention_mask')
    position_ids = batch.get('position_ids')

    return tokens, labels, loss_mask, attention_mask, position_ids


# Define spiky loss threshold (typically 10x the max observed loss)
SPIKY_LOSS_FACTOR = 10


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    """
    Matches official Megatron loss_func. 
    output_tensor is now the [S, B] losses from the model.
    """
    # 1. Align the model output losses with the loss_mask
    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.view(-1).float()

    # 2. Compute the weighted sum
    total_loss = torch.sum(losses * loss_mask)
    num_tokens = loss_mask.sum()

    # 3. Compute the averaged loss
    if num_tokens > 0:
        averaged_loss = total_loss / num_tokens
    else:
        averaged_loss = total_loss

    # 4. Reporting (keep the original Megatron reporting logic)
    reporting_loss = torch.cat([
        total_loss.clone().detach().view(1), 
        num_tokens.clone().detach().view(1).to(torch.float32)
    ])

    return averaged_loss, {'lm loss': reporting_loss}


def forward_step(data_iterator, model):
    """
    Forward training step for Dense FM9G model.
    Optimized for Sequence-first (S, B) internal logic to avoid gradient shape mismatch.
    """
    args = get_args()
    timers = get_timers()

    # 1. Get the batch (B, S)
    timers('batch-generator', log_level=2).start()
    vp_stage = get_attr_wrapped_model(model, "vp_stage")
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator, vp_stage)
    timers('batch-generator').stop()

    # 2. Extract metadata and Determine Shapes
    if tokens is not None:
        batch_size, seq_length = tokens.shape
    else:
        actual_model = model[0] if isinstance(model, list) else model
        input_tensor = getattr(actual_model, 'input_tensor', None)
        if input_tensor is not None:
            # Megatron P2P buffer is usually [S, B, H]
            seq_length = input_tensor.shape[0]
            batch_size = input_tensor.shape[1]
        else:
            batch_size = args.micro_batch_size
            seq_length = args.seq_length

    device = torch.cuda.current_device()

    # 3. Handle position_ids & Transpose to [S, B]
    # We generate [B, S] and then transpose to [S, B] to match model internal expectation
    if position_ids is None:
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=device
        ).unsqueeze(0).expand(batch_size, -1)
    
    # --- CRITICAL DIMENSION ALIGNMENT START ---
    # Transpose all relevant tensors to Sequence-first [S, B]
    # This ensures that gradients generated during backward will be [S, B, H]
    if tokens is not None:
        tokens = tokens.transpose(0, 1).contiguous()         # [B, S] -> [S, B]
    if labels is not None:
        labels = labels.transpose(0, 1).contiguous()         # [B, S] -> [S, B]
    if loss_mask is not None:
        loss_mask = loss_mask.transpose(0, 1).contiguous()   # [B, S] -> [S, B]
    if position_ids is not None:
        position_ids = position_ids.transpose(0, 1).contiguous() # [B, S] -> [S, B]
    # --- CRITICAL DIMENSION ALIGNMENT END ---

    # 4. Handle FlashAttention metadata
    cu_seqlens = torch.arange(
        0, (batch_size + 1) * seq_length, step=seq_length, 
        dtype=torch.int32, device=device
    )
    max_seqlen = seq_length

    # 5. Model Forward
    # Now model receives inputs in [S, B] layout
    output_tensor = model(
        input_ids=tokens,
        position_ids=position_ids,
        attention_mask=attention_mask,
        labels=labels, 
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen
    )

    # 6. Return the loss tensor and the partial loss function.
    # We pass the transposed loss_mask [S, B] to match output_tensor [S, B]
    return output_tensor, partial(loss_func, loss_mask)


def is_dataset_built_on_rank(vp_stage=None):
    """Check if the dataset should be built on the current rank to save memory/IO."""
    return is_first_or_last_pipeline_stage(vp_stage) and parallel_state.get_tensor_model_parallel_rank() == 0


def core_gpt_dataset_config_from_args(args):
    """
    Build GPTDatasetConfig using Megatron standard build_tokenizer.
    Handles SentencePiece via --tokenizer-type and --tokenizer-model.
    """
    tokenizer = build_tokenizer(args)

    # Patch: Ensure the tokenizer has an .eod attribute for dataset logic
    if not hasattr(tokenizer, 'eod'):
        tokenizer.eod = tokenizer.eos_id

    blend, blend_per_split = get_blend_and_blend_per_split(args)

    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=blend,
        blend_per_split=blend_per_split,
        split=args.split,
        multiple_validation_sets=args.multiple_validation_sets,
        full_validation=args.full_validation,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
        object_storage_cache_path=args.object_storage_cache_path,
        mid_level_dataset_surplus=args.mid_level_dataset_surplus,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    """
    Build the train, validation, and test datasets for FM9G.

    Args:
        train_val_test_num_samples: List of samples needed for [train, val, test].
        vp_stage: Virtual pipeline stage identifier.
    """
    args = get_args()

    config = core_gpt_dataset_config_from_args(args)

    # Select dataset class based on training mode
    if args.sft:
        dataset_type = SFTDataset
    else:
        if args.mock_data:
            dataset_type = MockGPTDataset
        else:
            dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for FM9G ...")

    # Use the standard builder for dataset construction
    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type, 
        train_val_test_num_samples, 
        partial(is_dataset_built_on_rank, vp_stage=vp_stage), 
        config
    ).build()

    print_rank_0("> finished creating FM9G datasets ...")

    return train_ds, valid_ds, test_ds


def model_provider(pre_process: bool, post_process: bool, vp_stage: int = None, config=None, pg_collection=None):
    """
    Standard Megatron model provider. 
    In VP mode, Megatron calls this multiple times, once for each chunk.
    """
    model = fm9g_builder(
        pre_process=pre_process, 
        post_process=post_process, 
        vp_stage=vp_stage,
        config=config,
        # pg_collection=pg_collection
        pg_collection=None
    )

    return model


def extra_args_provider(parser):
    """
    Combine standard Megatron args with JIUGE specific CLI arguments.
    """
    # 1. Add ModelOpt args for optimization features if available
    if has_nvidia_modelopt:
        parser = add_modelopt_args(parser)
    
    # 2. Add JIUGE FM9G specific arguments (e.g., pos-bias-type)
    parser = JIUGEArgument.add_jiuge_args(parser)
    
    return parser


if __name__ == "__main__":
    # 1. Enable distributed dataset flag for core datasets transition
    train_valid_test_datasets_provider.is_distributed = True

    # If user only wants to see CLI help, avoid inprocess_restart wrapper
    if "--help" in sys.argv or "-h" in sys.argv:
        pretrain(
            train_valid_test_datasets_provider,
            model_provider,
            ModelType.encoder_or_decoder,
            forward_step,
            extra_args_provider=extra_args_provider,
            args_defaults={'tokenizer_type': 'SentencePieceTokenizer'},
        )
    else:
        # Official Megatron training path with restart support
        pretrain_func, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

        pretrain_func(
            train_valid_test_datasets_provider,
            model_provider,
            ModelType.encoder_or_decoder,
            forward_step,
            extra_args_provider=extra_args_provider,
            args_defaults={'tokenizer_type': 'SentencePieceTokenizer'},
            store=store,
        )