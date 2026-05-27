#!/usr/bin/env python3
"""Test script for ArcTopkState's start_grad_sync and finish_grad_sync methods."""

import torch
import torch.distributed as dist

# Import using absolute imports
from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBucket
import megatron.core.distributed.grad_compression
from megatron.core.distributed.grad_compression import get_arc_topk_state, init_arc_topk_state

rand_gen = []
rand_gen.append(torch.Generator(device=f'cuda:0'))
rand_gen[0].manual_seed(0)
rand_gen.append(torch.Generator(device=f'cuda:1'))
rand_gen[1].manual_seed(1)


def main():
    # Initialize distributed environment
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    assert world_size == 2, "This test requires exactly 2 ranks"
    
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    
    global rand_gen
    local_rand_gen = rand_gen[rank]
    
    # Create test parameters
    params1 = [torch.nn.Parameter(torch.randn(4, 2, device=device, generator=local_rand_gen))]
    params2 = [torch.nn.Parameter(torch.randn(4, 4, device=device, generator=local_rand_gen))]
    
    # Create grad_data tensors
    # Set rows 0, 2, 4, 6, 8 to large values (8-12), others to small values (2-4)
    def create_grad_data(rows, cols):
        grad = torch.empty(rows, cols, device=device, dtype=torch.float32)
        # Set even rows (0, 2, 4, 6, 8) to large values (8-12)
        for i in range(0, rows, 2):
            grad[i, :] = torch.randint(8, 13, (cols,), device=device, generator=local_rand_gen).float()
        # Set odd rows to small values (2-4)
        for i in range(1, rows, 2):
            grad[i, :] = torch.randint(2, 5, (cols,), device=device, generator=local_rand_gen).float()
        return grad
    
    grad_data1 = create_grad_data(4, 2)
    print(f"[Rank {rank}] grad_data1: {grad_data1}")
    grad_data2 = create_grad_data(4, 4)
    print(f"[Rank {rank}] grad_data2: {grad_data2}")
    
    # Create param_data (can be None for test)
    param_data1 = None
    param_data2 = None
    
    # Create param_index_map
    param_index_map1 = {params1[0]: (0, grad_data1.numel(), 0)}
    param_index_map2 = {params2[0]: (0, grad_data2.numel(), 1)}
    
    # Create buckets
    bucket1 = _ParamAndGradBucket(
        params=params1,
        param_data=param_data1,
        grad_data=grad_data1,
        offset=0,
        numel_unpadded=grad_data1.numel(),
        gradient_scaling_factor=1.0,
        bucket_id=0,
        param_index_map=param_index_map1,
        params_with_extra_main_grads=[],
    )
    
    bucket2 = _ParamAndGradBucket(
        params=params2,
        param_data=param_data2,
        grad_data=grad_data2,
        offset=0,
        numel_unpadded=grad_data2.numel(),
        gradient_scaling_factor=1.0,
        bucket_id=1,
        param_index_map=param_index_map2,
        params_with_extra_main_grads=[],
    )
    
    # Save original gradients for verification
    original_grad1 = grad_data1.clone()
    original_grad2 = grad_data2.clone()
    
    # Initialize ArcTopkState
    arc_topk_state = get_arc_topk_state()
    
    # Create CUDA stream
    stream = torch.cuda.Stream(device=device)
    stream_context = torch.cuda.stream(stream)
    
    # Test start_grad_sync with single bucket each time
    print(f"[Rank {rank}] Starting first bucket sync...")
    arc_topk_state.start_grad_sync([bucket1], stream_context, async_op=False)
    print(f"[Rank {rank}] Starting second bucket sync...")
    arc_topk_state.start_grad_sync([bucket2], stream_context, async_op=False)

    print(f"[Rank {rank}] Starting finish_grad_sync...")
    arc_topk_state.finish_grad_sync()
    
    # Verify results
    torch.cuda.synchronize()
    
    # Print results
    print(f"[Rank {rank}] Bucket 1 grad_data after sync: \n {grad_data1}")
    print(f"[Rank {rank}] Bucket 2 grad_data after sync: \n {grad_data2}")
    


if __name__ == '__main__':
    dist.init_process_group(backend='nccl')
    init_arc_topk_state(
        process_group=dist.group.WORLD,
        priority_rank=4,
        compression_ratio=0.5,
        use_error_feedback=True,
    )
    main()
    main()
    dist.destroy_process_group()