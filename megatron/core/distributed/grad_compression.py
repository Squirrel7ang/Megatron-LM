import math
import logging
from typing import List

import torch
import torch.distributed as dist
from torch.distributed import _coalescing_manager

from .param_and_grad_buffer import _ParamAndGradBucket

_ARC_TOPK_STATE = None
_GRAD_QUANTIZATION_STATE = None


logger = logging.getLogger(__name__)


def init_arc_topk_state(
        process_group,
        priority_rank,
        compression_ratio,
        use_error_feedback = False,
):
    global _ARC_TOPK_STATE
    if _ARC_TOPK_STATE is None:
        _ARC_TOPK_STATE = ArcTopkState(
            process_group,
            priority_rank,
            compression_ratio,
            use_error_feedback,
        )
    return _ARC_TOPK_STATE


def init_grad_quantization_state(
        process_group,
        use_error_feedback,
        dtype=torch.uint8,
        use_hadamard_transformation=False,
):
    global _GRAD_QUANTIZATION_STATE
    if _GRAD_QUANTIZATION_STATE is None:
        _GRAD_QUANTIZATION_STATE = GradQuantizationState(
            process_group=process_group,
            use_error_feedback=use_error_feedback,
            dtype=dtype,
            use_hadamard_transformation=use_hadamard_transformation,
        )
    return _GRAD_QUANTIZATION_STATE


def get_arc_topk_state():
    if _ARC_TOPK_STATE is None:
        raise ValueError("ARC-Top-K state is not initialized.")
    return _ARC_TOPK_STATE


def get_grad_quantization_state():
    if _GRAD_QUANTIZATION_STATE is None:
        raise ValueError("Gradient quantization state is not initialized.")
    return _GRAD_QUANTIZATION_STATE


def _is_integer(dtype: torch.dtype):
    return dtype in [
        torch.uint8,
        torch.uint16,
        torch.uint32,
        torch.uint64,
        torch.int8,
        torch.int16,
        torch.short,
        torch.int32,
        torch.int,
        torch.int64,
        torch.long,
        torch.quint8,
        torch.qint8,
        torch.qint32,
        torch.bool,
        torch.quint4x2,
        torch.quint2x4,
    ]


def _pack_for_bits4x2(x: torch.Tensor):
    x = x.view(-1)
    n = x.numel()
    len = (n+1) // 2

    y = torch.empty(len, dtype=x.dtype, device=x.device)
    limit = (n // 2) * 2

    x_pair = x[:limit]
    y[:limit] = (x_pair[0::2] & 0x0F) << 4 | (x_pair[1::2] & 0x0F)

    if n % 2 == 1:
        y[-1] = x[-1]
    return y


def _unpack_for_bits4x2(y: torch.Tensor, origin: torch.Tensor):
    n = origin.numel()
    x = torch.empty_like(origin)
    x[0::2] = (y[:(n//2)] & 0xF0) >> 4
    x[1::2] = (y[:(n//2)] & 0x0F)
    if n % 2 == 1:
        x[-1] = y[-1]
    return x


def _get_dtype_range(dtype):
    if dtype.is_floating_point:
        info = torch.finfo(dtype)
    elif _is_integer(dtype):
        info = torch.iinfo(dtype)
    elif dtype is torch.bits4x2:
        return -8, 7
    else:
        raise NotImplementedError(f"Unsupported dtype: {dtype=}")

    return info.min, info.max


def _quantize_per_tensor_backend(x, scale, zero_point, dtype):
    d_min, d_max = _get_dtype_range(dtype)
    if dtype.is_floating_point:
        y = x / scale + zero_point
        y = torch.clamp(y, d_min, d_max).to(dtype)
    elif _is_integer(dtype):
        y = torch.round(x / scale) + zero_point
        y = torch.clamp(y, d_min, d_max).to(dtype)
    elif dtype is torch.bits4x2:
        y = torch.round(x / scale) + zero_point
        y = torch.clamp(y, d_min, d_max).to(torch.uint8)
        y = _pack_for_bits4x2(y)
    else:
        raise NotImplementedError(f"Unsupported dtype: {dtype=}")
    return y


def _dequantize_per_tensor_backend(y, scale, zero_point, dtype, origin=None):
    if dtype is torch.bits4x2:
        x = _unpack_for_bits4x2(y, origin)
        x = scale * (x.to(torch.float32) - zero_point)
    else:
        x = scale * (y.to(torch.float32) - zero_point)
    return x


def _quantize_per_channel_backend(x, scale, zero_point, dtype):
    d_min, d_max = _get_dtype_range(dtype)
    y = torch.zeros(x.size(), device=x.device)
    for i in range(x.size()[0]):
        if dtype.is_floating_point:
            y[i, :] = x[i, :] / scale[i] + zero_point[i]
        else:
            y[i, :] = torch.round(x[i, :] / scale[i]) + zero_point[i]
    y = torch.clamp(y, d_min, d_max).to(dtype)
    return y


def _dequantize_per_channel_backend(y, scale, zero_point):
    y = y.to(torch.float32).to(y.device)
    x = torch.zeros_like(y, device=y.device)
    for i in range(x.size()[0]):
        x[i, :] = scale[i] * (y[i, :] - zero_point[i])
    return x


class ArcTopkState:
    def __init__(
        self,
        process_group,
        priority_rank,
        compression_ratio,
        use_error_feedback,
    ):
        logger.info(
            "ArcTopKState: priority_rank=%d, compression_ratio=%.4f; ",
            priority_rank,
            compression_ratio,
        )

        self.process_group = process_group
        self.priority_rank = priority_rank
        self.compression_ratio = compression_ratio
        self.use_error_feedback = use_error_feedback
        self.error_dict = {}
        self.all_futs = []


    def start_grad_sync_overlap(
        self,
        buckets: List[_ParamAndGradBucket],
        stream_context: torch.stream.StreamContext,
        async_op: bool = False
    ):
        r"""
        Implement ARC-Top-K algorithm.

        This arc_topk_preprocess function implements ARC-Top-K gradient compression
        algorithm described in the `paper <https://arxiv.org/abs/2510.26709>`_.
        Once gradient tensors are aggregated across all workers, this function applies
        compression as follows:
        1. calculate global priority

            1.1. let n * m = d, reshape grad of size d into n * m matrix G_i for node i.

            1.2. let V = torch.nrand(n, r) be an n * r matrix, where vec(V) ~ N(0, I_nr) and
            r stands for projection_rank.

            1.3. let priority matrix P_i of node i be matmul(G, V)/sqrt(r), which stands for
            the priority of each row of G in node i.

        2. calculate Indices

            2.1. perform an All-Reduce on P_i to get global priority matrix P = mean(P_i).

            2.2. let Indices be Top-K(diag(matmul(P, P.T))) where K stands for compression_ratio.

        3. gradient compression and All-Reduce

            3.1. let compressed gradient matrix G_i be G[Indices, :] and perform an All-Reduce on G'.
        """
        def _cal_max_factor(size: int):
            factor: int = 1
            while size % factor == 0 and size / factor > factor:
                factor *= 2
            return factor
        pass

        # indices_holder = {}
        infos = []
        # comm_futs = []

        world_size = self.process_group.size()
        process_group = self.process_group

        with stream_context, _coalescing_manager(self.process_group, async_ops=async_op) as cm:
            for _, bucket in enumerate(buckets):
                gradient = bucket.grad_data
                gradient = gradient.view(-1, row_num)
                d = gradient.numel()
                row_num = _cal_max_factor(d)

                V = torch.randn(row_num, self.priority_rank, device=gradient.device, dtype=gradient.dtype)
                P = torch.matmul(gradient, V) / math.sqrt(self.priority_rank)

                fut_p = dist.all_reduce(
                    P, group=process_group, async_op=async_op
                )

                info = {
                    "future": fut_p,
                    "d": d,
                    "row_num": row_num,
                    "gradient": gradient,
                }
                infos.append(info)


        for info in infos:
            fut_p = info["future"].get_future()

            def compress_and_sync(fut):
                synced_P = fut.value()[0]
                score = torch.diag(torch.matmul(synced_P, synced_P.T))

                col_num = info["d"] / info["row_num"]
                K = max(1, int(col_num * self.compression_ratio))
                indices = torch.topk(score, k=K).indices
                info['indices'] = indices

                comm_gradient = info["gradient"][indices, :]

                return dist.all_reduce(
                    comm_gradient, group=self.process_group, async_op=True
                ).get_future()

            def decompress_and_finalize(fut):
                gradient = info["gradient"]
                avg_gradient = fut.wait()[0] / world_size
                indices = info["indices"]
                gradient.zero_()
                gradient[indices, :] = avg_gradient

                return gradient

            self.all_futs.append(
                fut_p.then(compress_and_sync).then(decompress_and_finalize)
            )

    def finish_grad_sync(self):
        if self.all_futs:
            return
        for fut in self.all_futs:
            fut.wait()
        self.all_futs = []



class GradQuantizationState:
    __slots__ = [
        "process_group",
        "use_error_feedback",
        "error_dict",
        "dtype",
        "use_hadamard_transformation",
        "all_infos",
        "all_futs",
    ]

    def __init__(
        self,
        process_group,
        use_error_feedback,
        dtype,
        use_hadamard_transformation
    ):
        self.process_group = process_group
        self.use_error_feedback = use_error_feedback
        self.error_dict = {}
        self.dtype = dtype
        self.use_hadamard_transformation = use_hadamard_transformation
        self.all_infos = []
        self.all_futs = []

    def start_grad_sync(
            self,
            buckets: List[_ParamAndGradBucket],
            stream_context: torch.stream.StreamContext,
            async_op: bool = False
    ):
        def _get_all_gather_out_list(all_gather_in_list, world_size):
            out_list = [
                torch.zeros_like(
                    all_gather_in_list,
                    device=all_gather_in_list.device,
                    dtype=all_gather_in_list.dtype,
                )
                for _ in range(world_size)
            ]
            return out_list

        process_group = self.process_group
        rank = process_group.rank() if process_group is not None else dist.get_rank()
        # pyrefly: ignore [missing-attribute]
        world_size = process_group.size()
        original_dtype = buckets[0].grad_data.dtype if buckets else torch.float32

        with stream_context, _coalescing_manager(self.process_group, async_ops=async_op) as cm:
            for _, bucket in enumerate(buckets):
                gradient = bucket.grad_data
                bucket_index = bucket.bucket_id
                total_length = gradient.shape[0]
                if self.use_error_feedback:
                    if bucket_index in self.error_dict:
                        gradient.add_(self.error_dict[bucket_index])
                    else:
                        logger.info(
                            "A zero tensor of length %s that represents local error is created.",
                            total_length,
                        )
                        self.error_dict[bucket_index] = torch.zeros(
                            total_length, device=gradient.device, dtype=self.dtype
                        )
                my_observer = torch.ao.quantization.MinMaxObserver(dtype=self.dtype).to(gradient.device)
                my_observer(gradient)

                s, z = my_observer.calculate_qparams()
                s_and_z = torch.FloatTensor([s, z]).to(gradient.device)
                all_ranks_s_and_z = _get_all_gather_out_list(s_and_z, world_size)
                # First, allgather scale and zeros.
                fut_sz = dist.all_gather(
                    all_ranks_s_and_z, s_and_z, group=process_group, async_op=True
                )
                self.all_infos.append({
                    "fut": fut_sz,
                    "gradient": gradient,
                })

        for info in self.all_infos:
            gradient = info["gradient"]
            fut_sz = info["fut"]
            def quantize_and_allgather(fut):
                # Store scale and zeros across all workers.
                s_and_z_synced = fut.value()[0]
                # All workers quantize their own ``GradBucket`` tensors.
                quantized_gradient = _quantize_per_tensor_backend(
                    gradient, s_and_z_synced[rank][0], s_and_z_synced[rank][1], self.dtype
                )
                # Store quantization error in error_dict
                if self.use_error_feedback:
                    if quantized_gradient is None:
                        raise AssertionError
                    self.error_dict[bucket_index] = gradient - quantized_gradient
                # Allgather quantized tensors.
                fut_q = dist.all_gather(
                    _get_all_gather_out_list(quantized_gradient, world_size),
                    quantized_gradient,
                    group=process_group,
                    async_op=True,
                ).get_future()
                return fut_q

            def dequantize_and_aggregate(fut):
                all_ranks_quantized_tensor = fut.value()[0]

                aggregated_dequantized_tensor = torch.zeros_like(
                    all_ranks_quantized_tensor[0], device=gradient.device, dtype=original_dtype
                )
                # Using previously allgathered scales and zeros, dequantize gradient tensors
                # locally and then aggregate them.
                for r, quantized_tensor in enumerate(all_ranks_quantized_tensor):
                    aggregated_dequantized_tensor += _dequantize_per_tensor_backend(
                        quantized_tensor, all_ranks_s_and_z[r][0], all_ranks_s_and_z[r][1]
                    )

                return aggregated_dequantized_tensor / world_size

            self.all_futs.append(
                fut_sz.then(quantize_and_allgather).then(dequantize_and_aggregate)
            )

    def finish_grad_sync(self):
        if self.all_futs:
            return
        for fut in self.all_futs:
            fut.wait()
        self.all_futs = []



