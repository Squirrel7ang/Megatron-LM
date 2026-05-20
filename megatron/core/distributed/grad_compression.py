import math
import logging
from typing import List

import torch
import torch.distributed as dist
from torch.distributed import _coalescing_manager

from .param_and_grad_buffer import _ParamAndGradBucket
from .overlap_tracker import overlap_tracker

try:
    from bitscom import LowBitGroup
except:
    LowBitGroup = None

_ARC_TOPK_STATE = None
_GRAD_QUANTIZATION_STATE = None
_GRAD_QUANTIZATION2_STATE = None

logger = logging.getLogger(__name__)


def init_arc_topk_state(
        process_group,
        priority_rank,
        compression_ratio=0.1,
        use_error_feedback=False,
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


def init_grad_quantization2_state(
        process_group,
        use_error_feedback,
        bitwidth: int = 4,
        stochastic_rounding: bool = False,
        block_size: int = 256,
):
    """Initialize GradQuantization2State using bitscom library.

    Args:
        process_group: The process group to use for communication.
        use_error_feedback: Whether to use error feedback mechanism.
        bitwidth: Quantization bit width (1, 2, 4, 8, 12, 16).
        stochastic_rounding: Whether to use stochastic rounding.
        block_size: Block size for block-wise quantization.
    """
    global _GRAD_QUANTIZATION2_STATE
    if _GRAD_QUANTIZATION2_STATE is None:
        _GRAD_QUANTIZATION2_STATE = GradQuantization2State(
            process_group=process_group,
            use_error_feedback=use_error_feedback,
            bitwidth=bitwidth,
            stochastic_rounding=stochastic_rounding,
            block_size=block_size,
        )
    return _GRAD_QUANTIZATION2_STATE


def get_arc_topk_state():
    if _ARC_TOPK_STATE is None:
        raise ValueError("ARC-Top-K state is not initialized.")
    return _ARC_TOPK_STATE


def get_grad_quantization_state():
    if _GRAD_QUANTIZATION_STATE is None:
        raise ValueError("Gradient quantization state is not initialized.")
    return _GRAD_QUANTIZATION_STATE


def get_grad_quantization2_state():
    if _GRAD_QUANTIZATION2_STATE is None:
        raise ValueError("Gradient quantization2 state is not initialized.")
    return _GRAD_QUANTIZATION2_STATE


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


cnt=0


class ArcTopkState:
    def __init__(
        self,
        process_group,
        priority_rank,
        compression_ratio,
        use_error_feedback,
    ):
        logger.info(
            "ArcTopkState: priority_rank=%d, compression_ratio=%.4f; ",
            priority_rank,
            compression_ratio,
        )

        self.process_group = process_group
        self.priority_rank = priority_rank
        self.compression_ratio = compression_ratio
        self.use_error_feedback = use_error_feedback
        self.error_dict = {}
        self.all_futs = []
        self.infos = {}
        self.cm = None
        self.stream_context = None
        self.buckets = []
        
        # 动态调整相关
        self._prev_compression_ratio = compression_ratio
        self._prev_overlap_ratio = 0.0
        self._adjustment_count = 0
        self._min_compression_ratio = 0.25  # 最小压缩比 10%
        self._max_compression_ratio = 1.0  # 最大压缩比 100%


    def _adjust_compression_ratio(self):
        """根据 OverlapRatio 动态调整 Compression Ratio
        
        调整策略：
        - 如果 OverlapRatio 相比上一轮增加，说明通信成为瓶颈，需要降低 Compression Ratio
        - 每次降低 5%
        - 如果 OverlapRatio 不增加或降低，则保持当前 Compression Ratio 不变
        
        Compression Ratio 含义：
        - 0.75 表示保留 75% 的梯度（压缩到 75%）
        - 1.0 表示保留 100% 的梯度（不压缩）
        - 0.1 表示保留 10% 的梯度（激进压缩）
        """
        from megatron.core.distributed.overlap_tracker import get_overlap_tracker
        
        try:
            overlap_tracker = get_overlap_tracker()
            current_overlap_ratio = overlap_tracker.get_overlap_ratio()

            # 第一轮不调整，只记录初始值
            if self._adjustment_count == 0:
                self._prev_overlap_ratio = current_overlap_ratio
                self._prev_compression_ratio = self.compression_ratio
                self._adjustment_count += 1
                logger.info(
                    f"[ArcTopk] Initial state: compression_ratio={self.compression_ratio:.4f}, "
                    f"overlap_ratio={current_overlap_ratio:.2%}"
                )
                return
            
            # 计算 OverlapRatio 的变化
            overlap_delta = current_overlap_ratio - self._prev_overlap_ratio
            
            # 如果 OverlapRatio 增加（通信成为瓶颈），降低 Compression Ratio
            if overlap_delta > 0:
                new_compression_ratio = max(
                    self._min_compression_ratio,
                    self.compression_ratio - 0.05  # 降低 5%
                )
                
                if new_compression_ratio != self.compression_ratio:
                    logger.info(
                        f"[ArcTopk] OverlapRatio increased from {self._prev_overlap_ratio:.2%} "
                        f"to {current_overlap_ratio:.2%} (delta={overlap_delta:.2%}), "
                        f"reducing compression_ratio from {self.compression_ratio:.4f} "
                        f"to {new_compression_ratio:.4f}"
                    )
                    self.compression_ratio = new_compression_ratio
                else:
                    logger.info(
                        f"[ArcTopk] OverlapRatio increased but compression_ratio already at minimum "
                        f"({self.compression_ratio:.4f}), keeping unchanged"
                    )
            else:
                # OverlapRatio 不增加或降低，保持当前 Compression Ratio
                logger.info(
                    f"[ArcTopk] OverlapRatio stable/decreased from {self._prev_overlap_ratio:.2%} "
                    f"to {current_overlap_ratio:.2%} (delta={overlap_delta:.2%}), "
                    f"keeping compression_ratio at {self.compression_ratio:.4f}"
                )
            
            # 更新历史记录
            self._prev_overlap_ratio = current_overlap_ratio
            self._prev_compression_ratio = self.compression_ratio
            self._adjustment_count += 1
            
        except Exception as e:
            logger.warning(f"[ArcTopk] Failed to adjust compression ratio: {e}")


    def start_grad_sync(
        self,
        buckets: List[_ParamAndGradBucket],
        stream_context,
        # stream_context: torch.cuda.StreamContext,
        async_op: bool = False,
        adjust_compression_ratio: bool = False,
    ):
        self._adjust_compression_ratio()
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
        global cnt
        cnt += 1

        self.all_futs = []
        
        process_group = self.process_group
        rank = process_group.rank()
        # logger.info(f"Start Grad Sync Begin {cnt=}, rank={rank}")

        self.stream_context = stream_context
        self.buckets = buckets
        # 显式遍历每一个 bucket
        with stream_context, _coalescing_manager(self.process_group, async_ops=async_op) as cm:
            for bucket in buckets:

                def _cal_max_factor(size: int):
                    factor: int = 1
                    while size % factor == 0 and size / factor > factor:
                        factor *= 2
                    if size % factor != 0:
                        factor = factor // 2
                    return factor

                # 在 Megatron 中使用 bucket.grad_data 代替 bucket.buffer()
                gradient = bucket.grad_data
                if (
                    self.use_error_feedback
                    and bucket.bucket_id in self.error_dict
                    and self.error_dict[bucket.bucket_id] is not None
                ):
                    gradient += self.error_dict[bucket.bucket_id].to(gradient.device).view(gradient.shape)
                d = gradient.numel()
                row_num = _cal_max_factor(d)
                gradient = gradient.view(-1, row_num)

                # calculate global priority
                V = torch.randn(row_num, self.priority_rank, device=gradient.device, dtype=gradient.dtype)
                P = torch.matmul(gradient, V) / math.sqrt(self.priority_rank)
                
                # 发起第一次 All-Reduce
                if adjust_compression_ratio:
                    overlap_tracker.start_gpu_communication()
                dist.all_reduce(
                    P, group=self.process_group, async_op=async_op
                )
                if adjust_compression_ratio:
                    overlap_tracker.stop_gpu_communication()
                self.infos[bucket.bucket_id] = {
                    "gradient": gradient,
                    "P": P,
                    "d": d,
                    "row_num": row_num,
                }
                # logger.info(f"{bucket.bucket_id=}, {d=}, {row_num=}")
                
        # 先保证正确性。理论上一个 CUDA Stream 应该是顺序执行的。
        cm.wait()

        with stream_context, _coalescing_manager(self.process_group, async_ops=async_op) as cm:
            for bucket in buckets:
                info = self.infos[bucket.bucket_id]
                gradient = info["gradient"]
                P = info["P"]
                d = info["d"]
                row_num = info["row_num"]

                # logger.info(f"290: {bucket.bucket_id=}, {d=}, {row_num=}")
                
                score = (P * P).sum(dim=1)
                col_num = d / row_num
                K = max(1, int(col_num * self.compression_ratio))
                indices = torch.topk(score, k=K).indices.sort().values
                
                comm_gradient = gradient[indices, :]
                if self.use_error_feedback:
                    err_gradient = gradient.clone()
                    err_gradient[indices, :] = 0
                    err_gradient = gradient - err_gradient
                    self.error_dict[bucket.bucket_id] = err_gradient.to('cpu')
                    
                if adjust_compression_ratio:
                    overlap_tracker.start_gpu_communication()
                dist.all_reduce(
                    comm_gradient, group=self.process_group, async_op=async_op
                )
                if adjust_compression_ratio:
                    overlap_tracker.stop_gpu_communication()
                self.infos[bucket.bucket_id]["comm_gradient"] = comm_gradient
                self.infos[bucket.bucket_id]["indices"] = indices
                
        self.cm = cm
        # logger.info(f"Start Grad Sync End {cnt=}, rank={rank}")


        # with stream_context:
        #     for bucket in buckets:
        #         # --- 以下代码完全保留源代码变量名称与逻辑 ---
                
        #         process_group = self.process_group
        #         # 适配 Megatron 进程组获取逻辑
        #         group_to_use = (
        #             process_group if process_group is not None else dist.group.WORLD
        #         )
        #         world_size = dist.get_world_size(group=group_to_use)

        #         def _cal_max_factor(size: int):
        #             factor: int = 1
        #             while size % factor == 0 and size / factor > factor:
        #                 factor *= 2
        #             if size % factor != 0:
        #                 factor = factor // 2
        #             return factor

        #         # 在 Megatron 中使用 bucket.grad_data 代替 bucket.buffer()
        #         gradient = bucket.grad_data
        #         d = gradient.numel()
        #         row_num = _cal_max_factor(d)
        #         gradient = gradient.view(-1, row_num)

        #         # calculate global priority
        #         V = torch.randn(row_num, self.priority_rank, device=gradient.device, dtype=gradient.dtype)
        #         P = torch.matmul(gradient, V) / math.sqrt(self.priority_rank)
                
        #         # 发起第一次 All-Reduce
        #         fut = dist.all_reduce(
        #             P, group=self.process_group, async_op=async_op
        #         ).get_future()
                
        #         # 为当前 bucket 独立维护 indices_holder 避免多 bucket 并发干扰
        #         indices_holder = []

        #         def compress_and_allreduce(fut):
        #             # fut 是前一个阶段返回的 future，通过 wait() 或 value() 获取 P
        #             score = (P * P).sum(dim=1)
        #             # score = torch.diag(torch.matmul(P, P.T))
        #             col_num = d / row_num
        #             K = max(1, int(col_num * self.compression_ratio))
        #             indices = torch.topk(score, k=K).indices
        #             indices_holder.append(indices)

        #             comm_gradient = gradient[indices, :]
        #             comm_fut = dist.all_reduce(
        #                 comm_gradient, group=self.process_group, async_op=async_op
        #             ).get_future()

        #             # 原代码逻辑：返回 wait() 的结果
        #             return comm_fut.wait()

        #         def decompress_and_finalize(fut):
        #             # 获取第二次 All-Reduce 的结果
        #             avg_gradient = fut.wait()[0] / world_size
        #             indices = indices_holder[0]
        #             gradient.zero_()
        #             gradient[indices, :] = avg_gradient

        #             return gradient

        #         # 链式调用并将最终的 future 存入列表
        #         # 注意：这里的 .then() 逻辑会按顺序在后台执行
        #         chained_fut = fut.then(compress_and_allreduce).then(decompress_and_finalize)
        #         self.all_futs.append(chained_fut)

    def finish_grad_sync(
        self,
        # buckets: List[_ParamAndGradBucket],
        # stream_context,
    ):
        """
        确保在这之前完成所有梯度的同步。
        """
        # if self.all_futs:
        #     for fut in self.all_futs:
        #         # 阻塞直到该 bucket 的整个异步链条（P 同步 -> 计算 -> G 同步 -> 还原）完成
        #         fut.wait()
        #     # 清空以备下一轮使用
        #     self.all_futs = []
        
        if self.stream_context == None:
            return
        
        rank = self.process_group.rank() if self.process_group else -1
        global cnt
        # logger.info(f"393:{rank=}, {cnt=}, {self.stream_context=}, {len(self.buckets)=}")

        group_to_use = (
            self.process_group if self.process_group is not None else dist.group.WORLD
        )
        world_size = dist.get_world_size(group=group_to_use)
        
        with self.stream_context:
            self.cm.wait()
            for bucket in self.buckets:
                info = self.infos[bucket.bucket_id]
                gradient = info["gradient"]
                comm_gradient = info["comm_gradient"]
                indices = info["indices"]
                avg_gradient = comm_gradient / world_size
                gradient.zero_()
                gradient[indices, :] = avg_gradient
        self.infos = {}
        self.stream_context = None
        self.buckets = []


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
        self.all_infos = {}

    def start_grad_sync(
            self,
            buckets: List[_ParamAndGradBucket],
            stream_context: torch.cuda.StreamContext,
            async_op: bool = False,
            adjust_compression_ratio: bool = False,
    ):
        def _get_all_gather_out_list(tensor, world_size):
            shape = (world_size,) + tensor.shape
            coalescing_tensor = torch.zeros(
                size=shape,
                dtype=tensor.dtype,
                device=tensor.device,
            )

            out_list = [coalescing_tensor[r] for r in range(world_size)]

            return coalescing_tensor, out_list

        # def coalescing_all_gather(coalescing_tensor, tensor, group=None, async_op=False):
        #     dist.all_gather_into_tensor(
        #         coalescing_tensor, tensor, group=group, async_op=async_op
        #     )

        process_group = self.process_group
        rank = process_group.rank() if process_group is not None else dist.get_rank()
        # pyrefly: ignore [missing-attribute]
        world_size = process_group.size()
        original_dtype = buckets[0].grad_data.dtype if buckets else torch.float32

        # with stream_context, _coalescing_manager(self.process_group, async_ops=async_op) as cm:
        #     for bucket in buckets:
        #         gradient = bucket.grad_data
        #         bucket_index = bucket.bucket_id
        #         total_length = gradient.shape[0]
        #         if self.use_error_feedback:
        #             if bucket_index in self.error_dict:
        #                 gradient.add_(self.error_dict[bucket_index])
        #             else:
        #                 logger.info(
        #                     "A zero tensor of length %s that represents local error is created.",
        #                     total_length,
        #                 )
        #                 self.error_dict[bucket_index] = torch.zeros(
        #                     total_length, device=gradient.device, dtype=self.dtype
        #                 )
        #         my_observer = torch.ao.quantization.MinMaxObserver(dtype=self.dtype).to(gradient.device)
        #         my_observer(gradient)
        #
        #         s, z = my_observer.calculate_qparams()
        #         s_and_z = torch.FloatTensor([s, z]).to(gradient.device)
        #         all_ranks_s_and_z = _get_all_gather_out_list(s_and_z, world_size)
        #         # First, allgather scale and zeros.
        #         dist.all_gather(
        #             all_ranks_s_and_z, s_and_z, group=process_group, async_op=True
        #         )
        #         self.all_infos.append({
        #             "gradient": gradient,
        #         })


        with stream_context, _coalescing_manager(self.process_group, async_ops=async_op) as cm:
            for bucket in buckets:
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

                d_min, d_max = _get_dtype_range(self.dtype)
                t_min, t_max = gradient.min(), gradient.max()
                
                s = (t_max - t_min) / (d_max - d_min)
                z = d_min - t_min / s

                s_and_z = torch.FloatTensor([s, z]).to(gradient.device)
                s_and_z_dtype = s_and_z.dtype
                comm_s_and_z = s_and_z.view(self.dtype)
                s_and_z_size = comm_s_and_z.numel()

                quantized_gradient = _quantize_per_tensor_backend(
                    gradient, s, z, self.dtype
                )
                grad_shape = quantized_gradient.shape
                comm_gradient = quantized_gradient.flatten()
                if self.use_error_feedback:
                    self.error_dict[bucket_index] = gradient - _dequantize_per_tensor_backend(
                        quantized_gradient, s, z, dtype=original_dtype,
                    )

                packed_data = torch.cat([comm_s_and_z, comm_gradient])

                coalescing_data, packed_data_list = _get_all_gather_out_list(packed_data, world_size)

                dist.all_gather_into_tensor(
                    coalescing_data,
                    packed_data,
                    group=process_group,
                    async_op=async_op,
                )

                self.all_infos[bucket_index] = {
                    "packed_data_list": packed_data_list,
                    "grad_shape": grad_shape,
                    "s_and_z_dtype": s_and_z_dtype,
                    "s_and_z_size": s_and_z_size,
                }

        for bucket in buckets:
            info = self.all_infos[bucket.bucket_id]
            packed_data_list = info["packed_data_list"]
            grad_shape = info["grad_shape"]
            s_and_z_size = info["s_and_z_size"]
            s_and_z_dtype = info["s_and_z_dtype"]

            aggregated_grad = torch.zeros(size=grad_shape, dtype=original_dtype, device=comm_gradient.device)

            self.all_infos[bucket.bucket_id] = None
            for i, packed_data in enumerate(packed_data_list):
                s_and_z = packed_data[:s_and_z_size].view(s_and_z_dtype)
                comm_gradient = packed_data[s_and_z_size:].view(grad_shape)
                aggregated_grad += _dequantize_per_tensor_backend(
                    comm_gradient, s_and_z[0], s_and_z[1], dtype=original_dtype,
                ) / world_size
                packed_data_list[i] = None
            packed_data_list = None

    def finish_grad_sync(self):
        self.all_infos = {}


class GradQuantization2State:
    """
    Gradient quantization state using bitscom library.
    
    This class wraps bitscom's LowBitGroup to provide low-bit gradient compression
    with error feedback mechanism, compatible with All-Reduce operations.
    """
    __slots__ = [
        "process_group",
        "use_error_feedback",
        "error_dict",
        "bitwidth",
        "stochastic_rounding",
        "block_size",
        "lowbit_group",
        "all_infos",
        "stream_context",
        "buckets",
        "cm",
    ]

    def __init__(
        self,
        process_group,
        use_error_feedback: bool,
        bitwidth: int = 4,
        stochastic_rounding: bool = False,
        block_size: int = 256,
    ):
        """
        Initialize GradQuantization2State.
        
        Args:
            process_group: The process group to use for communication.
            use_error_feedback: Whether to use error feedback mechanism.
            bitwidth: Quantization bit width (1, 2, 4, 8, 12, 16).
            stochastic_rounding: Whether to use stochastic rounding.
            block_size: Block size for block-wise quantization.
        """
        # Import bitscom here to avoid import issues during module loading
        # self._has_bitscom = False
        if LowBitGroup is None:
            logger.warning(f"bitscom library not found")

        self.process_group = process_group
        self.use_error_feedback = use_error_feedback
        self.bitwidth = bitwidth
        self.stochastic_rounding = stochastic_rounding
        self.block_size = block_size
        self.error_dict = {}
        self.all_infos = {}
        self.stream_context = None
        self.buckets = []
        self.cm = None
        
        # Initialize LowBitGroup if bitscom is available
        self.lowbit_group = LowBitGroup(
            bitwidth=bitwidth,
            process_group=process_group,
            stochastic_rounding=stochastic_rounding,
            block_size=block_size,
        )

    def start_grad_sync(
            self,
            buckets: List[_ParamAndGradBucket],
            stream_context: torch.cuda.StreamContext,
            async_op: bool = False,
            adjust_compression_ratio: bool = False,
    ):
        """
        Start gradient synchronization using bitscom low-bit All-Reduce.
        
        Args:
            buckets: List of parameter and gradient buckets.
            stream_context: CUDA stream context for async operations.
            async_op: Whether to perform asynchronous operations.
        """

        process_group = self.process_group
        rank = process_group.rank() if process_group is not None else dist.get_rank()
        world_size = process_group.size()
        
        self.stream_context = stream_context
        self.buckets = buckets

        def sync_bucket(
                b,
        ):
            gradient = b.grad_data
            bucket_index = b.bucket_id

            # Apply error feedback if enabled
            if self.use_error_feedback:
                if bucket_index in self.error_dict:
                    gradient.add_(self.error_dict[bucket_index])
                else:
                    logger.info(
                        "A zero tensor of length %s that represents local error is created.",
                        gradient.numel(),
                    )
                    self.error_dict[bucket_index] = torch.zeros(
                        gradient.shape,
                        device=gradient.device,
                        dtype=gradient.dtype
                    )

            # Store original gradient for error calculation
            original_grad = gradient.clone() if self.use_error_feedback else None

            # Start GPU communication tracking
            if adjust_compression_ratio:
                overlap_tracker.start_gpu_communication()

                # Use bitscom's low-bit all_reduce
                self.lowbit_group.all_reduce(
                    gradient,
                    op=dist.ReduceOp.SUM,
                    async_op=async_op,
                )

            # Stop GPU communication tracking
            if adjust_compression_ratio:
                overlap_tracker.stop_gpu_communication()

            # Calculate error for feedback
            if self.use_error_feedback:
                # Error = original - (reduced / world_size)
                reduced_avg = gradient / world_size
                self.error_dict[bucket_index] = original_grad - reduced_avg

            # Store info for finish_grad_sync
            self.all_infos[bucket_index] = {
                "gradient": gradient,
                "world_size": world_size,
            }
            pass

        if self.bitwidth >= 8:
            with stream_context, _coalescing_manager(self.process_group, async_ops=async_op) as cm:
                for bucket in buckets:
                    sync_bucket(bucket)
        else:
            with stream_context:
                for bucket in buckets:
                    sync_bucket(bucket)
                # gradient = bucket.grad_data
                # bucket_index = bucket.bucket_id
                #
                # # Apply error feedback if enabled
                # if self.use_error_feedback:
                #     if bucket_index in self.error_dict:
                #         gradient.add_(self.error_dict[bucket_index])
                #     else:
                #         logger.info(
                #             "A zero tensor of length %s that represents local error is created.",
                #             gradient.numel(),
                #         )
                #         self.error_dict[bucket_index] = torch.zeros(
                #             gradient.shape,
                #             device=gradient.device,
                #             dtype=gradient.dtype
                #         )
                #
                # # Store original gradient for error calculation
                # original_grad = gradient.clone() if self.use_error_feedback else None
                #
                # # Start GPU communication tracking
                # if adjust_compression_ratio:
                #     overlap_tracker.start_gpu_communication()
                #
                # # Use bitscom's low-bit all_reduce
                #     self.lowbit_group.all_reduce(
                #         gradient,
                #         op=dist.ReduceOp.SUM,
                #         async_op=async_op,
                #     )
                #
                # # Stop GPU communication tracking
                # if adjust_compression_ratio:
                #     overlap_tracker.stop_gpu_communication()
                #
                # # Calculate error for feedback
                # if self.use_error_feedback:
                #     # Error = original - (reduced / world_size)
                #     reduced_avg = gradient / world_size
                #     self.error_dict[bucket_index] = original_grad - reduced_avg
                #
                # # Store info for finish_grad_sync
                # self.all_infos[bucket_index] = {
                #     "gradient": gradient,
                #     "world_size": world_size,
                # }
        
        self.cm = cm

    def finish_grad_sync(self):
        """
        Finalize gradient synchronization.
        
        Averages the reduced gradients by world_size.
        """
        if self.stream_context is None:
            return
        
        with self.stream_context:
            if self.cm is not None:
                self.cm.wait()
            
            for bucket in self.buckets:
                info = self.all_infos.get(bucket.bucket_id)
                if info is not None:
                    gradient = info["gradient"]
                    world_size = info["world_size"]
                    gradient.div_(world_size)
        
        # Clean up
        self.all_infos = {}
        self.stream_context = None
        self.buckets = []
        self.cm = None