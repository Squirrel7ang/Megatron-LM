import logging
import torch
import time
import threading
from typing import Optional, Dict, List, Callable

logger = logging.getLogger(__name__)

class OverlapTracker:
    _lock: threading.Lock = threading.Lock()
    
    def __init__(self) -> None:
        """初始化跟踪器（线程安全）"""
        if getattr(self, '_initialized', False):
            return
        
        # 时间累加器（毫秒）
        self._cpu_compute_time: float = 0.0
        # self._gpu_compute_time: float = 0.0
        self._gpu_communication_time: float = 0.0
        
        # CUDA Events 用于精确计时
        self._cpu_compute_events = []
        # self._gpu_compute_events = []
        self._gpu_communication_events = []

        # GPU Comp
        # self._gpu_compute_start_event = None
        # self._gpu_compute_end_event = None

        # GPU Comm
        self._gpu_communication_start_event = None
        self._gpu_communication_end_event = None

        # CPU 计时
        self._cpu_start_time: float = 0.0
        self._cpu_end_time: float = 0.0
        self._cpu_anchor_time: float = 0.0
        
        # 计数
        self._gpu_compute_count: int = 0
        self._gpu_communication_count: int = 0
        
        # 当前状态
        self._is_recording_cpu: bool = False
        # self._is_recording_gpu: bool = False
        self._is_recording_comm: bool = False

        self._iteration_anchor = None

        self._initialized = True

        self.stats = None
        self.origin_cpu_cuda_synchronize: Callable
    
    # ==================== 上下文管理器（推荐用法） ====================

    def start_recording(self):
        with self._lock:
            self._record_start_time()
            self._start_mock_cpu_cuda_synchronize()
            self.start_cpu_compute()


    def _record_start_time(self):
        self._iteration_anchor = torch.cuda.Event(enable_timing=True)
        self._iteration_anchor.record(torch.cuda.current_stream())
        self._cpu_anchor_time = time.time()

    def stop_recording(self):
        with self._lock:
            self.stats = self._cal_stats()
            self._stop_mock_cpu_cuda_synchronize()
            self._reset()

    # ==================== CPU ====================

    def _start_mock_cpu_cuda_synchronize(self):
        # Mock 掉 torch.cuda.synchronize() 函数
        self.origin_cpu_cuda_synchronize = torch.cuda.synchronize
        torch.cuda.synchronize = _tracked_cuda_synchronize
        pass

    def _stop_mock_cpu_cuda_synchronize(self):
        # 取消 Mock
        torch.cuda.synchronize = self.origin_cpu_cuda_synchronize
        self.origin_cpu_cuda_synchronize = None
        pass

    def start_cpu_compute(self) -> float:
        """开始记录 CPU 计算时间"""
        with self._lock:
            if self._is_recording_cpu:
                raise RuntimeError("CPU compute recording is already in progress")

            self._cpu_start_time = time.time()
            self._is_recording_cpu = True
            return self._cpu_start_time

    def stop_cpu_compute(self) -> float:
        """结束记录 CPU 计算时间，返回本次记录的时间（毫秒）"""
        with self._lock:
            if not self._is_recording_cpu:
                raise RuntimeError("CPU compute recording is not in progress")
            self._cpu_end_time = time.time()

            # 记录 CPU 计算区间（起始时间，终止时间，单位：秒，相对于迭代锚点）
            self._cpu_compute_events.append((
                self._cpu_start_time - self._cpu_anchor_time,
                self._cpu_end_time - self._cpu_anchor_time
            ))

            elapsed_ms = (self._cpu_end_time - self._cpu_start_time) * 1000
            self._cpu_compute_time += elapsed_ms
            self._cpu_compute_count += 1
            self._is_recording_cpu = False

            return elapsed_ms


    # ==================== Comm ====================

    def start_gpu_communication(self, stream: Optional[torch.cuda.Stream] = None) -> None:
        """开始记录 GPU 通信时间

        Args:
            stream: 通信使用的 CUDA stream，默认为当前 stream
        """
        with self._lock:
            if self._is_recording_comm:
                raise RuntimeError("GPU communication recording is already in progress")

            if self._gpu_communication_start_event is None:
                self._gpu_communication_start_event = torch.cuda.Event(enable_timing=True)

            target_stream = stream if stream is not None else torch.cuda.current_stream()
            self._gpu_communication_start_event.record(target_stream)
            self._is_recording_comm = True

    def stop_gpu_communication(self, stream: Optional[torch.cuda.Stream] = None) -> float:
        """结束记录 GPU 通信时间，返回本次记录的时间（毫秒）

        Args:
            stream: 通信使用的 CUDA stream，需与 start_gpu_communication 一致
        """
        with self._lock:
            if not self._is_recording_comm:
                raise RuntimeError("GPU communication recording is not in progress")

            if self._gpu_communication_end_event is None:
                self._gpu_communication_end_event = torch.cuda.Event(enable_timing=True)

            target_stream = stream if stream is not None else torch.cuda.current_stream()
            self._gpu_communication_end_event.record(target_stream)
            self._gpu_communication_end_event.synchronize()

            elapsed_ms = self._gpu_communication_start_event.elapsed_time(self._gpu_communication_end_event)

            # 记录 GPU 通信区间（起始时间，终止时间，单位：秒，相对于迭代锚点）
            start_time = self._iteration_anchor.elapsed_time(self._gpu_communication_start_event) / 1000
            end_time = start_time + elapsed_ms / 1000
            self._gpu_communication_events.append((start_time, end_time))

            self._gpu_communication_time += elapsed_ms
            self._gpu_communication_count += 1
            self._is_recording_comm = False

            return elapsed_ms

    # def start_gpu_compute(self) -> None:
    #     """开始记录 GPU 计算时间"""
    #     if self._is_recording_gpu:
    #         raise RuntimeError("GPU compute recording is already in progress")
    #
    #     if self._gpu_start_event is None:
    #         self._gpu_start_event = torch.cuda.Event(enable_timing=True)
    #
    #     self._gpu_start_event.record(torch.cuda.current_stream())
    #     self._is_recording_gpu = True
    #
    # def end_gpu_compute(self) -> float:
    #     """结束记录 GPU 计算时间，返回本次记录的时间（毫秒）"""
    #     if not self._is_recording_gpu:
    #         raise RuntimeError("GPU compute recording is not in progress")
    #
    #     if self._gpu_end_event is None:
    #         self._gpu_end_event = torch.cuda.Event(enable_timing=True)
    #
    #     self._gpu_end_event.record(torch.cuda.current_stream())
    #     self._gpu_end_event.synchronize()
    #
    #     elapsed_ms = self._gpu_start_event.elapsed_time(self._gpu_end_event)
    #     self._gpu_compute_time += elapsed_ms
    #     self._gpu_compute_count += 1
    #     self._is_recording_gpu = False
    #
    #     return elapsed_ms


    # ==================== 统计信息 ====================

    def _calculate_interval_overlap(self, events1: List[tuple], events2: List[tuple]) -> float:
        """计算两组时间区间的总重叠时间（单位：毫秒）

        Args:
            events1: 第一组时间区间，每个区间为 (start, end)，单位毫秒
            events2: 第二组时间区间，每个区间为 (start, end)，单位毫秒

        Returns:
            总重叠时间（毫秒）
        """
        total_overlap = 0.0

        for start1, end1 in events1:
            for start2, end2 in events2:
                # 计算两个区间的重叠
                overlap_start = max(start1, start2)
                overlap_end = min(end1, end2)

                if overlap_start < overlap_end:
                    total_overlap += overlap_end - overlap_start
                if start2 > end1:
                    break

        return total_overlap

    def _cal_stats(self) -> Dict[str, float]:
        """获取统计信息

        Returns:
            dict: 包含以下字段
                - cpu_compute_time_ms: CPU 计算总时间（毫秒）
                - gpu_communication_time_ms: GPU 通信总时间（毫秒）
                - overlap_time_ms: CPU 计算与 GPU 通信的重叠时间（毫秒）
                - total_time_ms: 总时间（毫秒）
                - cpu_compute_count: CPU 计算次数
                - gpu_communication_count: GPU 通信次数
                - overlap_ratio: 通信-计算重叠率（0.0-1.0）
        """
        # CPU 计算总时间（毫秒）
        cpu_time_ms = self._cpu_compute_time

        # GPU 通信总时间（毫秒）
        communication_time_ms = self._gpu_communication_time
        
        # 计算 CPU 计算与 GPU 通信的重叠时间（毫秒）
        overlap_time_ms = self._calculate_interval_overlap(
            self._cpu_compute_events,
            self._gpu_communication_events,
        )

        # 计算总时间 = CPU 计算时间 + GPU 通信时间 - 重叠时间
        total_time_ms = cpu_time_ms + communication_time_ms - overlap_time_ms

        # 计算重叠率 = 重叠时间 / 总时间
        if total_time_ms <= 0:
            overlap_ratio = 0.0
        else:
            overlap_ratio = overlap_time_ms / total_time_ms
        
        return {
            'cpu_compute_time_ms': cpu_time_ms,
            'gpu_communication_time_ms': communication_time_ms,
            'overlap_time_ms': overlap_time_ms,
            'total_time_ms': total_time_ms,
            'cpu_compute_count': self._cpu_compute_count,
            'gpu_communication_count': self._gpu_communication_count,
            'overlap_ratio': overlap_ratio
        }

    def get_overlap_ratio(self):
        with self._lock:
            return self.stats['overlap_ratio']
    
    def _reset(self) -> None:
        """重置所有计时器和计数器"""
        self._cpu_compute_time = 0.0
        self._gpu_communication_time = 0.0
        self._cpu_compute_count = 0
        self._gpu_communication_count = 0
        self._is_recording_cpu = False
        self._is_recording_comm = False
        self._cpu_compute_events.clear()
        self._gpu_communication_events.clear()
        try:
            self.stop_cpu_compute()
        finally:
            pass
        try:
            self.stop_gpu_communication()
        finally:
            pass

    def step(self):
        with self._lock:
            self.stats = self._cal_stats()
            self._reset()
    
    def __str__(self) -> str:
        """返回可读的统计信息字符串"""
        with self._lock:
            stats = self._cal_stats()
            return (
                f"OverlapTracker Statistics:\n"
                f"  GPU Compute Time: {stats['gpu_compute_time_ms']:.2f} ms ({stats['gpu_compute_count']} calls)\n"
                f"  CPU Compute Time: {stats['cpu_compute_time_ms']:.2f} ms ({stats['cpu_compute_count']} calls)\n"
                f"  GPU Comm Time: {stats['gpu_communication_time_ms']:.2f} ms ({stats['gpu_communication_count']} calls)\n"
                f"  Total Compute Time: {stats['total_compute_time_ms']:.2f} ms\n"
                f"  Overlap Ratio: {stats['overlap_ratio']:.2%}"
            )


# 创建全局实例
overlap_tracker = OverlapTracker()

_flag = False

def _tracked_cuda_synchronize():
    global _flag
    if not _flag:
        _flag = True
        logger.info("CUDA synchronize tracking started, Mock Success!")

    if overlap_tracker.origin_cpu_cuda_synchronize is None:
        raise RuntimeError("CUDA synchronize tracking not started!")


    overlap_tracker.stop_cpu_compute()
    overlap_tracker.origin_cpu_cuda_synchronize()
    overlap_tracker.start_cpu_compute()


def get_overlap_tracker() -> OverlapTracker:
    """获取全局 OverlapTracker 实例"""
    return overlap_tracker