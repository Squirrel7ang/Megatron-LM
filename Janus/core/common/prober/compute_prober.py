import time
import gc
from typing import Tuple
from core.common.cluster_context import GPUInfo, NodeInfo

try:
    import torch
except ImportError:
    torch = None

class ComputeProber:
    def __init__(self, node_info: NodeInfo):
        self.node_info = node_info
        self.matmul_sizes = [4096, 8192, 12288, 16384]
        self.warmup_iters = 10
        self.benchmark_iters = 50
        
        # 256MB elements (1GB for Float32). Should be large enough to overflow L2 cache 
        # (e.g., H100 has ~50MB L2) to measure closer-to-HBM bandwidth.
        self.bw_tensor_size = 256 * 1024 * 1024 

        # Note: TF32 primarily affects FP32 inputs, but keeping it for completeness 
        # in case mixed-precision workloads fall back to it.
        if torch and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

    def probe(self) -> NodeInfo:
        if not torch: return self.node_info

        print(f"[ComputeProber] Starting Architecture-Level Profiling for {len(self.node_info.gpus)} GPUs...")

        for gpu in self.node_info.gpus:
            try:
                device_id = gpu.local_id
                device_type = self._get_device_type()
                
                with self._device_context(device_id, device_type):
                    # 1. Sweep TFLOPS and get the optimal matrix size
                    peak_gemm, best_size = self._sweep_tflops(device_id, device_type)
                    gpu.peak_gemm_tflops = round(peak_gemm, 2)
                    gpu.best_matmul_size = best_size
                    
                    # 2. Measure empirical Memory Bandwidth
                    bandwidth = self._measure_bandwidth(device_id, device_type)
                    gpu.memory_bandwidth_gbps = round(bandwidth, 2)
                    
                    # 3. Calculate GEMM Efficiency vs Theoretical Peak
                    if gpu.theoretical_tflops > 0:
                        gpu.gemm_efficiency = round(gpu.peak_gemm_tflops / gpu.theoretical_tflops, 4)
                    
                    # 4. Calculate Hardware Ridge Point (The fundamental hardware characteristic)
                    # Formula: TFLOPS (1e12) / GBps (1e9) = FLOPS per Byte = (TFLOPS * 1000) / GBps
                    if gpu.memory_bandwidth_gbps > 0:
                        gpu.ridge_point_flops_per_byte = round(
                            (gpu.peak_gemm_tflops * 1000) / gpu.memory_bandwidth_gbps, 2
                        )
                    
                    print(f"  -> GPU[{device_id}] Peak GEMM: {gpu.peak_gemm_tflops} TFLOPS "
                          f"(Eff: {gpu.gemm_efficiency*100:.1f}%) | "
                          f"Best Size: {gpu.best_matmul_size} | "
                          f"BW: {gpu.memory_bandwidth_gbps} GB/s | "
                          f"Ridge Point: {getattr(gpu, 'ridge_point_flops_per_byte', 0)} FLOPS/Byte")
                
            except Exception as e:
                print(f"  -> GPU[{gpu.local_id}] Profiling failed: {e}")
            finally:
                # Clear cache only at the very end of this GPU's profiling session
                # to prevent allocator jitter during the micro-benchmarks.
                self._clear_cache(device_type)

        return self.node_info

    def _sweep_tflops(self, device_id: int, device_type: str) -> Tuple[float, int]:
        """
        Sweeps multiple matrix sizes. 
        Returns: (Absolute Peak TFLOPS, Optimal Matrix Size)
        """
        device_str = f"{device_type}:{device_id}"
        if device_type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float16  # safer fallback
        
        peak_tflops = 0.0
        best_size = 0

        for n in self.matmul_sizes:
            a = torch.randn(n, n, device=device_str, dtype=dtype)
            b = torch.randn(n, n, device=device_str, dtype=dtype)
            
            for _ in range(self.warmup_iters):
                torch.matmul(a, b)
            self._sync(device_id, device_type)

            start_event, end_event = self._get_events(device_type)
            start_event.record()
            for _ in range(self.benchmark_iters):
                torch.matmul(a, b)
            end_event.record()
            
            self._sync(device_id, device_type)
            
            elapsed_ms = start_event.elapsed_time(end_event)
            avg_time_s = (max(elapsed_ms, 0.01) / 1000.0) / self.benchmark_iters
            
            current_tflops = (2.0 * n**3) / (avg_time_s * 1e12)
            
            if current_tflops > peak_tflops:
                peak_tflops = current_tflops
                best_size = n
                
            del a, b

        return peak_tflops, best_size

    def _measure_bandwidth(self, device_id: int, device_type: str) -> float:
        device_str = f"{device_type}:{device_id}"
        size = self.bw_tensor_size
        a = torch.randn(size, device=device_str, dtype=torch.float32)
        b = torch.empty_like(a)

        for _ in range(self.warmup_iters):
            b.copy_(a)
        self._sync(device_id, device_type)

        start_event, end_event = self._get_events(device_type)
        start_event.record()
        for _ in range(self.benchmark_iters):
            b.copy_(a)
        end_event.record()
        
        self._sync(device_id, device_type)
        
        elapsed_ms = start_event.elapsed_time(end_event)
        avg_time_s = (max(elapsed_ms, 0.01) / 1000.0) / self.benchmark_iters
        
        gbps = (2.0 * a.nelement() * a.element_size()) / (avg_time_s * 1e9)
        return gbps

    def _get_device_type(self) -> str:
        if torch.cuda.is_available(): return "cuda"
        if hasattr(torch, "npu") and torch.npu.is_available(): return "npu"
        return "cpu"

    def _device_context(self, device_id: int, device_type: str):
        if device_type == "cuda": return torch.cuda.device(device_id)
        if device_type == "npu": return torch.npu.device(device_id)
        class DummyContext:
            def __enter__(self): pass
            def __exit__(self, *args): pass
        return DummyContext()

    def _get_events(self, device_type: str):
        if device_type == "cuda":
            return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        if device_type == "npu":
            return torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        return None, None

    def _sync(self, device_id: int, device_type: str):
        if device_type == "cuda":
            torch.cuda.synchronize(device_id)
        elif device_type == "npu":
            torch.npu.synchronize(device_id)

    def _clear_cache(self, device_type: str):
        gc.collect()
        if device_type == "cuda": torch.cuda.empty_cache()
        if device_type == "npu": torch.npu.empty_cache()