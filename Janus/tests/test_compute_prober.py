import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure the core module is discoverable
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.common.cluster_context import GPUInfo, NodeInfo
from core.common.prober.compute_prober import ComputeProber

class TestComputeProber(unittest.TestCase):
    def setUp(self):
        """
        Initialize a NodeInfo skeleton containing exactly 2 realistic GPUs 
        based on the actual hardware environment (e.g., TianGai-V150).
        """
        # Based on actual environment specs: 32GB memory, 200 TFLOPS theoretical peak
        self.gpu0 = GPUInfo(
            global_id=0,
            local_id=0,
            pci_bus_id="0000:13:00.0",
            type="TianGai-V150",
            memory_capacity_gb=32.0,
            theoretical_tflops=100.0  
        )
        self.gpu1 = GPUInfo(
            global_id=1,
            local_id=1,
            pci_bus_id="0000:16:00.0",
            type="TianGai-V150",
            memory_capacity_gb=32.0,
            theoretical_tflops=100.0
        )
        
        self.node_info = NodeInfo(
            node_id=0,
            hostname="worker-node-01",
            ip="10.31.10.210",
            cpu_cores=128,
            sys_mem_gb=512.0,
            gpus=[self.gpu0, self.gpu1]
        )

    def test_inplace_modification(self):
        """
        Verifies that the ComputeProber correctly calculates metrics and 
        modifies the GPUInfo objects in-place without creating new instances.
        """
        prober = ComputeProber(self.node_info)
        
        # Record the original memory address of the object
        gpu0_id_before = id(self.node_info.gpus[0])
        
        # Mock the PyTorch environment to test the logical flow
        with patch("core.common.prober.compute_prober.torch") as mock_torch:
            mock_torch.is_available.return_value = True
            
            # Simulate measured values for TianGai-V150
            # FIX: Return a tuple matching (peak_tflops, best_size)
            prober._sweep_tflops = MagicMock(return_value=(95.0, 16384))
            prober._measure_bandwidth = MagicMock(return_value=1200.0) 
            prober._clear_cache = MagicMock()

            # Execute the prober
            prober.probe()

            # 1. Verify Object Identity (In-place pipeline check)
            self.assertEqual(id(self.node_info.gpus[0]), gpu0_id_before)
            
            # 2. Verify Data Population (Using new rigorous field names)
            target_gpu = self.node_info.gpus[0]
            self.assertEqual(target_gpu.peak_gemm_tflops, 95.0)
            self.assertEqual(target_gpu.best_matmul_size, 16384)
            self.assertEqual(target_gpu.memory_bandwidth_gbps, 1200.0)
            
            # 3. Verify Efficiency Calculation Logic (95.0 / 100.0 = 0.950)
            expected_efficiency = round(95.0 / 100.0, 4)
            self.assertEqual(target_gpu.gemm_efficiency, expected_efficiency)
            
            # 4. Verify Ridge Point Logic (TFLOPS * 1000 / GBps)
            expected_ridge_point = round((95.0 * 1000) / 1200.0, 2)
            self.assertEqual(target_gpu.ridge_point_flops_per_byte, expected_ridge_point)
            
            print(f"\n[Mock Test] {target_gpu.type} - Measured Efficiency: {target_gpu.gemm_efficiency * 100:.2f}%")

    @unittest.skipUnless(os.environ.get("HAS_GPU"), "Requires real GPU environment for actual hardware benchmarking")
    def test_real_gpu_performance(self):
        """
        Executes actual PyTorch matrix multiplications on the physical GPUs.
        Only runs if the 'HAS_GPU' environment variable is set.
        """
        print("\n--- Starting REAL Hardware Benchmark ---")
        prober = ComputeProber(self.node_info)
        
        # This will run the actual PyTorch logic on local_id 0 and 1
        prober.probe()
        
        for gpu in self.node_info.gpus:
            # FIX: Use getattr to safely check dynamic attributes and use the updated names
            self.assertGreater(getattr(gpu, 'peak_gemm_tflops', 0.0), 0.0)
            self.assertGreater(getattr(gpu, 'memory_bandwidth_gbps', 0.0), 0.0)
            self.assertGreaterEqual(getattr(gpu, 'gemm_efficiency', 0.0), 0.0)
            self.assertGreater(getattr(gpu, 'ridge_point_flops_per_byte', 0.0), 0.0)
            
            print(f"[Real Benchmark] GPU {gpu.local_id} ({gpu.type}):")
            print(f"  -> TFLOPS: {getattr(gpu, 'peak_gemm_tflops', 0)} (Eff: {getattr(gpu, 'gemm_efficiency', 0) * 100:.2f}%)")
            print(f"  -> Bandwidth: {getattr(gpu, 'memory_bandwidth_gbps', 0)} GB/s")
            print(f"  -> Ridge Point: {getattr(gpu, 'ridge_point_flops_per_byte', 0)} FLOPS/Byte")

if __name__ == "__main__":
    print("If GPUs are available, set HAS_GPU=1 to run the test.")
    unittest.main()