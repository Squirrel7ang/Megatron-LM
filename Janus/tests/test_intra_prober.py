import os
import sys
import unittest
import torch
import socket
from unittest.mock import MagicMock, patch

# Ensure the core module is discoverable
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.common.cluster_context import GPUInfo, NodeInfo, CollectiveType, CommPerformance
from core.common.prober.network.intra_prober import IntraNodeProber

class TestIntraNodeProber(unittest.TestCase):
    def setUp(self):
        """
        Initialize NodeInfo with 8 GPUs. 
        Adjusts local_id automatically if CUDA_VISIBLE_DEVICES is present.
        """
        self.gpus = []
        # Check if environment is restricted to a subset of GPUs
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        
        for i in range(8):
            physical_id = i + 6
            # If visible_devices is set, physical GPU 6 maps to logical 0, etc.
            logical_id = i if visible_devices else physical_id
            
            gpu = GPUInfo(
                global_id=physical_id,
                local_id=logical_id,
                pci_bus_id=f"0000:{13+i:02x}:00.0",
                type="TianGai-V150",
                memory_capacity_gb=32.0,
                theoretical_tflops=100.0
            )
            self.gpus.append(gpu)
        
        self.node_info = NodeInfo(
            node_id=0,
            hostname=socket.gethostname(),
            ip="10.31.10.210",
            cpu_cores=128,
            sys_mem_gb=512.0,
            gpus=self.gpus
        )

    def test_real_intra_node_performance(self):
        """
        Executes actual benchmarks and prints detailed performance tables.
        """
        print(f"\n" + "="*80)
        print(f" INTER-NODE COMMUNICATION BENCHMARK REPORT ")
        print(f" Node: {self.node_info.hostname} | Target GPUs: {[g.global_id for g in self.node_info.gpus]}")
        print("="*80)

        prober = IntraNodeProber(self.node_info)
        
        # In a real environment, this triggers NCCL/HCCL kernels
        # For this test, we assume CollectiveType includes the 4 main types
        prober.probe()

        # --- 1. Collective Communication Table ---
        print(f"\n[Part 1: Collective Communication Performance]")
        print(f"{'Collective Op':<20} | {'Latency (us)':<15} | {'Bus BW (Gbps)':<15} | {'Alg BW (Gbps)':<15}")
        print("-" * 75)
        
        collectives_to_show = [
            CollectiveType.ALL_REDUCE, 
            CollectiveType.ALL_GATHER, 
            CollectiveType.REDUCE_SCATTER, 
            CollectiveType.BROADCAST
        ]

        for coll in collectives_to_show:
            perf = self.node_info.intra_node_comm.get(coll)
            if perf:
                print(f"{coll.value:<20} | {perf.latency_us:<15.2f} | {perf.bus_bandwidth_gbps:<15.2f} | {perf.bandwidth_gbps:<15.2f}")
            else:
                print(f"{coll.value:<20} | {'N/A':<15} | {'N/A':<15} | {'N/A':<15}")

        # --- 2. P2P Topology Matrix (Bandwidth) ---
        print(f"\n[Part 2: P2P Bandwidth Matrix (Gbps)]")
        gpu_ids = [g.global_id for g in self.node_info.gpus]
        
        # Header
        header = "      " + "".join([f"GPU{id:<6}" for id in gpu_ids])
        print(header)
        
        for src in gpu_ids:
            row = f"GPU{src:<2} "
            for dst in gpu_ids:
                if src == dst:
                    row += f"{'X':<9}"
                else:
                    # Look up in the probed topology
                    # Note: We store by global_id in the matrix
                    entry = self.node_info.intra_node_topology.get((src, dst))
                    if entry:
                        bw = entry.get('bandwidth_gbps', 0)
                        row += f"{bw:<9.1f}"
                    else:
                        row += f"{'-':<9}"
            print(row)

        # --- 3. P2P Topology Matrix (Latency) ---
        print(f"\n[Part 3: P2P Latency Matrix (us)]")
        print(header)
        for src in gpu_ids:
            row = f"GPU{src:<2} "
            for dst in gpu_ids:
                if src == dst:
                    row += f"{'X':<9}"
                else:
                    entry = self.node_info.intra_node_topology.get((src, dst))
                    if entry:
                        lat = entry.get('latency_us', 0)
                        row += f"{lat:<9.2f}"
                    else:
                        row += f"{'-':<9}"
            print(row)
        
        print("\n" + "="*80)

if __name__ == "__main__":
    # Ensure standard output is flushed to see results immediately
    unittest.main()