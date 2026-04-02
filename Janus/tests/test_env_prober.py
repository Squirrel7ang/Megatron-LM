import os
import json
import sys
import unittest
from unittest.mock import patch, MagicMock

# Ensure the core module is discoverable
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Updated imports: Removed DistLevel, TopoParser, and NodeTopology
from core.common.prober.env_prober import EnvProber
from core.common.cluster_context import GPUInfo, NodeInfo

class TestEnvProber(unittest.TestCase):
    def setUp(self):
        """Set up a mock cluster config matching the new schema."""
        self.config_path = "mock_cluster_spec.json"
        self.mock_config = {
            "nodes": [
                {
                    "node_id": 0,
                    "hostname": "u210",
                    "ip": "10.31.10.210",
                    "gpus": [
                        {
                            "global_id": i, 
                            "local_id": i, 
                            "type": "TianGai-V150",
                            "theoretical_tflops": 200.0,
                            "memory_capacity_gb": 32.0
                        } 
                        for i in range(16)
                    ]
                }
            ]
        }
        with open(self.config_path, "w") as f:
            json.dump(self.mock_config, f)

    def tearDown(self):
        """Cleanup mock config files."""
        if os.path.exists(self.config_path):
            os.remove(self.config_path)

    @patch("psutil.cpu_count")
    @patch("psutil.virtual_memory")
    @patch("shutil.which")
    @patch("subprocess.check_output")
    def test_env_probing_with_new_data_structures(self, mock_subproc, mock_which, mock_vm, mock_cpu):
        """
        Verify that EnvProber correctly populates the simplified NodeInfo 
        and the enhanced GPUInfo (without topology logic).
        """
        
        # 1. Mock System Hardware Info (u210 Specs: 128 cores, 512GB RAM)
        mock_cpu.return_value = 128
        mock_vm.return_value.total = 512 * 1024**3
        
        # 2. Mock Tool Existence
        def side_effect_which(cmd):
            # We still need vendor tools to get VRAM/PCI IDs, 
            # and ibv_devinfo to detect RDMA/IB.
            if cmd in ["ixsmi", "ibv_devinfo"]:
                return f"/usr/bin/{cmd}"
            return None
        mock_which.side_effect = side_effect_which
        
        # 3. Mock Command Outputs (u210 real-world style)
        def side_effect_output(cmd_list, **kwargs):
            cmd_str = " ".join(cmd_list) if isinstance(cmd_list, list) else cmd_list
            
            # Mock ixsmi: returns PCI_ID, Total_Mem, Free_Mem, Temp
            if "ixsmi" in cmd_str:
                return "\n".join([
                    f"00000000:{13+i:02X}:00.0, 32768, 31000, 45" 
                    for i in range(16)
                ])
            
            # Mock ibv_devinfo: u210 is native InfiniBand
            if "ibv_devinfo" in cmd_str:
                return """
                hca_id: mlx5_0
                    transport: InfiniBand (0)
                    port: 1
                        state: PORT_ACTIVE (4)
                        link_layer: InfiniBand
                """
            return ""
        
        mock_subproc.side_effect = side_effect_output

        # 4. Initialize NodeInfo using your latest dataclass schema
        target_node_spec = self.mock_config["nodes"][0]
        gpu_objects = [
            GPUInfo(
                global_id=s["global_id"],
                local_id=s["local_id"],
                pci_bus_id="N/A", # Will be updated by prober
                type=s["type"],
                memory_capacity_gb=s["memory_capacity_gb"],
                theoretical_tflops=s["theoretical_tflops"]
            ) for s in target_node_spec["gpus"]
        ]
        
        node_info = NodeInfo(
            node_id=target_node_spec["node_id"],
            hostname=target_node_spec["hostname"],
            ip=target_node_spec["ip"],
            gpus=gpu_objects
        )

        # 5. Execute Probing (In-place modification)
        prober = EnvProber(node_info)
        prober.probe()

        # 6. Validation
        print(f"\n--- Strategy-Driven Probing Results (Node: {node_info.hostname}) ---")
        print(f"CPU: {node_info.cpu_cores} Cores | RAM: {node_info.sys_mem_gb} GB")
        print(f"Network: {node_info.nic_type} (RDMA: {node_info.has_rdma})")
        print(f"GPU[0] PCI: {node_info.gpus[0].pci_bus_id}")
        
        # System Level Assertions
        self.assertEqual(node_info.cpu_cores, 128)
        self.assertEqual(node_info.sys_mem_gb, 512.0)
        self.assertEqual(node_info.nic_type, "IB")
        self.assertTrue(node_info.has_rdma)

        # GPU Level Assertions (Telemetry & ID)
        self.assertEqual(len(node_info.gpus), 16)
        self.assertTrue(node_info.gpus[0].pci_bus_id.startswith("00000000:"))
        # Checking memory conversion logic (31000 MB / 1024)
        self.assertAlmostEqual(node_info.gpus[0].available_mem_gb, 30.27, places=2)
        
        # Verify that static fields were preserved
        self.assertEqual(node_info.gpus[0].theoretical_tflops, 200.0)

if __name__ == "__main__":
    unittest.main()