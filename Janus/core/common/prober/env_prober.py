import os
import json
import subprocess
import shutil
import psutil
import socket
import re
from typing import List, Dict, Any, Optional, Tuple
from core.common.cluster_context import GPUInfo, NodeInfo

class EnvProber:
    """
    Comprehensive environmental prober for Slave nodes. 
    Focused on static hardware specifications and system environment.
    """
    def __init__(self, node_info: NodeInfo):
        # Directly modify the NodeInfo object in-place
        self.node_info = node_info
        
        self.vendor_patterns = {
            "NVIDIA": ["A100", "H100", "V100", "RTX", "A800", "H800"],
            "TianShu": ["TianGai", "BI-", "MR-", "V150", "IX"],
            "Huawei": ["Ascend", "910", "310", "NPU"],
            "AMD": ["MI200", "MI300", "Instinct"]
        }

    def _get_vendor(self, gpu_type: str) -> str:
        """Determines the hardware vendor from the GPU type string."""
        for vendor, keywords in self.vendor_patterns.items():
            if any(key.lower() in gpu_type.lower() for key in keywords):
                return vendor
        return "Unknown"

    def probe(self) -> NodeInfo:
        """
        Collects local node and GPU information. 
        Topology parsing is removed as per the strategy-driven performance model.
        """
        # 1. Update basic system specs (CPU/Mem/Net)
        self._update_system_info()

        if not self.node_info.gpus:
            print("[EnvProber] WARNING: No GPU information pre-initialized. Skipping GPU probe.")
            return self.node_info

        # 2. Determine vendor based on the first GPU
        vendor = self._get_vendor(self.node_info.gpus[0].type)
        
        # 3. Vendor-specific spec updates (VRAM, Bus ID, Temperature)
        if vendor == "NVIDIA":
            self._update_nvidia()
        elif vendor == "TianShu":
            self._update_tianshu()
        elif vendor == "Huawei":
            self._update_huawei()
        elif vendor == "AMD":
            self._update_amd()
        else:
            print(f"[EnvProber] WARNING: Unknown vendor '{vendor}'. GPU metrics might be incomplete.")

        return self.node_info

    def _update_nvidia(self):
        """Updates GPUInfo using nvidia-smi."""
        if not shutil.which("nvidia-smi"): return 
        try:
            query = "pci.bus_id,memory.total,memory.free,temperature.gpu"
            cmd = f"nvidia-smi --query-gpu={query} --format=csv,noheader,nounits"
            output = subprocess.check_output(cmd.split(), encoding='utf-8').strip()
            lines = output.split('\n')
            
            for i, gpu in enumerate(self.node_info.gpus):
                if i < len(lines):
                    p = [x.strip() for x in lines[i].split(',')]
                    gpu.pci_bus_id = p[0]
                    gpu.memory_capacity_gb = round(float(p[1]) / 1024.0, 2)
                    # Note: available_mem is a runtime snapshot
                    gpu.available_mem_gb = round(float(p[2]) / 1024.0, 2)
        except Exception as e:
            print(f"[EnvProber] NVIDIA probe failed: {e}")

    def _update_tianshu(self):
        """Updates GPUInfo using ixsmi."""
        if not shutil.which("ixsmi"): return
        try:
            query = "pci.bus_id,memory.total,memory.free,temperature.gpu"
            cmd = f"ixsmi --query-gpu={query} --format=csv,noheader,nounits"
            output = subprocess.check_output(cmd.split(), encoding='utf-8').strip()
            lines = output.split('\n')
            
            for i, gpu in enumerate(self.node_info.gpus):
                if i < len(lines):
                    p = [x.strip() for x in lines[i].split(',')]
                    gpu.pci_bus_id = p[0]
                    gpu.memory_capacity_gb = round(float(p[1]) / 1024.0, 2)
                    gpu.available_mem_gb = round(float(p[2]) / 1024.0, 2)
        except Exception as e:
            print(f"[EnvProber] TianShu probe failed: {e}")

    def _update_huawei(self):
        """Updates GPUInfo using npu-smi."""
        if not shutil.which("npu-smi"): return
        for i, gpu in enumerate(self.node_info.gpus):
            try:
                out = subprocess.check_output(["npu-smi", "info", "-i", str(i)], encoding='utf-8')
                mem_pattern = r"Memory Usage.*?(\d+)\s*/\s*(\d+)\s*MB"
                mem_match = re.search(mem_pattern, out)
                if mem_match:
                    total_mb = float(mem_match.group(2))
                    gpu.memory_capacity_gb = round(total_mb / 1024.0, 2)
                    gpu.available_mem_gb = round((total_mb - float(mem_match.group(1))) / 1024.0, 2)
                gpu.pci_bus_id = f"NPU:{i}"
            except Exception as e:
                print(f"[EnvProber] Huawei NPU:{i} probe failed: {e}")

    def _update_amd(self):
        """Updates GPUInfo using rocm-smi."""
        if not shutil.which("rocm-smi"): return
        try:
            out = subprocess.check_output(["rocm-smi", "-a", "--json"], encoding='utf-8')
            data = json.loads(out)
            for i, gpu in enumerate(self.node_info.gpus):
                card_key = f"card{i}"
                if card_key in data:
                    metrics = data[card_key]
                    vram_total = float(metrics.get("VRAM Total Memory (B)", 0))
                    vram_used = float(metrics.get("VRAM Used Memory (B)", 0))
                    gpu.pci_bus_id = metrics.get("PCI Bus", "N/A")
                    gpu.memory_capacity_gb = round(vram_total / (1024**3), 2)
                    gpu.available_mem_gb = round((vram_total - vram_used) / (1024**3), 2)
        except Exception as e:
            print(f"[EnvProber] AMD probe failed: {e}")

    def _update_system_info(self):
        """Updates NodeInfo with CPU, Memory, and RDMA capability."""
        # 1. CPU & Memory
        try:
            self.node_info.cpu_cores = psutil.cpu_count(logical=False)
            self.node_info.sys_mem_gb = round(psutil.virtual_memory().total / (1024**3), 2)
        except Exception:
            pass

        # 2. Network Probing (RDMA focus)
        self.node_info.has_rdma = False
        self.node_info.nic_type = "Ethernet"

        # Check for InfiniBand/RoCE devices
        if shutil.which("ibv_devinfo"):
            try:
                out = subprocess.check_output(["ibv_devinfo"], encoding='utf-8', errors='ignore')
                if "PORT_ACTIVE" in out:
                    self.node_info.has_rdma = True
                    self.node_info.nic_type = "IB" if "transport: InfiniBand" in out else "RoCE"
                    return 
            except Exception:
                pass 

        # Fallback: Kernel modules check
        try:
            with open("/proc/modules", "r") as f:
                lsmod_out = f.read()
            rdma_modules = ["ib_core", "mlx5_ib", "hns_roce", "siw"]
            if any(m in lsmod_out for m in rdma_modules):
                self.node_info.has_rdma = True
                self.node_info.nic_type = "RoCE"
        except Exception:
            pass