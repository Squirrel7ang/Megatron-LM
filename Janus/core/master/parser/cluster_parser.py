import json
from typing import Set, List
from core.common.cluster_context import ClusterContext, NodeInfo, GPUInfo

class ClusterParser:
    """
    Parses cluster_spec.json, references the hardware library, 
    and constructs a ClusterContext with vendor-level validation.
    """
    def __init__(self, config_path: str):
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.raw_data = json.load(f)
        
        # Reference the library for hardware specs
        self.hw_lib = self.raw_data.get("hardware_library", {})
        if not self.hw_lib:
            raise ValueError("[Cluster Error] 'hardware_library' is missing in config.")

        # Define vendor keywords to map GPU types to specific vendors
        # This ensures driver and communication library compatibility within a node
        self.vendor_patterns = {
            "NVIDIA": ["A100", "H100", "V100", "RTX", "A800", "H800", "L40"],
            "TianShu": ["TianGai", "BI-", "MR-", "V150", "智铠"],
            "Huawei": ["Ascend", "910", "310", "Atlas"],
            "AMD": ["MI200", "MI300", "Radeon", "Instinct"]
        }

    def _get_vendor(self, gpu_type: str) -> str:
        """
        Helper method to identify vendor based on GPU type string.
        Returns 'Unknown' if no pattern matches.
        """
        for vendor, keywords in self.vendor_patterns.items():
            if any(key.lower() in gpu_type.lower() for key in keywords):
                return vendor
        return "Unknown"

    def parse(self) -> ClusterContext:
        meta = self.raw_data["cluster_metadata"]
        
        # 1. Initialize top-level context
        # is_heterogeneous will be recalculated based on actual GPU types found
        cluster = ClusterContext(
            cluster_name=meta["name"],
            total_nodes=meta.get("total_nodes", 1),
            master_addr=meta["master_address"],
            master_port=meta["master_port"],
        )

        global_ids: Set[int] = set()
        all_gpu_types_in_cluster: Set[str] = set()

        # 2. Iterate through physical nodes defined in config
        for node_raw in self.raw_data.get("nodes", []):
            node = NodeInfo(
                node_id=node_raw["node_id"],
                hostname=node_raw["hostname"],
                ip=node_raw["ip"],
                nic_type=meta.get("network_type", "Ethernet")
            )

            gpus_raw = node_raw.get("gpus", [])
            if gpus_raw:
                # Use the first GPU in the node as the 'Vendor Anchor'
                first_gpu_type = gpus_raw[0]["type"]
                node_vendor = self._get_vendor(first_gpu_type)

                for gpu_raw in gpus_raw:
                    gpu_type = gpu_raw["type"]
                    
                    # Validation: Ensure the GPU type is defined in the hardware library
                    if gpu_type not in self.hw_lib:
                        raise ValueError(f"[Cluster Error] GPU type '{gpu_type}' not defined in hardware_library.")
                    
                    # Validation: Intra-node Vendor Consistency Check
                    # We allow different models (e.g., A100 & A800) but forbid mixing vendors (e.g., NVIDIA & Huawei)
                    current_vendor = self._get_vendor(gpu_type)
                    if current_vendor == "Unknown" or current_vendor != node_vendor:
                        raise ValueError(
                            f"[Cluster Error] Vendor Mismatch on Node {node.node_id} ({node.hostname}). "
                            f"Detected {current_vendor} ({gpu_type}) mixed with {node_vendor} ({first_gpu_type}). "
                            f"Mixed-vendor nodes are not supported due to driver conflicts."
                        )

                    # Validation: Prevent global_id collisions across the cluster
                    gid = gpu_raw["global_id"]
                    if gid in global_ids:
                        raise ValueError(f"[Cluster Error] Duplicate global_id {gid} detected.")
                    global_ids.add(gid)

                    # Fetch static specs from hardware library and create GPUInfo
                    spec = self.hw_lib[gpu_type]
                    gpu = GPUInfo(
                        global_id=gid,
                        local_id=gpu_raw["local_id"],
                        pci_bus_id="",  # Placeholder: to be populated by Slave's EnvProber
                        type=gpu_type,
                        memory_capacity_gb=spec["memory_capacity_gb"],
                        theoretical_tflops=spec["theoretical_tflops"]
                    )
                    node.gpus.append(gpu)
                    all_gpu_types_in_cluster.add(gpu_type)
            
            cluster.nodes.append(node)

        # 3. Cluster-level Heterogeneity Inference
        # Automatically set is_heterogeneous if multiple GPU models exist in the cluster
        if len(all_gpu_types_in_cluster) > 1:
            cluster.is_heterogeneous = True
        else:
            # Fallback to metadata if only one type is found (manual override for edge cases)
            cluster.is_heterogeneous = meta.get("is_heterogeneous", False)

        # 4. Final topology sanity check
        if len(cluster.nodes) != cluster.total_nodes:
            raise ValueError(
                f"[Cluster Warning] Found {len(cluster.nodes)} nodes in config, "
                f"but cluster_metadata expected {cluster.total_nodes}."
            )

        return cluster