import time
from typing import Dict, Any, List
from core.common.protocol import JanusMessage, ActionCode, StatusCode, JanusFuture
from core.common.cluster_context import NodeInfo
from core.master.orchestrator.messenger import MasterMessenger
from core.master.orchestrator.dim_future import DimFuture


class JanusMasterCollective:
    """
    High-level cluster coordination layer built on top of MasterMessenger.
    Provides:
        - Asynchronous task submission (submit, submit_to)
        - Result aggregation with quorum and failure reporting (gather_reports)
        - Global barrier synchronization
        - Automatic cluster context update from probe responses
    """

    def __init__(self, messenger: MasterMessenger):
        self.messenger = messenger

    # ------------------------------------------------------------------
    # Task Submission
    # ------------------------------------------------------------------

    def submit(self, action: ActionCode, payload: dict = None) -> JanusFuture:
        """
        Submit a task to all slaves. Returns a Future immediately.
        """
        msg = JanusMessage(
            source_rank=0,
            target="ALL",
            action=action,
            payload=payload or {}
        )
        self.messenger.broadcast(msg)

        return JanusFuture(
            request_id=msg.request_id,
            action=action,
            collective=self,
            expected_count=self.messenger.expected_slaves
        )

    def submit_to(self, rank: int, action: ActionCode, payload: dict = None) -> JanusFuture:
        """
        Submit a task to a specific slave. Returns a Future immediately.
        """
        msg = JanusMessage(
            source_rank=0,
            target=str(rank),
            action=action,
            payload=payload or {}
        )
        self.messenger.send_to_rank(rank, msg)

        return JanusFuture(
            request_id=msg.request_id,
            action=action,
            collective=self,
            expected_count=1
        )

    # ------------------------------------------------------------------
    # Result Aggregation
    # ------------------------------------------------------------------

    def gather_reports(self, action: ActionCode, request_id: str,
                       min_success_ratio: float = 1.0,
                       timeout: float = 60.0) -> Dict[str, Any]:
        """
        Aggregate responses for a given request_id, compute quorum,
        and update cluster context.
        """
        expected_count = self.messenger.expected_slaves
        min_expected = int(expected_count * min_success_ratio)

        # Use MasterMessenger.collect_responses to obtain raw responses
        # Note: collect_responses expects a list of expected ranks.
        # Since we want all known ranks, we fetch a snapshot under lock.
        with self.messenger.conn_lock:
            all_ranks = list(self.messenger.all_known_ranks)

        responses = self.messenger.collect_responses(request_id, all_ranks, timeout)

        # Classify responses
        success_nodes = [r for r in responses.values() if r.status == StatusCode.SUCCESS]
        failed_nodes = [r for r in responses.values() if r.status == StatusCode.ERROR]
        responded_ranks = set(responses.keys())
        missing_ranks = [r for r in all_ranks if r not in responded_ranks]
        is_quorum_met = len(success_nodes) >= min_expected

        # Update cluster context with all received reports
        for report in responses.values():
            self._update_cluster_context(report)

        return {
            "request_id": request_id,
            "success": success_nodes,
            "failed": failed_nodes,
            "missing": missing_ranks,
            "quorum_met": is_quorum_met,
            "status": "COMPLETED" if is_quorum_met else "FAILED",
            "completion_rate": len(success_nodes) / expected_count if expected_count > 0 else 0
        }

    # ------------------------------------------------------------------
    # Barrier
    # ------------------------------------------------------------------

    def barrier(self, timeout: float = 30.0):
        """
        Global barrier: wait for all slaves to acknowledge a BARRIER request.
        """
        future = self.submit(ActionCode.BARRIER)
        result = future.result(timeout=timeout, min_success_ratio=1.0)
        if not result["quorum_met"]:
            raise RuntimeError(f"Global barrier failed. Missing nodes: {result['missing']}")

    # ------------------------------------------------------------------
    # Cluster Context Updates
    # ------------------------------------------------------------------

    def _update_cluster_context(self, msg: JanusMessage):
        """
        Entry point for synchronizing probe results into ClusterContext.
        """
        if msg.status != StatusCode.SUCCESS:
            print(f"[Error] Action {msg.action} failed on Rank {msg.source_rank}: {msg.error_msg}")
            return

        target_node = self.messenger.cluster.nodes.get(msg.source_rank)
        if not target_node:
            return

        if msg.action == ActionCode.PROBE_ENV:
            self._sync_env_metrics(target_node, msg.payload)
        elif msg.action == ActionCode.PROBE_COMPUTE:
            self._sync_compute_metrics(target_node, msg.payload)
        elif msg.action == ActionCode.PROBE_NET_INTRA:
            self._sync_intra_net_metrics(target_node, msg.payload)
        elif msg.action in [ActionCode.PROBE_NET_INTRA, ActionCode.PROBE_NET_INTER]:
            self._sync_network_performance(msg.payload)

    def _sync_env_metrics(self, node: NodeInfo, data: Dict[str, Any]):
        """
        Update node with hardware environment and static topology.
        """
        # Node level: system specs
        node.hostname = data.get("hostname", node.hostname)
        node.ip = data.get("ip", node.ip)
        node.cpu_cores = data.get("cpu_cores", node.cpu_cores)
        node.sys_mem_gb = data.get("sys_mem_gb", node.sys_mem_gb)
        node.nic_type = data.get("nic_type", node.nic_type)
        node.has_rdma = data.get("has_rdma", node.has_rdma)

        # Topology (if present)
        if "topology" in data and data["topology"]:
            remote_topo = data["topology"]
            if node.topology is None:
                from core.common.cluster_context import NodeTopology
                node.topology = NodeTopology()
            node.topology.gpu_gpu_dist = remote_topo.get("gpu_gpu_dist", node.topology.gpu_gpu_dist)
            node.topology.gpu_nic_dist = remote_topo.get("gpu_nic_dist", node.topology.gpu_nic_dist)
            node.topology.cpu_affinity = remote_topo.get("cpu_affinity", node.topology.cpu_affinity)
            node.topology.numa_affinity = remote_topo.get("numa_affinity", node.topology.numa_affinity)

        # GPU details
        remote_gpus = data.get("gpus", [])
        for r_gpu in remote_gpus:
            l_gpu = next((g for g in node.gpus if g.local_id == r_gpu['local_id']), None)
            if l_gpu:
                l_gpu.pci_bus_id = r_gpu.get("pci_bus_id", l_gpu.pci_bus_id)
                l_gpu.memory_capacity_gb = r_gpu.get("memory_capacity_gb", l_gpu.memory_capacity_gb)
                l_gpu.available_mem_gb = r_gpu.get("available_mem_gb", l_gpu.available_mem_gb)
                l_gpu.temperature_celsius = r_gpu.get("temperature_celsius", l_gpu.temperature_celsius)

    def _sync_compute_metrics(self, node: NodeInfo, data: Dict[str, Any]):
        """
        Update node with compute performance metrics.
        """
        remote_gpus = data.get("gpus", [])
        for r_gpu in remote_gpus:
            l_gpu = next((g for g in node.gpus if g.local_id == r_gpu['local_id']), None)
            if l_gpu:
                l_gpu.peak_gemm_tflops = r_gpu.get("peak_gemm_tflops", l_gpu.peak_gemm_tflops)
                l_gpu.best_matmul_size = r_gpu.get("best_matmul_size", l_gpu.best_matmul_size)
                l_gpu.memory_bandwidth_gbps = r_gpu.get("memory_bandwidth_gbps", l_gpu.memory_bandwidth_gbps)
                l_gpu.gemm_efficiency = r_gpu.get("gemm_efficiency", l_gpu.gemm_efficiency)
                l_gpu.ridge_point_flops_per_byte = r_gpu.get("ridge_point_flops_per_byte", l_gpu.ridge_point_flops_per_byte)

    def _sync_network_performance(self, data: Dict[str, Any]):
        """
        Core logic: Write back performance results to the strategy_matrix based on task_id and strategy metadata.
        """
        task_id = data.get("task_id")
        dim = data.get("dim")           # 'tp', 'dp' or 'pp'
        strategy_raw = data.get("strategy") # might be [tp, pp, dp]
        perf_dict = data.get("perf")

        if not all([dim, strategy_raw, perf_dict]):
            print(f"[Warning] Incomplete network performance data for task {task_id}")
            return

        # 1. Convert strategy to Tuple to match the Key (tp, pp, dp) of strategy_matrix
        strategy_key = tuple(strategy_raw) if isinstance(strategy_raw, list) else strategy_raw
        
        # 2. Retrieve the corresponding ParallelStrategyPerformance object
        # Note: This object should have been created by init_parallel_strategy before dispatching the probing task
        strategy_perf = self.messenger.cluster.strategy_matrix.get(strategy_key)
        
        if not strategy_perf:
            print(f"[Error] Strategy {strategy_key} not found in matrix for task {task_id}")
            return

        # 3. Restore Dict to CommPerformance dataclass object
        from core.common.cluster_context import CommPerformance
        new_perf = CommPerformance(**perf_dict)

        # 4. Update the strategy matrix
        # communication_performance structure: { "tp": { (group_tuple): CommPerformance } }
        # The group key was not returned by the Slave side as a full group_tuple (to save bandwidth)
        # But since it is a collective, all ranks in the same group measure the same group's performance.
        # We adopt an overwriting update strategy here.
        if dim in strategy_perf.communication_performance:
            # Find the corresponding group under this dimension
            # Because a node may belong to multiple groups (though in the current probing logic usually one task corresponds to one group)
            # We directly update all group sampling points under this dimension, or precisely update by group_idx
            group_idx = data.get("group_idx")
            groups = strategy_perf.communication_groups.get(dim, [])
            
            if group_idx is not None and group_idx < len(groups):
                group_key = tuple(groups[group_idx])
                strategy_perf.communication_performance[dim][group_key] = new_perf
                
                print(f"[Master] Strategy {strategy_key} | {dim.upper()} Group {group_idx} updated: "
                      f"BW={new_perf.bandwidth_gbps}Gbps, Latency={new_perf.latency_us}us")
            else:
                # If no group_idx is provided, update as a global reference value for this dimension
                print(f"[Warning] No group_idx provided for task {task_id}, skipping precise update.")

    

    