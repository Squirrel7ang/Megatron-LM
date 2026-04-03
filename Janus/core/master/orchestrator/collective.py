import time
import threading
from typing import Dict, Any, List, Set
from core.common.protocol import JanusMessage, ActionCode, StatusCode, JanusFuture
from core.common.cluster_context import NodeInfo, CommPerformance, CollectiveType
from core.master.orchestrator.messenger import MasterMessenger

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
        # Initialize a thread lock for synchronized access
        self.sync_lock = threading.Lock()
        
        # Track processed task IDs to ensure idempotency across nodes
        self.processed_task_ids: Set[str] = set()

    # ------------------------------------------------------------------
    # Task Submission
    # ------------------------------------------------------------------

    def submit(self, action: ActionCode, payload: dict = None) -> JanusFuture:
        """
        Submit a task to all slaves. Returns a Future immediately.
        """
        msg = JanusMessage(
            source_rank=-1,
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
            source_rank=-1,
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
            print(f"[Error] Action {msg.action} failed on Rank {msg.source_rank}: {msg.payload.get('error', 'Unknown')}")
            return

        # Map source_rank to target_node
        # For simplicity, we use the rank as index, but with a fallback for single-node tests
        node_idx = msg.source_rank
        if node_idx >= len(self.messenger.cluster.nodes):
            if len(self.messenger.cluster.nodes) == 1:
                node_idx = 0
            else:
                print(f"[Error] Rank {msg.source_rank} out of bounds for cluster nodes")
                return
        target_node = self.messenger.cluster.nodes[node_idx]

        with self.sync_lock:
            # Handle network probe deduplication
            if msg.action in [ActionCode.PROBE_NET_INTRA, ActionCode.PROBE_NET_INTER]:
                data = msg.payload
                task_id = data.get("task_id")
                if not task_id:
                    return
                
                if task_id in self.processed_task_ids:
                    # Drop redundant reports from other nodes in the same group
                    return
                
                # Mark task as completed before writing to the matrix
                self.processed_task_ids.add(task_id)

        if msg.action == ActionCode.PROBE_ENV:
            self._sync_env_metrics(target_node, msg.payload)
        elif msg.action == ActionCode.PROBE_COMPUTE:
            self._sync_compute_metrics(target_node, msg.payload)
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
        Synchronizes network probe results into the cluster context.
        Updates 'all_case' for full profiling and 'the_worst_case'/'dimension_performance' 
        for bottleneck tracking using a bandwidth-first logic.
        """
        dim = data.get("dim")           # 'tp', 'dp', 'pp'
        strategy_raw = data.get("strategy")
        group_list = data.get("group")
        perf_dict = data.get("perf")
        
        if not all([dim, strategy_raw, group_list, perf_dict]):
            return

        # 1. Convert to hashable types
        strategy_key = tuple(strategy_raw)
        group_key = tuple(group_list)
        
        strategy_perf = self.messenger.cluster.strategy_matrix.get(strategy_key)
        if not strategy_perf:
            return
        
        # Determine protocol: PP is typically P2P, others are Collective
        coll_type = CollectiveType.P2P if dim == "pp" else CollectiveType.ALL_REDUCE

        # 2. Restore performance object from payload
        incoming_perf = CommPerformance(**perf_dict)

        # 3. Ensure dictionary hierarchy for the dimension exists in all stores
        for store in [strategy_perf.all_case, 
                      strategy_perf.the_worst_case, 
                      strategy_perf.dimension_performance]:
            if dim not in store:
                store[dim] = {}

        # ---------------------------------------------------------
        # Logic A: Persistence (Update all_case)
        # ---------------------------------------------------------
        # Map: dim -> group_ranks -> coll_type -> performance
        if group_key not in strategy_perf.all_case[dim]:
            strategy_perf.all_case[dim][group_key] = {}

        import copy
        strategy_perf.dimension_performance[dim][coll_type] = copy.deepcopy(incoming_perf)

        # ---------------------------------------------------------
        # Logic B: Bottleneck Tracking (Update the_worst_case & dimension_performance)
        # ---------------------------------------------------------
        # Use .get() to safely retrieve current worst performance
        current_worst_perf = strategy_perf.dimension_performance[dim].get(coll_type)

        # Straggler Decision: Bandwidth-First
        is_new_straggler = (
            current_worst_perf is None or 
            incoming_perf.bandwidth_gbps < current_worst_perf.bandwidth_gbps
        )

        if is_new_straggler:
            # Record "How much" it sucks (for Cost Model)
            strategy_perf.dimension_performance[dim][coll_type] = incoming_perf
            # Record "Who" sucks (for Profiling/Root Cause)
            strategy_perf.the_worst_case[dim][coll_type] = group_key
            
            print(f"[Master] NEW Bottleneck for {dim.upper()} strategy {strategy_key}: "
                  f"Group {group_key} @ {incoming_perf.bandwidth_gbps} Gbps")
        else:
            # Use .get() for the log to prevent KeyError if this is the first sample 
            # and it's somehow not considered a straggler (though None case handles first)
            current_worst_group = strategy_perf.the_worst_case[dim].get(coll_type, group_key)
            print(f"[Master] Group {group_key} logged. Current bottleneck remains "
                  f"{current_worst_group}")

    

    