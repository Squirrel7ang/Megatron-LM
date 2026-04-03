import time
from typing import List
from core.master.parser.cluster_parser import ClusterParser
from core.master.parser.model_parser import ModelParser
from core.master.orchestrator.messenger import MasterMessenger
from core.master.orchestrator.collective import JanusMasterCollective
from core.common.protocol import ActionCode


class JanusOrchestrator:
    """
    Top-level coordinator for the Janus master process.

    Responsibilities:
        - Parse model and cluster configuration files.
        - Initialise MasterMessenger and JanusMasterCollective.
        - Wait for all slave processes to handshake (including slave-0 which
          runs on the same physical machine as the master, keeping the two
          roles decoupled at the process level).
        - Drive the sequential probe pipeline:
              PROBE_ENV  →  PROBE_COMPUTE
          Network probing (PROBE_NET_INTRA / PROBE_NET_INTER) is intentionally
          omitted for now and will be added in a follow-up.

    Design notes on expected_slaves:
        We set expected_slaves = total_nodes (not total_nodes - 1).
        Slave-0 is a regular slave process that happens to share a host with
        the master; it connects via the same TCP handshake path as every other
        slave.  This keeps master logic uniform and avoids special-casing the
        co-located node.
    """

    # How long to wait for all slaves to complete their handshake before
    # declaring the cluster unreachable.
    SLAVE_CONNECT_TIMEOUT_S: float = 120.0

    # Per-probe timeout: env probe is lightweight; compute probe runs
    # benchmarks so needs more headroom.
    PROBE_ENV_TIMEOUT_S: float = 60.0
    PROBE_COMPUTE_TIMEOUT_S: float = 180.0
    PROBE_NETWORK_TIMEOUT_S: float = 180.0

    # Poll interval used while waiting for all slaves to come online.
    _CONNECT_POLL_INTERVAL_S: float = 1.0
    GLOBAL_WORLD_PORT: int = 29500

    def __init__(self, model_config_path: str, cluster_config_path: str):
        self.model_config_path   = model_config_path
        self.cluster_config_path = cluster_config_path

        # Parse configuration files.
        self.model_parser   = ModelParser(model_config_path)
        self.cluster_parser = ClusterParser(cluster_config_path)

        # Note: use self.* here — original code accidentally referenced the
        # bare local names model_parser / cluster_parser which would raise
        # NameError at construction time.
        self.model_context   = self.model_parser.parse()
        self.cluster_context = self.cluster_parser.parse()

        # Network identity.
        # cluster_context.master_addr is the bind address (e.g. "0.0.0.0" or a
        # specific IP); cluster_context.master_port is the TCP port number.
        # The original skeleton had these swapped (master_host used for port);
        # corrected here to match the field semantics implied by MasterMessenger.
        self.host = self.cluster_context.master_addr
        self.port = self.cluster_context.master_port

        # expected_slaves == total_nodes: slave-0 runs on the master machine
        # but is still a distinct process that goes through the normal handshake.
        self.expected_slaves = self.cluster_context.total_nodes

        # Communication layer.
        self.messenger  = MasterMessenger(
            self.host, self.port,
            self.expected_slaves, self.cluster_context,
        )
        self.collective = JanusMasterCollective(self.messenger)

        # Robust port allocation
        self._port_counter = 0  
        self.BASE_PORT = 29501

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self):
        """
        Full orchestration pipeline.  Call this after constructing the object.

        Steps
        -----
        1. Start the messenger (opens listening socket, launches polling thread).
        2. Wait until all slaves have completed their TCP handshake.
        3. Run PROBE_ENV on all slaves and merge results into cluster_context.
        4. Run PROBE_COMPUTE on all slaves and merge results into cluster_context.
        5. (Network probing — to be added later.)
        """
        self.messenger.start()

        try:
            self._wait_for_all_slaves()
            self._create_group()
            self._probe_env()
            self._probe_compute()
            self._probe_network()
        finally:
            # Ensure the messenger is cleanly shut down even if a probe fails.
            self.messenger.close()

    # ------------------------------------------------------------------
    # Connection phase
    # ------------------------------------------------------------------

    def _wait_for_all_slaves(self):
        print(f"[Orchestrator] Waiting for {self.expected_slaves} slaves...")

        is_ready = self.messenger.wait_until_ready(self.SLAVE_CONNECT_TIMEOUT_S)

        if not is_ready:
            missing = self.messenger.get_missing_ranks()
            raise RuntimeError(
                f"[Orchestrator] Slave connection timeout. Missing ranks: {missing}"
            )
        
        print("[Orchestrator] Cluster is ready for probing.")

    def _create_group(self):
        """
        Broadcast CREATE_GROUP to all slaves to initialize a persistent 
        global Process Group (WORLD).
        
        This rendezvous allows all GPUs in the cluster to see each other 
        once and maintain a long-lived NCCL communicator.
        """
        print(f"[Orchestrator] Phase 0.5 — Broadcasting CREATE_GROUP (Global WORLD)...")
        
        # We use the Orchestrator's host IP as the master address for NCCL.
        # total_gpus is the sum of all GPUs across all expected slaves.
        payload = {
            "master_addr": self.host, 
            "master_port": self.GLOBAL_WORLD_PORT,
            "world_size": self.cluster_context.total_gpus,
            "backend": "nccl"
        }

        # Broadcast to ALL slaves. 
        # Every slave will receive this and trigger its internal worker pool.
        future = self.collective.submit(ActionCode.CREATE_GROUP, payload=payload)

        # This is a critical sync point. If one node fails to join the WORLD,
        # all subsequent network probes will likely hang or fail.
        result = future.result(
            timeout=self.PROBE_NETWORK_TIMEOUT_S,
            min_success_ratio=1.0, # Strict: Every node must join
        )

        self._assert_quorum(result, phase="CREATE_GROUP")
        print("[Orchestrator] Global Process Group (WORLD) established across all nodes.")

    # ------------------------------------------------------------------
    # Probe phases
    # ------------------------------------------------------------------

    def _probe_env(self):
        """
        Broadcast PROBE_ENV to all slaves and wait for responses.

        Each slave collects static hardware information (CPU, memory, GPU specs,
        NIC type, NUMA topology) and returns it as a payload.  Results are merged
        into cluster_context by JanusMasterCollective._update_cluster_context via
        gather_reports.

        Raises RuntimeError if quorum is not met (i.e. at least one slave did
        not respond successfully within the timeout).
        """
        print("[Orchestrator] Phase 1 — broadcasting PROBE_ENV ...")
        future = self.collective.submit(ActionCode.PROBE_ENV)

        result = future.result(
            timeout=self.PROBE_ENV_TIMEOUT_S,
            min_success_ratio=1.0,
        )

        self._assert_quorum(result, phase="PROBE_ENV")
        print(f"[Orchestrator] PROBE_ENV complete. "
              f"Success: {len(result['success'])}, "
              f"Failed: {len(result['failed'])}, "
              f"Missing: {result['missing']}")

    def _probe_compute(self):
        """
        Broadcast PROBE_COMPUTE to all slaves and wait for responses.

        Each slave runs GEMM / memory-bandwidth micro-benchmarks on every local
        GPU and returns peak TFLOPS, memory bandwidth, and GEMM efficiency.
        Results are merged into cluster_context via gather_reports.

        PROBE_COMPUTE is submitted only after PROBE_ENV has succeeded so that
        the cluster context already holds accurate hardware metadata (GPU count,
        memory capacity, etc.) before compute benchmarks run.

        Raises RuntimeError if quorum is not met.
        """
        print("[Orchestrator] Phase 2 — broadcasting PROBE_COMPUTE ...")
        future = self.collective.submit(ActionCode.PROBE_COMPUTE)

        result = future.result(
            timeout=self.PROBE_COMPUTE_TIMEOUT_S,
            min_success_ratio=1.0,
        )

        self._assert_quorum(result, phase="PROBE_COMPUTE")
        print(f"[Orchestrator] PROBE_COMPUTE complete. "
              f"Success: {len(result['success'])}, "
              f"Failed: {len(result['failed'])}, "
              f"Missing: {result['missing']}")

    def _probe_network(self):
        """
        Phase 3: Network Probing with Topology-Aware Scheduling.
        - Intra-node groups: Maximum parallelism.
        - Inter-node groups: Conflict-aware concurrency (avoids node-level congestion).
        """
        print("[Orchestrator] Starting Topology-Aware Network Probing...")

        for strategy in self.cluster_context.get_plausible_strategies():
            tp, dp, pp = strategy
            self.cluster_context.init_parallel_strategy(tp, dp, pp)
            strategy_matrix = self.cluster_context.strategy_matrix[strategy]

            for dim in ["tp", "dp", "pp"]:
                dim_size = {"tp": tp, "dp": dp, "pp": pp}[dim]
                if dim_size <= 1: continue

                # Get the sampling groups for this dimension
                sample_groups = self.cluster_context.get_strategic_samples(
                    strategy_matrix.communication_groups[dim], dim
                )
                if not sample_groups: continue

                # --- 1. Classification & Bucketing ---
                intra_tasks = [] # (group_idx, group)
                inter_tasks = [] # (group_idx, group)
                for idx, group in enumerate(sample_groups):
                    if self._is_intra_group(group):
                        intra_tasks.append((idx, group))
                    else:
                        inter_tasks.append((idx, group))

                print(f"[Orchestrator] Strategy {strategy} Dim {dim.upper()}: "
                      f"{len(intra_tasks)} Intra, {len(inter_tasks)} Inter groups.")

                # --- 2. Dispatch Intra-node Tasks (Full Parallel) ---
                # These tasks don't compete for cross-node RDMA bandwidth
                intra_futures = []
                for idx, group in intra_tasks:
                    task_id = f"{strategy}-{dim}-intra-{idx}"
                    futures = self._dispatch_group(dim, strategy, idx, group, task_id, is_intra=True)
                    intra_futures.extend(futures)
                
                # Wait for all intra tasks in this dimension
                for fut in intra_futures:
                    fut.result(timeout=self.PROBE_NETWORK_TIMEOUT_S)

                # --- 3. Dispatch Inter-node Tasks (Conflict-Aware Parallel) ---
                # Using a 'Greedy Conflict-Free' scheduling window
                running_futures = []
                active_nodes = set()

                for idx, group in inter_tasks:
                    involved_nodes = set(self.cluster_context.rank_to_node_map[r] for r in group)
                    
                    # If conflict detected (node already busy), drain current pool
                    if involved_nodes & active_nodes:
                        for fut in running_futures:
                            fut.result(timeout=self.PROBE_NETWORK_TIMEOUT_S)
                        running_futures.clear()
                        active_nodes.clear()

                    # Dispatch new group
                    task_id = f"{strategy}-{dim}-inter-{idx}"
                    group_futs = self._dispatch_group(dim, strategy, idx, group, task_id, is_intra=False)
                    running_futures.extend(group_futs)
                    active_nodes.update(involved_nodes)

                # Final drain for the last batch of inter tasks
                for fut in running_futures:
                    fut.result(timeout=self.PROBE_NETWORK_TIMEOUT_S)

                # --- 4. Global Barrier & Cleanup ---
                # Ensure all slaves have released NCCL communicators and finished IO
                self.collective.barrier(timeout=self.PROBE_NETWORK_TIMEOUT_S)
                print(f"[Orchestrator] Completed Dimension: {dim.upper()}")

    def _dispatch_group(self, dim, strategy, group_idx, group, task_id, is_intra) -> List:
        """Internal helper to dispatch task to all nodes in a communication group."""
        involved_nodes = sorted(list(set(self.cluster_context.rank_to_node_map[r] for r in group)))
        
        master_addr, master_port = None, None
        if not is_intra:
            # Elect a master node for this group's rendezvous
            master_node_id = involved_nodes[0]
            master_addr = self.cluster_context.nodes[master_node_id].ip
            master_port = self._get_next_port()

        node_futures = []
        for node_id in involved_nodes:
            local_ranks = [self.cluster_context.rank_to_local_map[r] for r in group 
                           if self.cluster_context.rank_to_node_map[r] == node_id]
            
            payload = {
                "dim": dim,
                "strategy": strategy,
                "group_idx": group_idx,
                "global_ranks": group,
                "local_ranks": local_ranks,
                "task_id": task_id,
                "master_addr": master_addr,
                "master_port": master_port,
                "action": ActionCode.PROBE_NET_INTRA if is_intra else ActionCode.PROBE_NET_INTER
            }
            # The Collective will handle task identification via task_id internally
            node_futures.append(self.collective.submit_to(node_id, payload["action"], payload))
        
        return node_futures

    def _is_intra_group(self, group: List[int]) -> bool:
        return len(set(self.cluster_context.rank_to_node_map[r] for r in group)) == 1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assert_quorum(self, result: dict, phase: str):
        """
        Raise RuntimeError if the gather_reports result indicates quorum was
        not met, providing a clear diagnostic message with the failing ranks.
        """
        if not result["quorum_met"]:
            raise RuntimeError(
                f"[Orchestrator] {phase} failed to reach quorum. "
                f"Failed ranks: {[m.source_rank for m in result['failed']]}, "
                f"Missing ranks: {result['missing']}"
            )

    def _get_next_port(self) -> int:
        """Atomic-like port allocation to prevent address collision."""
        port = self.BASE_PORT + self._port_counter
        self._port_counter = (self._port_counter + 1) % 5000 # Wrap around
        return port