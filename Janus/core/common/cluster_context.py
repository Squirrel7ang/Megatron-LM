import re
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple, Any
import math

# --- 1. Communication Primitives and Performance Models ---

class CollectiveType(str, Enum):
    """Enumeration of collective communication primitives."""
    ALL_REDUCE = "all_reduce"
    ALL_GATHER = "all_gather"
    REDUCE_SCATTER = "reduce_scatter"
    P2P = "p2p"
    BROADCAST = "broadcast"


@dataclass
class CommPerformance:
    """
    Performance model parameters for a collective communication operation.

    Attributes:
        latency_us: The fixed latency (alpha) in microseconds.
        bandwidth_gbps: The effective bandwidth (beta) in Gbps.
        bus_bandwidth_gbps: The bus bandwidth after accounting for algorithmic factors.
    """
    latency_us: float = 0.0
    bandwidth_gbps: float = 0.0
    bus_bandwidth_gbps: float = 0.0


@dataclass
class ParallelStrategyPerformance:
    """
    Holds performance data and communication group definitions for a specific
    parallelization strategy (TP, DP, PP).

    Attributes:
        tp: Tensor parallelism degree.
        dp: Data parallelism degree.
        pp: Pipeline parallelism degree.
        communication_groups: Dictionary mapping group type ("tp", "dp", "pp")
                              to a list of rank lists.
        dimension_performance: tp/dp/pp -> Dict[CollectiveType, CommPerformance]
    """
    tp: int
    dp: int
    pp: int
    communication_groups: Dict[str, List[List[int]]] = field(default_factory=dict)
    dimension_performance: Dict[str, Dict[CollectiveType, CommPerformance]] = field(default_factory=dict)
    all_case: Dict[str, Dict[Tuple[int, ...], Dict[CollectiveType, CommPerformance]]] = field(default_factory=dict)
    the_worst_case: Dict[str, Dict[CollectiveType ,Tuple[int, ...]]] = field(default_factory=dict)

    def __repr__(self):
        return f"StrategyPerf(TP={self.tp}, DP={self.dp}, PP={self.pp})"


# --- 2. Hardware Unit Abstractions ---

@dataclass
class GPUInfo:
    """Information about a single GPU device."""
    global_id: int               # Rank in the global world
    local_id: int                # Rank within its node
    pci_bus_id: str              # PCI bus identifier
    type: str                    # GPU model (e.g., "A100")
    memory_capacity_gb: float
    memory_bandwidth_gbps: float = 0.0
    best_matmul_size: int = 0
    theoretical_tflops: float = 0.0
    peak_gemm_tflops: float = 0.0
    gemm_efficiency: float = 0.0
    ridge_point_flops_per_byte: float = 0.0
    available_mem_gb: float = 0.0
    temperature_celsius: float = 0.0


@dataclass
class NodeInfo:
    """Information about a physical node."""
    node_id: int
    hostname: str
    ip: str
    gpus: List[GPUInfo] = field(default_factory=list)
    cpu_cores: int = 0
    sys_mem_gb: float = 0.0
    nic_type: str = "RoCE"      # Network interface type
    has_rdma: bool = True       # Whether RDMA is supported


# --- 3. Master Cluster Context with Megatron Group Logic ---

@dataclass
class ClusterContext:
    """
    Centralized cluster state, including node descriptions and performance
    models for various parallelization strategies.

    This class also provides methods to generate orthogonal communication groups
    (TP, DP, PP) following the exact logic used in Megatron-LM.
    """
    cluster_name: str
    total_nodes: int
    master_addr: str
    master_port: int
    nodes: List[NodeInfo] = field(default_factory=list)
    env_fingerprint: str = ""
    strategy_matrix: Dict[Tuple[int, int, int], ParallelStrategyPerformance] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Dynamic Properties (Computed on-the-fly)
    # ------------------------------------------------------------------

    @property
    def gpus_per_node(self) -> int:
        """
        Dynamically infers gpus_per_node from the current nodes list.
        If nodes are not yet probed, returns 0.
        """
        if not self.nodes:
            return 0

        # Extract real-time GPU counts from each node
        counts = [len(node.gpus) for node in self.nodes]
        
        # If no GPUs are detected in any node yet
        if not counts or max(counts) == 0:
            return 0

        # Check homogeneity (ensure all nodes have the same number of GPUs)
        # Assuming validate_homogeneity is a helper method in this class
        if self.validate_homogeneity():
            return counts[0]
        else:
            # For heterogeneous clusters, use the minimum as a safe baseline
            min_count = min(counts)
            # print(f"[Warning] Heterogeneous cluster. Baseline GPUs: {min_count}")
            return min_count

    @property
    def min_gpu_mem_gb(self) -> float:
        """
        Dynamically calculates the minimum VRAM across all detected GPUs.
        """
        all_mems = [
            gpu.memory_capacity_gb 
            for node in self.nodes 
            for gpu in node.gpus
        ]
        return min(all_mems) if all_mems else 0.0

    @property
    def rank_to_node_map(self) -> Dict[int, int]:
        """
        Dynamically builds the global_rank -> node_id mapping.
        """
        mapping = {}
        for node in self.nodes:
            for gpu in node.gpus:
                mapping[gpu.global_id] = node.node_id
        return mapping

    @property
    def rank_to_local_map(self) -> Dict[int, int]:
        """
        Dynamically builds the global_rank -> local_rank mapping.
        """
        mapping = {}
        for node in self.nodes:
            for gpu in node.gpus:
                mapping[gpu.global_id] = gpu.local_id
        return mapping

    @property
    def total_gpus(self) -> int:
        """
        Total gpus in the cluster
        """
        gpu_cnt = 0
        for node in self.nodes:
            gpu_cnt += len(node.gpus)
        return gpu_cnt

    def get_global_rank(self, node_id: int, local_rank: int) -> int:
        """
        Translates a (node_id, local_rank) pair back into a cluster-wide global_rank.
        
        Args:
            node_id: The ID of the node (Slave.rank).
            local_rank: The local GPU index on that node (0-7).
            
        Returns:
            The global_rank assigned to this specific GPU.
            
        Raises:
            ValueError: If the combination of node_id and local_rank is not found.
        """
        # Iterate through the pre-built rank_to_node_map
        # We look for a rank that matches both the target node and the target local rank
        for g_rank, n_id in self.rank_to_node_map.items():
            if n_id == node_id and self.rank_to_local_map.get(g_rank) == local_rank:
                return g_rank
        
        raise ValueError(f"No global_rank found for Node {node_id} and Local Rank {local_rank}. "
                         f"Check if cluster initialization is complete.")

    def get_strategic_samples(self, groups: List[List[int]], parallel_type: str) -> List[List[int]]:
        """
        Multi-stage Priority Sampling Algorithm:
        1. Classifies groups into Cross-node and Intra-node pools.
        2. Prioritizes topologically unique Cross-node paths (RDMA bottlenecks).
        3. Prioritizes topologically unique Intra-node paths (local variance).
        4. Fallback fill: Relaxes deduplication to meet sample budget if needed.
        """
        if not groups:
            return []

        # Budget: Take the smaller value between total groups and total nodes
        max_samples = min(len(groups), len(self.nodes))
        
        samples = []
        # Use tuple representation of groups to track what has already been sampled
        sampled_group_tuples = set() 
        
        covered_node_pairs = set()   # Tracks unique node combinations (e.g., (Node0, Node1))
        covered_single_nodes = set() # Tracks unique single nodes

        cross_node_pool = [] # List of (group, node_tuple)
        intra_node_pool = [] # List of (group, node_id)

        # --- Stage 1: Classification ---
        for group in groups:
            involved_nodes = tuple(sorted(list(set(self.rank_to_node_map[r] for r in group))))
            if len(involved_nodes) > 1:
                cross_node_pool.append((group, involved_nodes))
            else:
                intra_node_pool.append((group, involved_nodes[0]))

        # --- Stage 2: Prioritize Unique Cross-node Paths ---
        for group, nodes in cross_node_pool:
            if nodes not in covered_node_pairs:
                samples.append(group)
                sampled_group_tuples.add(tuple(group))
                covered_node_pairs.add(nodes)
            
            if len(samples) >= max_samples:
                return samples

        # --- Stage 3: Prioritize Unique Intra-node Paths ---
        for group, node_id in intra_node_pool:
            if node_id not in covered_single_nodes:
                samples.append(group)
                sampled_group_tuples.add(tuple(group))
                covered_single_nodes.add(node_id)
            
            if len(samples) >= max_samples:
                return samples

        # --- Stage 4: Fallback Fill (Relax Deduplication constraints) ---
        # Fixes the edge case where topological duplication (e.g., 8 PP groups sharing 
        # identical node pairs across 2 nodes) leaves the sample budget unfulfilled.
        for group in groups:
            if tuple(group) not in sampled_group_tuples:
                samples.append(group)
                sampled_group_tuples.add(tuple(group))
            
            if len(samples) >= max_samples:
                break

        return samples

    # --- Megatron-style Rank Grouping Logic ---

    def _prefix_product(self, a: List[int], init: int = 1) -> List[int]:
        """
        Compute prefix products of a list.

        Args:
            a: List of integers.
            init: Initial value (usually 1).

        Returns:
            List of prefix products: [init, init*a[0], init*a[0]*a[1], ...].
        """
        r = [init]
        for v in a:
            init = init * v
            r.append(init)
        return r

    def _decompose(self, index: int, shape: List[int], stride: List[int]) -> List[int]:
        """
        Decompose a 1D index into N‑dimensional coordinates given a shape and stride.

        This is used to map a linear rank index to a multi‑dimensional coordinate
        in the parallelization space (TP, DP, PP).

        Args:
            index: The linear index to decompose.
            shape: The size of each dimension (e.g., [tp, dp, pp]).
            stride: The stride (offset) per dimension.

        Returns:
            List of coordinates, one per dimension.
        """
        idx = [(index // d) % s for s, d in zip(shape, stride)]
        return idx

    def _inner_product(self, a: List[int], b: List[int]) -> int:
        """Dot product of two lists."""
        return sum([x * y for x, y in zip(a, b)])

    def _generate_masked_orthogonal_rank_groups(
        self, world_size: int, parallel_size: List[int], mask: List[bool]
    ) -> List[List[int]]:
        """
        Core logic from Megatron‑LM to generate orthogonal rank groups.

        Given a 3‑dimensional parallel configuration (TP, DP, PP) and a mask that
        selects which dimensions are kept together, this method returns a list of
        groups where ranks are contiguous along the unmasked dimensions and
        interleaved along the masked ones.

        Args:
            world_size: Total number of GPUs.
            parallel_size: List of three integers: [tp, dp, pp] (order matters).
            mask: Boolean mask of length 3, True for dimensions that should
                  be part of the group (i.e., vary within the group).

        Returns:
            List of rank groups (each group is a list of integer ranks).
            Example: For TP groups, mask = [True, False, False] produces groups
            where ranks with the same DP and PP indices are grouped together.

        Implementation notes:
            - global_stride is the product of the dimensions after each step
              (prefix product excluding the final total product).
            - masked/unmasked stride and shape are extracted according to the mask.
            - The algorithm enumerates all possible group indices and for each,
              constructs the rank list by combining masked and unmasked coordinates.
        """
        # Extract dimensions that are masked (vary within group) and unmasked (fixed across groups)
        masked_shape = [s for s, m in zip(parallel_size, mask) if m]
        unmasked_shape = [s for s, m in zip(parallel_size, mask) if not m]

        # Compute global strides for each dimension (cumulative product of preceding dimensions)
        global_stride = self._prefix_product(parallel_size)[:-1]  # remove total product

        # Keep only the strides that correspond to masked/unmasked dimensions
        masked_stride = [d for d, m in zip(global_stride, mask) if m]
        unmasked_stride = [d for d, m in zip(global_stride, mask) if not m]

        # Number of groups = total ranks / (product of masked dimensions)
        group_size = np.prod(masked_shape)
        num_of_groups = world_size // group_size

        all_groups = []
        for group_index in range(num_of_groups):
            # Decompose group_index into coordinates along unmasked dimensions
            # Note: stride for decomposition is prefix product of unmasked_shape
            decomposed_group_idx = self._decompose(group_index, unmasked_shape,
                                                   self._prefix_product(unmasked_shape)[:-1])

            group_ranks = []
            for rank_in_group in range(group_size):
                # Decompose rank_in_group into coordinates along masked dimensions
                decomposed_rank_idx = self._decompose(rank_in_group, masked_shape,
                                                      self._prefix_product(masked_shape)[:-1])

                # Compute global rank by combining masked and unmasked coordinates
                # using the appropriate strides
                global_rank = (self._inner_product(decomposed_rank_idx, masked_stride) +
                               self._inner_product(decomposed_group_idx, unmasked_stride))
                group_ranks.append(int(global_rank))
            all_groups.append(group_ranks)

        return all_groups

    def init_parallel_strategy(self, tp: int, dp: int, pp: int) -> ParallelStrategyPerformance:
        """
        Initializes a ParallelStrategyPerformance object, automatically generating 
        Megatron-style rank groups, and registers it in the strategy matrix.

        Args:
            tp: Tensor parallelism degree.
            dp: Data parallelism degree.
            pp: Pipeline parallelism degree.

        Returns:
            The populated ParallelStrategyPerformance instance.
        """
        world_size = sum(len(node.gpus) for node in self.nodes)
        if world_size != (tp * pp * dp):
            raise ValueError(f"World size {world_size} mismatch with TP({tp})*PP({pp})*DP({dp})")

        # Create the strategy object
        strategy_perf = ParallelStrategyPerformance(tp=tp, dp=dp, pp=pp)

        # Standard Megatron dimension order: [TP, DP, PP]
        # This order determines the stride logic: global_rank = tp + dp*TP + pp*TP*DP
        parallel_size = [tp, dp, pp]

        # Populate logical groups
        strategy_perf.communication_groups = {
            "tp": self._generate_masked_orthogonal_rank_groups(world_size, parallel_size, [True, False, False]),
            "dp": self._generate_masked_orthogonal_rank_groups(world_size, parallel_size, [False, True, False]),
            "pp": self._generate_masked_orthogonal_rank_groups(world_size, parallel_size, [False, False, True])
        }
        
        for dim in ["tp", "dp", "pp"]:
            groups = strategy_perf.communication_groups.get(dim, [])
            
            # 1. Initialize all_case: dim -> group_tuple -> {CollectiveType: CommPerformance}
            strategy_perf.all_case[dim] = {
                tuple(group): {} for group in groups
            }
            
            # 2. Initialize dimension_performance: dim -> {CollectiveType: CommPerformance}
            strategy_perf.dimension_performance[dim] = {}
            
            # 3. Initialize the_worst_case: dim -> {CollectiveType: group_tuple}
            strategy_perf.the_worst_case[dim] = {}

        # Store in matrix
        self.strategy_matrix[(tp, dp, pp)] = strategy_perf
        return strategy_perf

    @staticmethod
    def _get_factors(n: int) -> List[int]:
        """Get all factors of n"""
        if n <= 0:
            raise ValueError(f"n must be a positive integer, got {n}")
        factors = set()
        for i in range(1, int(math.sqrt(n)) + 1):
            if n % i == 0:
                factors.add(i)
                factors.add(n // i)
        return sorted(factors)

    def get_plausible_strategies(
        self,
        preferred_tp: int = 4,
        model_ctx: Optional[Any] = None,  
        mbs: int = 1                      
    ) -> List[Tuple[int, int, int]]:
        """
        Generate robust (tp, dp, pp) strategies with realistic constraints.
        Optionally prune invalid/OOM strategies if a ModelContext is provided.

        Args:
            preferred_tp: The preferred TP degree used in scoring heuristic (default=4).
            model_ctx: Optional ModelContext instance to evaluate structural & memory feasibility.
            mbs: Micro-batch size used for memory estimation (default=1).

        Returns:
            A sorted list of (tp, dp, pp) tuples representing viable parallelism strategies.

        Raises:
            ValueError: If cluster configuration is invalid (e.g., zero GPUs).
        """
        # --- 0. Validate cluster config ---
        if self.total_nodes <= 0 or self.gpus_per_node <= 0:
            raise ValueError(
                f"Invalid cluster config: total_nodes={self.total_nodes}, "
                f"gpus_per_node={self.gpus_per_node}"
            )

        total_gpus = self.total_nodes * self.gpus_per_node
        raw_strategies = []

        # --- 1. TP candidates (strictly intra-node) ---
        tp_candidates = self._get_factors(self.gpus_per_node)

        # --- 2. Enumerate (tp, pp, dp) combinations ---
        for tp in tp_candidates:
            # Avoid full-TP (no model replication at all)
            if tp == total_gpus:
                continue

            remaining = total_gpus // tp

            for pp in self._get_factors(remaining):
                dp = remaining // pp  # exact by construction

                # --- 3. Hard constraints ---
                assert self.gpus_per_node % tp == 0, (
                    f"Invariant violated: gpus_per_node={self.gpus_per_node} not divisible by tp={tp}"
                )

                if dp < 1:
                    continue

                # --- 4. Heuristic filtering ---
                # Avoid trivial strategy: pure TP only
                if dp == 1 and pp == 1:
                    continue

                # Avoid full PP
                if pp == total_gpus:
                    continue

                # Avoid excessively deep pipeline
                if pp > self.total_nodes * 2:
                    continue

                # Prefer PP aligned with node boundaries
                if pp > self.total_nodes and pp % self.total_nodes != 0:
                    continue

                raw_strategies.append((tp, dp, pp))

        # --- 5. Deduplicate ---
        raw_strategies = list(set(raw_strategies))

        # --- 6. Model Context Pruning (High-Fidelity) ---
        strategies = []
        for tp, dp, pp in raw_strategies:
            if model_ctx is not None:

                report = model_ctx.evaluate_strategy_memory(
                    tp=tp, pp=pp, dp=dp, 
                    mbs=mbs, 
                    gpus_per_node=self.gpus_per_node, 
                    gpu_mem_limit_gb=self.min_gpu_mem_gb
                )
                
                # Key logic:
                # In evaluate_strategy_memory, 'feasible' is False only if utilization > 1.0 (absolute OOM)
                # or if dimensions are not divisible. Strategies that are tight or dangerous still have feasible=True,
                # so they are kept.
                if not report["feasible"]:
                    continue  # Skip strategies that are absolutely infeasible
                
            strategies.append((tp, dp, pp))

        # --- 7. Sort by heuristic score ---
        effective_preferred_tp = min(preferred_tp, self.gpus_per_node)

        def score(x: Tuple[int, int, int]) -> Tuple[int, int, int]:
            tp, dp, pp = x
            return (
                abs(tp - effective_preferred_tp),  # TP closer to preferred_tp is better
                -dp,                               # Larger DP is better
                abs(pp - self.total_nodes),        # PP near total_nodes is better
            )

        strategies.sort(key=score)

        print(f"[ClusterContext] Found {len(strategies)} viable strategies for "
            f"{self.total_nodes} nodes x {self.gpus_per_node} GPUs = {total_gpus} total GPUs")
        if model_ctx is not None:
            print(f"[ClusterContext] Strategies pruned using ModelContext constraints.")
            
        print(f"Top strategies (tp, dp, pp): {strategies[:10]}")

        return strategies

    # --- Helper Methods ---

    def validate_homogeneity(self) -> bool:
        """
        Check if all nodes have the same number of GPUs.
        Returns:
            True if homogeneous, False otherwise.
        """
        if not self.nodes:
            return True
        gpu_counts = [len(node.gpus) for node in self.nodes]
        return all(count == gpu_counts[0] for count in gpu_counts)

    def add_strategy_performance(self, perf: ParallelStrategyPerformance):
        """
        Store performance data for a specific parallel strategy.
        Args:
            perf: The ParallelStrategyPerformance object to add.
        """
        self.strategy_matrix[(perf.tp, perf.pp, perf.dp)] = perf