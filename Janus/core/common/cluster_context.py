import re
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple, Any

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
    parallelization strategy (TP, PP, DP).

    Attributes:
        tp: Tensor parallelism degree.
        pp: Pipeline parallelism degree.
        dp: Data parallelism degree.
        communication_groups: Dictionary mapping group type ("tp", "pp", "dp")
                              to a list of rank lists.
        dimension_performance: tp/dp/pp -> Dict[CollectiveType, CommPerformance]
    """
    tp: int
    pp: int
    dp: int
    communication_groups: Dict[str, List[List[int]]] = field(default_factory=dict)
    dimension_performance: Dict[str, Dict[CollectiveType, CommPerformance]] = field(default_factory=dict)

    def __repr__(self):
        return f"StrategyPerf(TP={self.tp}, PP={self.pp}, DP={self.dp})"


# --- 2. Hardware Unit Abstractions ---

@dataclass
class GPUInfo:
    """Information about a single GPU device."""
    global_id: int               # Rank in the global world
    local_id: int                # Rank within its node
    pci_bus_id: str              # PCI bus identifier
    type: str                    # GPU model (e.g., "A100")
    memory_capacity_gb: float
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
    (TP, PP, DP) following the exact logic used in Megatron-LM.
    """
    cluster_name: str
    total_nodes: int
    master_addr: str
    master_port: int
    gpus_per_node: int
    nodes: List[NodeInfo] = field(default_factory=list)
    env_fingerprint: str = ""
    # Map: global_rank -> node_id
    rank_to_node_map: Dict[int, int] = field(default_factory=dict, init=False)
    # Map: global_rank -> local_rank (rank within the node)
    rank_to_local_map: Dict[int, int] = field(default_factory=dict, init=False)
    strategy_matrix: Dict[Tuple[int, int, int], ParallelStrategyPerformance] = field(default_factory=dict)

    def __post_init__(self):
        """Build mappings and infer cluster properties."""
        self._build_rank_map()
        self._infer_gpus_per_node()

    def _build_rank_map(self):
        """Constructs both global-to-node and global-to-local mappings."""
        self.rank_to_node_map = {}
        self.rank_to_local_map = {}
        for node in self.nodes:
            for gpu in node.gpus:
                self.rank_to_node_map[gpu.global_id] = node.node_id
                self.rank_to_local_map[gpu.global_id] = gpu.local_id

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

    def _infer_gpus_per_node(self):
        """
        Infers gpus_per_node from the provided nodes list.
        If the cluster is heterogeneous, this marks the property as a 'reference' 
        value (usually the most common or the first node's count).
        """
        if not self.nodes:
            self.gpus_per_node = 0
            return

        counts = [len(node.gpus) for node in self.nodes]
        first_count = counts[0]
        
        is_homogeneous = self.validate_homogeneity()
        
        if is_homogeneous:
            self.gpus_per_node = first_count
        else:
            self.gpus_per_node = min(counts) 
            print(f"[Warning] Heterogeneous cluster detected. "
                  f"Using min(gpus_per_node)={self.gpus_per_node} for strategy generation.")

    def get_strategic_samples(self, groups: List[List[int]], parallel_type: str) -> List[List[int]]:
        """
        Two-stage Priority Sampling Algorithm:
        1. Separates groups into Cross-node and Intra-node pools.
        2. Prioritizes Cross-node samples (RDMA bottlenecks) using node-pair deduplication.
        3. Fills remaining budget with Intra-node samples to capture local variance.
        """
        if not groups:
            return []

        samples = []
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

        # Max samples capped at number of nodes to ensure efficiency
        max_samples = len(self.nodes)

        # --- Stage 2: Prioritize Cross-node Paths (The RDMA Bottleneck) ---
        for group, nodes in cross_node_pool:
            if nodes not in covered_node_pairs:
                samples.append(group)
                covered_node_pairs.add(nodes)
            
            if len(samples) >= max_samples:
                break

        # --- Stage 3: Fill remaining budget with Intra-node Paths ---
        if len(samples) < max_samples:
            for group, node_id in intra_node_pool:
                if node_id not in covered_single_nodes:
                    samples.append(group)
                    covered_single_nodes.add(node_id)
                
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

    def init_parallel_strategy(self, tp: int, pp: int, dp: int) -> ParallelStrategyPerformance:
        """
        Initializes a ParallelStrategyPerformance object, automatically generating 
        Megatron-style rank groups, and registers it in the strategy matrix.

        Args:
            tp: Tensor parallelism degree.
            pp: Pipeline parallelism degree.
            dp: Data parallelism degree.

        Returns:
            The populated ParallelStrategyPerformance instance.
        """
        world_size = sum(len(node.gpus) for node in self.nodes)
        if world_size != (tp * pp * dp):
            raise ValueError(f"World size {world_size} mismatch with TP({tp})*PP({pp})*DP({dp})")

        # Create the strategy object
        strategy_perf = ParallelStrategyPerformance(tp=tp, pp=pp, dp=dp)

        # Standard Megatron dimension order: [TP, DP, PP]
        # This order determines the stride logic: global_rank = tp + dp*TP + pp*TP*DP
        parallel_size = [tp, dp, pp]

        # Populate logical groups
        strategy_perf.communication_groups = {
            "tp": self._generate_masked_orthogonal_rank_groups(world_size, parallel_size, [True, False, False]),
            "dp": self._generate_masked_orthogonal_rank_groups(world_size, parallel_size, [False, True, False]),
            "pp": self._generate_masked_orthogonal_rank_groups(world_size, parallel_size, [False, False, True])
        }
        
        # Initialize the nested dict for performance metrics (empty placeholders)
        for dim in ["tp", "dp", "pp"]:
            strategy_perf.communication_performance[dim] = {
                tuple(group): {} for group in strategy_perf.communication_groups[dim]
            }

        # Store in matrix
        self.strategy_matrix[(tp, pp, dp)] = strategy_perf
        return strategy_perf

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
    ) -> List[Tuple[int, int, int]]:
        """
        Generate robust (tp, dp, pp) strategies with realistic constraints.

        Args:
            preferred_tp: The preferred TP degree used in scoring heuristic (default=4).

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
        strategies = []

        # --- 1. TP candidates (strictly intra-node) ---
        # Since tp divides gpus_per_node and total_gpus = nodes * gpus_per_node,
        # tp always divides total_gpus exactly — no remainder risk below.
        tp_candidates = _get_factors(self.gpus_per_node)

        # --- 2. Enumerate (tp, pp, dp) combinations ---
        for tp in tp_candidates:
            # Avoid full-TP (no model replication at all)
            if tp == total_gpus:
                continue

            # total_gpus // tp is exact because tp | gpus_per_node | total_gpus
            remaining = total_gpus // tp

            for pp in _get_factors(remaining):
                dp = remaining // pp  # exact by construction

                # --- 3. Hard constraints ---
                # TP must divide node GPUs (intra-node parallelism) — already guaranteed
                # by construction, but kept as an explicit safety assertion.
                assert self.gpus_per_node % tp == 0, (
                    f"Invariant violated: gpus_per_node={self.gpus_per_node} not divisible by tp={tp}"
                )

                # dp >= 1 is guaranteed by construction (remaining // pp >= 1),
                # but guard against unexpected edge cases.
                if dp < 1:
                    continue

                # --- 4. Heuristic filtering ---

                # Avoid trivial strategy: pure TP only (no DP, no PP)
                # Note: tp == total_gpus is already excluded above, so this
                # catches the degenerate dp=1, pp=1 case properly.
                if dp == 1 and pp == 1:
                    continue

                # Avoid full PP (entire model pipelined across all GPUs)
                if pp == total_gpus:
                    continue

                # Avoid excessively deep pipeline
                if pp > self.total_nodes * 2:
                    continue

                # --- 5. Topology-aware filtering ---
                # Prefer PP aligned with node boundaries for clean pipeline layout.
                if pp > self.total_nodes and pp % self.total_nodes != 0:
                    continue

                strategies.append((tp, dp, pp))

            # --- 6. Deduplicate (set handles duplicates from factor enumeration) ---
            strategies = list(set(strategies))

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

        print(f"[ClusterContext] Found {len(strategies)} strategies for "
            f"{self.total_nodes} nodes x {self.gpus_per_node} GPUs = {total_gpus} total GPUs")
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