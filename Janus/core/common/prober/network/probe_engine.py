import time
import socket
import random
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datetime import timedelta
from queue import Empty
from typing import List, Optional, Union

from core.common.cluster_context import CommPerformance, CollectiveType


class ProbeEngine:
    """
    Expert-level probing engine for Megatron-LM parallel strategy selection.

    Correctness guarantees:
    - True serial Ping-Pong RTT via isend/irecv sequential wait (NCCL-safe).
    - GPU-side timing with cuda.synchronize() bracketing send loops.
    - aggregate_worst_case uses independent contiguous tensors (no view in-place).
    - Bus-bandwidth = AlgoBW / factor (NCCL convention: BusBW <= AlgoBW).
    - Megatron-aware size distributions: small sizes for TP latency, configurable
      activation size for PP bandwidth, bucket-granularity for DP gradient sync.
    - Bandwidth reported as top-k mean to reduce sensitivity to one-off bursts.
    - Compute-jitter injection placed after the opening barrier so each rank
      independently delays before posting the collective, correctly modelling the
      skew accumulated during forward/backward computation.
    - P2P loop restricted to neighbor pairs (rank ↔ rank+neighbor_distance) to
      reflect Megatron PP topology instead of an unrealistic full-mesh sweep.
    - Closing dist.barrier() in benchmark_p2p is always reached via finally,
      so a mid-pair exception cannot leave non-participant ranks hung forever.
    """

    # ---------------------------------------------------------------------------
    # Megatron communication size profiles
    # ---------------------------------------------------------------------------
    # TP: attention output tensors are typically small (512B ~ 1MB).
    TP_SIZES_BYTES: List[int] = [
        512, 1024, 2048, 4096, 8192, 16384, 32768,
        65536, 131072, 262144, 524288, 1048576,
    ]

    # DP: gradient bucket sizes (PyTorch DDP default is 25 MB; sweep around it).
    DP_SIZES_BYTES: List[int] = [
        1   * 1024 * 1024,
        5   * 1024 * 1024,
        10  * 1024 * 1024,
        25  * 1024 * 1024,
        50  * 1024 * 1024,
        100 * 1024 * 1024,
        200 * 1024 * 1024,
        256 * 1024 * 1024,
    ]

    # Generic sweep used when no profile is specified.
    GENERIC_SIZES_BYTES: List[int] = [4096 * (2 ** i) for i in range(16)]

    # ---------------------------------------------------------------------------
    # Backend detection
    # ---------------------------------------------------------------------------

    @staticmethod
    def detect_backend() -> str:
        if torch.cuda.is_available():
            return "nccl"
        if hasattr(torch, "npu") and torch.npu.is_available():
            return "hccl"
        return "gloo"

    # ---------------------------------------------------------------------------
    # Aggregation
    # ---------------------------------------------------------------------------

    @staticmethod
    def aggregate_worst_case(perf: CommPerformance, device: torch.device) -> CommPerformance:
        """
        Reduces group-wide worst-case performance: MAX latency, MIN bandwidth.

        Uses three independent scalar tensors (not views of a shared buffer) to
        satisfy NCCL's requirement for contiguous, non-aliased all_reduce inputs.

        Ranks that did not act as P2P src report bandwidth=0.0.  To prevent their
        zero from corrupting the MIN, unmeasured ranks substitute +inf before the
        MIN reduction.  A companion flag tensor tracks whether any rank actually
        measured something; if none did, the final bandwidth is reported as 0.0.
        """
        has_bw = perf.bandwidth_gbps > 0.0

        lat_t  = torch.tensor(perf.latency_us,
                              device=device, dtype=torch.float32)
        bw_t   = torch.tensor(perf.bandwidth_gbps if has_bw else float("inf"),
                              device=device, dtype=torch.float32)
        flag_t = torch.tensor(1.0 if has_bw else 0.0,
                              device=device, dtype=torch.float32)

        dist.all_reduce(lat_t,  op=dist.ReduceOp.MAX)
        dist.all_reduce(bw_t,   op=dist.ReduceOp.MIN)
        dist.all_reduce(flag_t, op=dist.ReduceOp.MAX)

        final_bw = float(bw_t.item()) if float(flag_t.item()) > 0.0 else 0.0
        if final_bw == float("inf"):
            final_bw = 0.0

        return CommPerformance(
            latency_us=round(float(lat_t.item()), 2),
            bandwidth_gbps=round(final_bw, 2),
        )

    # ---------------------------------------------------------------------------
    # Collective benchmark
    # ---------------------------------------------------------------------------

    @staticmethod
    def benchmark_collective(
        device: torch.device,
        coll: CollectiveType,
        sizes_bytes: Optional[List[int]] = None,
        compute_jitter_ms: float = 0.0,
        bw_top_k: int = 3,
    ) -> CommPerformance:
        """
        Measures collective communication performance across a sweep of message sizes.

        Args:
            device:             Target device.
            coll:               Collective type (ALL_REDUCE, ALL_GATHER, etc.).
            sizes_bytes:        Message sizes in bytes.  Defaults to GENERIC_SIZES_BYTES.
                                Pass TP_SIZES_BYTES for tensor-parallel workloads or
                                DP_SIZES_BYTES for data-parallel gradient buckets.
            compute_jitter_ms:  Max random sleep (ms) injected *after* the opening
                                barrier but *before* the warmup, simulating the
                                cross-rank compute imbalance each rank accumulates
                                during forward/backward before entering the collective.
                                Disabled (0.0) by default for raw link characterisation.
            bw_top_k:           Number of top bandwidth samples to average when
                                computing the reported bandwidth.  Using the mean of
                                top-k (rather than the single peak) reduces sensitivity
                                to one-off measurement bursts while still reflecting
                                near-peak sustained throughput.  Default is 3.
        """
        if sizes_bytes is None:
            sizes_bytes = ProbeEngine.GENERIC_SIZES_BYTES

        world_size = dist.get_world_size()
        times: List[float] = []
        use_cuda = device.type == "cuda"

        for size in sizes_bytes:
            t_in, t_out = ProbeEngine._prepare_tensor(coll, size, world_size, device)

            # Opening barrier: ensures no rank begins kernel submission while
            # another is still allocating tensors (visible on slow NUMA nodes).
            dist.barrier()

            # Jitter is injected AFTER the barrier so that each rank independently
            # delays before posting the collective.  This correctly models the skew
            # that accumulates during computation rather than the skew that exists
            # before ranks first rendezvous.
            if compute_jitter_ms > 0.0:
                time.sleep(random.uniform(0.0, compute_jitter_ms) / 1000.0)

            for _ in range(5):
                ProbeEngine._dispatch(coll, t_in, t_out)

            # Second barrier + GPU drain before the timed window.
            dist.barrier()
            if use_cuda:
                from megatron.core.distributed.overlap_tracker import overlap_tracker
                overlap_tracker.stop_cpu_compute()
                torch.cuda.synchronize()
                overlap_tracker.start_cpu_compute()

            iters = 20
            if use_cuda:
                # CUDA Events measure GPU-side elapsed time, unaffected by CPU
                # scheduling jitter — important for short, latency-dominated collectives.
                start_ev = torch.cuda.Event(enable_timing=True)
                end_ev   = torch.cuda.Event(enable_timing=True)
                start_ev.record()
            else:
                t0 = time.perf_counter()

            for _ in range(iters):
                ProbeEngine._dispatch(coll, t_in, t_out)

            if use_cuda:
                end_ev.record()
                from megatron.core.distributed.overlap_tracker import overlap_tracker
                overlap_tracker.stop_cpu_compute()
                torch.cuda.synchronize()
                overlap_tracker.start_cpu_compute()
                elapsed_ms = start_ev.elapsed_time(end_ev)
            else:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

            times.append((elapsed_ms / 1000.0) / iters)

        return ProbeEngine._extract_alpha_beta(sizes_bytes, times, world_size, coll,
                                               bw_top_k=bw_top_k)

    # ---------------------------------------------------------------------------
    # P2P benchmark
    # ---------------------------------------------------------------------------

    @staticmethod
    def benchmark_p2p(
        rank: int,
        device: torch.device,
        activation_size_bytes: int = 64 * 1024 * 1024,
        neighbor_distance: int = 1, 
    ) -> CommPerformance:
        """
        P2P benchmark modelling Megatron pipeline-parallel activation transfers.
 
        Only neighbor pairs (rank, rank+neighbor_distance) are tested, matching
        the unidirectional ring topology that Megatron PP actually uses.  A full
        N×N mesh would measure worst-case cross-rail links that are never exercised
        during training, artificially inflating latency and deflating bandwidth.
 
        Latency — True serial Ping-Pong (src times, dst echoes):
            Each iteration: src issues isend→wait, then irecv→wait.
            Dst mirrors:    irecv→wait, then isend→wait.
            Strictly serial: the next op is posted only after the current one
            completes.  Safe under NCCL because no unmatched sends accumulate.
            Reported latency = total_wall_time / (iters * 2)  [RTT / 2 = one-way].
 
        Bandwidth — Unidirectional (src → dst only):
            Models the one-way activation tensor transfer between PP stages.
            cuda.synchronize() brackets the timing window to capture true DMA
            completion, not kernel-launch time.
 
        Barrier safety:
            The closing dist.barrier() is always executed via a finally block.
            This guarantees that a mid-pair exception (e.g. NCCL timeout on a
            degraded link) does not leave non-participant ranks permanently hung.
 
        Args:
            rank:                   This process's group rank.
            device:                 Target device.
            activation_size_bytes:  Bytes per activation tensor.  Should match
                                    micro_batch_size × seq_len × hidden_dim × dtype_bytes
                                    for the target model configuration.
            neighbor_distance:      Step size between PP stages.  Use 1 for standard
                                    Megatron PP (consecutive ranks); use gpus_per_node
                                    for inter-node PP where stages span nodes.
        """
        world_size      = dist.get_world_size()
        worst_latency   = 0.0
        worst_bandwidth = 0.0
        use_cuda = device.type == "cuda"

        pairs = [
            (s, (s + neighbor_distance) % world_size)
            for s in range(world_size)
            if s != (s + neighbor_distance) % world_size
        ]

        # ------------------------------------------------------------------
        # P2P measurement
        # ------------------------------------------------------------------
        for src, dst in pairs:
            dist.barrier()

            participating = rank in (src, dst)
            try:
                if not participating:
                    continue

                peer = dst if rank == src else src

                # ---- Latency ----
                ping_t = torch.zeros(1, device=device)
                pong_t = torch.zeros(1, device=device)
                iters  = 20

                for _ in range(5):
                    if rank == src:
                        dist.isend(ping_t, peer).wait()
                        dist.irecv(pong_t, peer).wait()
                    else:
                        dist.irecv(pong_t, peer).wait()
                        dist.isend(ping_t, peer).wait()

                if use_cuda:
                    from megatron.core.distributed.overlap_tracker import overlap_tracker
                    overlap_tracker.stop_cpu_compute()
                    torch.cuda.synchronize()
                    overlap_tracker.start_cpu_compute()

                if rank == src:
                    t_start = time.perf_counter()
                    for _ in range(iters):
                        dist.isend(ping_t, peer).wait()
                        dist.irecv(pong_t, peer).wait()
                    if use_cuda:
                        from megatron.core.distributed.overlap_tracker import overlap_tracker
                        overlap_tracker.stop_cpu_compute()
                        torch.cuda.synchronize()
                        overlap_tracker.start_cpu_compute()
                    lat_us = ((time.perf_counter() - t_start) * 1e6) / (iters * 2)
                    worst_latency = max(worst_latency, lat_us)
                else:
                    for _ in range(iters):
                        dist.irecv(pong_t, peer).wait()
                        dist.isend(ping_t, peer).wait()
                    if use_cuda:
                        from megatron.core.distributed.overlap_tracker import overlap_tracker
                        overlap_tracker.stop_cpu_compute()
                        torch.cuda.synchronize()
                        overlap_tracker.start_cpu_compute()

                # ---- Bandwidth ----
                buf      = torch.randn(activation_size_bytes // 4, device=device)
                recv_buf = torch.empty(activation_size_bytes // 4, device=device)
                bw_iters = 10

                if rank == src:
                    if use_cuda:
                        from megatron.core.distributed.overlap_tracker import overlap_tracker
                        overlap_tracker.stop_cpu_compute()
                        torch.cuda.synchronize()
                        overlap_tracker.start_cpu_compute()
                    t_start = time.perf_counter()
                    for _ in range(bw_iters):
                        dist.isend(buf, peer).wait()
                    if use_cuda:
                        from megatron.core.distributed.overlap_tracker import overlap_tracker
                        overlap_tracker.stop_cpu_compute()
                        torch.cuda.synchronize()
                        overlap_tracker.start_cpu_compute()
                    t_end = time.perf_counter()

                    bw_gbps = (activation_size_bytes * bw_iters * 8) / \
                              ((t_end - t_start) * 1e9)

                    worst_bandwidth = bw_gbps if worst_bandwidth == 0.0 \
                                      else min(worst_bandwidth, bw_gbps)
                else:
                    for _ in range(bw_iters):
                        dist.irecv(recv_buf, peer).wait()
                    if use_cuda:
                        from megatron.core.distributed.overlap_tracker import overlap_tracker
                        overlap_tracker.stop_cpu_compute()
                        torch.cuda.synchronize()
                        overlap_tracker.start_cpu_compute()

            finally:
                # barrier always executed by ALL ranks
                dist.barrier()

        return CommPerformance(latency_us=worst_latency, bandwidth_gbps=worst_bandwidth)

    # ---------------------------------------------------------------------------
    # Alpha-Beta extraction
    # ---------------------------------------------------------------------------

    @staticmethod
    def _extract_alpha_beta(
        sizes: List[int],
        times: List[float],
        world_size: int,
        coll: CollectiveType,
        bw_top_k: int = 3,
    ) -> CommPerformance:
        """
        Extracts alpha (base latency) and beta (inverse bandwidth) from timing data.

        Algorithm bandwidth (AlgoBW) = MessageSize / Time.

        Bus bandwidth (BusBW) reflects physical link utilisation as quoted by
        hardware vendors.  NCCL's convention is:

            BusBW = AlgoBW / factor

        where factor < 1 for ring-based collectives (so BusBW <= AlgoBW):
            ALL_REDUCE:             factor = 2 * (n-1) / n
            ALL_GATHER / RED_SCAT:  factor = (n-1) / n
            BROADCAST:              factor = 1

        Bandwidth is reported as the mean of the top-k AlgoBW samples instead
        of the single peak.  This reduces the influence of one-off measurement
        bursts (e.g. a cache-hot buffer or a momentarily quiet fabric) while
        still reflecting the near-peak sustained throughput that training sees
        over many iterations.  bw_top_k=3 is a reasonable default; increase it
        for noisier environments.

        Alpha (latency intercept) is the minimum of the four smallest-message
        timings, keeping the estimate in the latency-dominated regime.
        """
        alpha_us = min(times[:4]) * 1e6

        if coll == CollectiveType.ALL_REDUCE:
            factor = 2.0 * (world_size - 1) / world_size if world_size > 1 else 1.0
        elif coll in (CollectiveType.ALL_GATHER, CollectiveType.REDUCE_SCATTER):
            factor = (world_size - 1) / world_size if world_size > 1 else 1.0
        else:
            factor = 1.0

        # Compute AlgoBW for every (size, time) pair.
        algo_bws = [(s * 8) / (t * 1e9) for s, t in zip(sizes, times)]

        # Top-k mean: sort ascending, take the last k entries, average them.
        k = min(bw_top_k, len(algo_bws))
        top_k_bws  = sorted(algo_bws)[-k:]
        rep_algo_bw = sum(top_k_bws) / len(top_k_bws)

        bus_bw = rep_algo_bw / factor if factor > 0.0 else rep_algo_bw

        return CommPerformance(
            latency_us=round(max(alpha_us, 0.1), 2),
            bandwidth_gbps=round(rep_algo_bw, 2),
            bus_bandwidth_gbps=round(bus_bw, 2),
        )

    # ---------------------------------------------------------------------------
    # Tensor helpers
    # ---------------------------------------------------------------------------

    @staticmethod
    def _prepare_tensor(coll, size, ws, device):
        """
        Allocates input/output tensors for a collective op at the given byte size.
        The minimum size guard prevents zero-element tensors on very large world sizes.
        """
        min_el = max(ws * 4, 128)
        size   = max(size, min_el)

        if coll == CollectiveType.ALL_GATHER:
            t_in  = torch.randn(size // ws // 4, device=device).contiguous()
            t_out = torch.empty(size // 4,        device=device).contiguous()
            assert t_in.numel() > 0 and t_out.numel() > 0, \
                f"ALL_GATHER tensor degenerated: size={size}, ws={ws}"
            return t_in, t_out

        if coll == CollectiveType.REDUCE_SCATTER:
            t_in  = torch.randn(size // 4,        device=device).contiguous()
            t_out = torch.empty(size // ws // 4,  device=device).contiguous()
            assert t_in.numel() > 0 and t_out.numel() > 0, \
                f"REDUCE_SCATTER tensor degenerated: size={size}, ws={ws}"
            return t_in, t_out

        t = torch.randn(size // 4, device=device).contiguous()
        return t, t

    @staticmethod
    def _dispatch(coll, t_in, t_out):
        if coll == CollectiveType.ALL_REDUCE:
            dist.all_reduce(t_in)
        elif coll == CollectiveType.ALL_GATHER:
            dist.all_gather_into_tensor(t_out, t_in)
        elif coll == CollectiveType.REDUCE_SCATTER:
            dist.reduce_scatter_tensor(t_out, t_in)
        elif coll == CollectiveType.BROADCAST:
            dist.broadcast(t_in, src=0)


# ---------------------------------------------------------------------------
# IntraNodeProber
# ---------------------------------------------------------------------------

class IntraNodeProber:
    """
    Spawns local sub-processes to characterise intra-node communication
    (NVLink bandwidth for TP, PCIe for CPU-only nodes).

    Uses an explicit tcp:// init_method with rank/world_size passed directly to
    init_process_group, avoiding os.environ mutation that would interfere with
    other probers running concurrently in the same parent process.
    """

    def __init__(self, local_rank_list: List[int], coll: Union[CollectiveType, str]):
        self.local_rank_list = local_rank_list
        self.coll            = coll
        self.world_size      = len(local_rank_list)
        self.backend         = ProbeEngine.detect_backend()

    def probe(
        self,
        activation_size_bytes: int = 64 * 1024 * 1024,
        sizes_bytes: Optional[List[int]] = None,
        compute_jitter_ms: float = 0.0,
        neighbor_distance: int = 1,
        pg_timeout_s: int = 120,
    ) -> CommPerformance:
        """
        Args:
            activation_size_bytes:  Forwarded to benchmark_p2p (PP activation tensor size).
            sizes_bytes:            Forwarded to benchmark_collective.  Use
                                    ProbeEngine.TP_SIZES_BYTES for tensor-parallel or
                                    ProbeEngine.DP_SIZES_BYTES for data-parallel workloads.
            compute_jitter_ms:      Max jitter (ms) injected after each barrier to
                                    simulate compute imbalance.
            neighbor_distance:      Forwarded to benchmark_p2p (PP stage stride).
            pg_timeout_s:           Seconds before init_process_group is declared hung.
        """
        if self.world_size < 2:
            return CommPerformance()

        ctx  = mp.get_context("spawn")
        q    = ctx.Queue()
        port = self._get_free_port()

        mp.spawn(
            self._worker_fn,
            args=(self.local_rank_list, self.coll, self.backend, port, q,
                  activation_size_bytes, sizes_bytes, compute_jitter_ms,
                  neighbor_distance, pg_timeout_s),
            nprocs=self.world_size,
            join=True,
        )

        try:
            return q.get(timeout=300)
        except Empty:
            return CommPerformance()
        finally:
            q.close()

    @staticmethod
    def _worker_fn(rank, local_rank_list, coll, backend, port, q,
                   activation_size_bytes, sizes_bytes, compute_jitter_ms,
                   neighbor_distance, pg_timeout_s):
        init_method = f"tcp://127.0.0.1:{port}"
        gpu_id = local_rank_list[rank]

        if backend in ["nccl", "hccl"]:
            torch.cuda.set_device(gpu_id)
            device = torch.device(f"cuda:{gpu_id}")
        else:
            device = torch.device("cpu")

        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            rank=rank,
            world_size=len(local_rank_list),
            timeout=timedelta(seconds=pg_timeout_s),
        )

        try:
            if coll == "P2P":
                perf = ProbeEngine.benchmark_p2p(
                    rank, device,
                    activation_size_bytes=activation_size_bytes,
                    neighbor_distance=neighbor_distance,
                )
            else:
                perf = ProbeEngine.benchmark_collective(
                    device, coll,
                    sizes_bytes=sizes_bytes,
                    compute_jitter_ms=compute_jitter_ms,
                )

            result = ProbeEngine.aggregate_worst_case(perf, device)
            if rank == 0:
                q.put(result)
        finally:
            dist.destroy_process_group()

    @staticmethod
    def _get_free_port() -> int:
        """
        Finds a free TCP port with randomised retry to reduce TOCTOU collision
        probability in high-concurrency cluster environments.

        NOTE: For large-scale deployments where port collisions remain a concern,
        consider having the orchestrator pre-allocate and distribute ports before
        launching any prober, eliminating the race entirely.
        """
        for _ in range(20):
            port = random.randint(20000, 60000)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("", port))
                    return port
                except OSError:
                    time.sleep(random.uniform(0.01, 0.05))
        raise RuntimeError(
            "IntraNodeProber: could not acquire a free TCP port after 20 attempts."
        )


# ---------------------------------------------------------------------------
# InterNodeProber
# ---------------------------------------------------------------------------

class InterNodeProber:
    """
    Joins a globally coordinated distributed group to characterise inter-node
    communication (InfiniBand / RoCE for PP and DP workloads).

    Concurrency contract:
        All ranks in rank_list MUST call probe() concurrently.  Staggered calls
        will cause init_process_group to time out after pg_timeout_s seconds.
        Enforcing this constraint is the orchestrator's responsibility; the prober
        cannot verify it independently.

    Uses an explicit tcp:// init_method with rank/world_size passed directly to
    init_process_group to prevent os.environ races when multiple InterNodeProber
    instances probe different rank groups within the same process.
    """

    def __init__(
        self,
        master_addr: str,
        master_port: int,
        rank_list: List[int],
        coll: Union[CollectiveType, str],
    ):
        self.master_addr = master_addr
        self.master_port = master_port
        self.rank_list   = rank_list
        self.coll        = coll
        self.backend     = ProbeEngine.detect_backend()

    def probe(
        self,
        my_global_rank: int,
        my_local_gpu: int,
        activation_size_bytes: int = 64 * 1024 * 1024,
        sizes_bytes: Optional[List[int]] = None,
        compute_jitter_ms: float = 0.0,
        neighbor_distance: int = 1,
        pg_timeout_s: int = 120,
    ) -> CommPerformance:
        """
        Args:
            my_global_rank:         This process's global rank in the full job.
            my_local_gpu:           Local GPU index to bind (ignored for CPU backend).
            activation_size_bytes:  Forwarded to benchmark_p2p.
            sizes_bytes:            Forwarded to benchmark_collective.
            compute_jitter_ms:      Max jitter (ms) injected after each barrier.
            neighbor_distance:      Forwarded to benchmark_p2p (PP stage stride).
            pg_timeout_s:           Seconds before init_process_group is declared hung.
                                    60–120 s is safe for most clusters; increase if
                                    the rendezvous backend (e.g. etcd, c10d-store) is slow.
        """
        try:
            group_rank = self.rank_list.index(my_global_rank)
        except ValueError:
            return CommPerformance()

        init_method = f"tcp://{self.master_addr}:{self.master_port}"

        if self.backend in ["nccl", "hccl"]:
            torch.cuda.set_device(my_local_gpu)
            device = torch.device(f"cuda:{my_local_gpu}")
        else:
            device = torch.device("cpu")

        dist.init_process_group(
            backend=self.backend,
            init_method=init_method,
            rank=group_rank,
            world_size=len(self.rank_list),
            timeout=timedelta(seconds=pg_timeout_s),
        )

        try:
            if self.coll == "P2P":
                perf = ProbeEngine.benchmark_p2p(
                    group_rank, device,
                    activation_size_bytes=activation_size_bytes,
                    neighbor_distance=neighbor_distance,
                )
            else:
                perf = ProbeEngine.benchmark_collective(
                    device, self.coll,
                    sizes_bytes=sizes_bytes,
                    compute_jitter_ms=compute_jitter_ms,
                )

            return ProbeEngine.aggregate_worst_case(perf, device)
        finally:
            # NOTE: NCCL communicator teardown flushes all pending streams and
            # carries measurable overhead.  If probing multiple parallel groups
            # back-to-back, consider keeping the process group alive across calls
            # and passing it explicitly instead of rebuilding it per probe.
            dist.destroy_process_group()