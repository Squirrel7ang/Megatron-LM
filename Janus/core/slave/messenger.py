import socket
import struct
import threading
import time
import queue
import traceback
from typing import Optional, Dict, Any
from dataclasses import asdict

from core.common.protocol import JanusMessage, ActionCode, MessageType, StatusCode
from core.common.cluster_context import NodeInfo, CollectiveType
from core.common.prober.env_prober import EnvProber
from core.common.prober.compute_prober import ComputeProber
from core.common.prober.network.probe_engine import IntraNodeProber, InterNodeProber


class SlaveMessenger:
    """
    Hardened SlaveMessenger with extensive debug logging.
    """

    MAX_MESSAGE_SIZE = 50 * 1024 * 1024

    def __init__(self, master_host: str, master_port: int, rank: int, node_info: Any):
        self.master_host = master_host
        self.master_port = master_port
        self.rank = rank
        self.node_info = node_info

        self.master_rank: Optional[int] = None
        self.is_registered = threading.Event()

        # Prober Initialization
        self.env_prober = EnvProber(self.node_info)
        self.compute_prober = ComputeProber(self.node_info)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.send_lock = threading.Lock()

        self.command_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.recv_buffer = bytearray()

        self.listener_thread: Optional[threading.Thread] = None
        self.heartbeat_thread: Optional[threading.Thread] = None
        self.worker_thread: Optional[threading.Thread] = None

    def connect_to_master(self):
        try:
            self.sock.connect((self.master_host, self.master_port))
            handshake = JanusMessage(
                source_rank=self.rank, target=0,
                action=ActionCode.HANDSHAKE, msg_type=MessageType.REQUEST,
                payload={"version": "1.2.0-Hardened", "node_info": asdict(self.node_info)}
            )
            self.send_to_master(handshake)
            self.listener_thread = threading.Thread(target=self._run_listener, daemon=True)
            self.listener_thread.start()
            print(f"[Slave {self.rank}] Awaiting registration...")
        except Exception as e:
            raise ConnectionError(f"[Slave {self.rank}] Connection failed: {e}")

    def start_runtime(self):
        if self.heartbeat_thread:
            return
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self.heartbeat_thread.start()
        # Immediately send a heartbeat to let master know we are alive
        self._send_immediate_heartbeat()
        self.worker_thread = threading.Thread(target=self._dispatch_loop, daemon=True)
        self.worker_thread.start()

    def _send_immediate_heartbeat(self):
        if self.master_rank is not None:
            hb = JanusMessage(
                source_rank=self.rank, target=self.master_rank,
                action=ActionCode.HEARTBEAT, msg_type=MessageType.EVENT
            )
            self.send_to_master(hb)
            print(f"[Slave {self.rank}] Sent immediate heartbeat")

    def _run_listener(self):
        while not self.stop_event.is_set():
            try:
                self.sock.settimeout(1.0)
                data = self.sock.recv(16384)
                if not data:
                    print(f"[Slave {self.rank}] Master closed connection, listener exiting")
                    break
                self.recv_buffer.extend(data)
                self._process_buffer()
            except socket.timeout:
                continue
            except Exception as e:
                print(f"[Slave {self.rank}] Listener error: {e}")
                traceback.print_exc()
                break
        print(f"[Slave {self.rank}] Listener thread exiting")

    def _process_buffer(self):
        while len(self.recv_buffer) >= 4:
            body_len = struct.unpack(">I", self.recv_buffer[:4])[0]
            if body_len > self.MAX_MESSAGE_SIZE:
                print(f"[Slave {self.rank}] Message too large ({body_len}), closing")
                self.close()
                break
            if len(self.recv_buffer) < 4 + body_len:
                break
            body = self.recv_buffer[4 : 4 + body_len]
            del self.recv_buffer[: 4 + body_len]
            try:
                msg = JanusMessage.from_json(body.decode('utf-8'))
                if not self.is_registered.is_set():
                    if msg.action == ActionCode.HANDSHAKE and msg.status == StatusCode.SUCCESS:
                        self.master_rank = msg.source_rank
                        self.is_registered.set()
                        self.start_runtime()
                        print(f"[Slave {self.rank}] Registered successfully.")
                        continue
                if msg.msg_type == MessageType.REQUEST:
                    self.command_queue.put(msg)
                    print(f"[DEBUG] Slave {self.rank} enqueued command {msg.action}")
            except Exception as e:
                print(f"[Slave {self.rank}] Protocol error: {e}")

    def _dispatch_loop(self):
        print(f"[Slave {self.rank}] Dispatch loop started")
        while not self.stop_event.is_set():
            try:
                cmd = self.command_queue.get(timeout=1.0)
                print(f"[DEBUG] Slave {self.rank} received command: {cmd.action}")
            except queue.Empty:
                continue

            if cmd.action == ActionCode.PROBE_ENV:
                self._wrap_probe(self.env_prober.probe, cmd)
            elif cmd.action == ActionCode.PROBE_COMPUTE:
                self._wrap_probe(self.compute_prober.probe, cmd)
            elif cmd.action == ActionCode.PROBE_NET_INTRA or cmd.action == ActionCode.PROBE_NET_INTER:
                self._handle_probe_network(cmd)
            elif cmd.action == ActionCode.TERMINATE:
                self.close()
                break

    def _wrap_probe(self, probe_func, request: JanusMessage):
        try:
            probe_func()
            self._send_response(request, StatusCode.SUCCESS, asdict(self.node_info))
        except Exception as e:
            print(f"[Slave {self.rank}] Probe error: {e}")
            traceback.print_exc()
            self._send_response(request, StatusCode.ERROR, {"error": str(e)})

    def _handle_probe_network(self, request: JanusMessage):
        """
        Handles network probing tasks with multi-process spawning.
        Aggregates results using the 'Straggler' logic (worst-case performance)
        and returns complete task metadata (task_id, dim, etc.) to the Master.
        """
        import dataclasses
        import multiprocessing as mp
        from queue import Empty
        import traceback

        action_code = request.action
        payload = request.payload
        
        # 1. Extract metadata from payload for task tracking
        probe_dim = payload.get("dim")
        strategy = payload.get("strategy")
        group_idx = payload.get("group_idx")
        task_id = payload.get("task_id")
        
        local_ranks = payload.get("local_ranks")   
        global_ranks = payload.get("global_ranks") 
        
        # 2. Determine communication protocol (P2P for PP, All-Reduce for TP/DP)
        communication_type = "P2P" if probe_dim == "pp" else CollectiveType.ALL_REDUCE
        
        # 3. Select benchmark message sizes based on the parallel dimension
        if probe_dim == "tp":
            sizes_bytes = ProbeEngine.TP_SIZES_BYTES
        elif probe_dim == "dp":
            sizes_bytes = ProbeEngine.DP_SIZES_BYTES
        else:
            sizes_bytes = ProbeEngine.GENERIC_SIZES_BYTES

        try:
            perf = None

            # --- Case A: Intra-node Probing ---
            if action_code == ActionCode.PROBE_NET_INTRA:
                print(f"[Slave {self.rank}] INTRA-node probe for task {task_id}")
                prober = IntraNodeProber(local_rank_list=local_ranks, coll=communication_type)
                perf = prober.probe(sizes_bytes=sizes_bytes)

            # --- Case B: Inter-node Probing ---
            elif action_code == ActionCode.PROBE_NET_INTER:
                master_addr = payload.get("master_addr")
                master_port = payload.get("master_port")
                
                ctx = mp.get_context("spawn")
                q = ctx.Queue()

                # Map local hardware IDs to cluster-wide global ranks
                rank_map = {lr: self.cluster_context.get_global_rank(self.rank, lr) 
                           for lr in local_ranks}

                print(f"[Slave {self.rank}] INTER-node probe for task {task_id}, spawning workers...")
                
                mp.spawn(
                    self._inter_probe_worker,
                    args=(local_ranks, global_ranks, rank_map, master_addr, master_port, 
                          communication_type, sizes_bytes, q),
                    nprocs=len(local_ranks),
                    join=True
                )

                # Collect results from all participants on this node
                results = []
                num_expected = len(local_ranks)
                try:
                    for _ in range(num_expected):
                        results.append(q.get(timeout=20))
                except Empty:
                    print(f"[Slave {self.rank}] Timeout: Task {task_id} received {len(results)}/{num_expected}")
                finally:
                    q.close()

                if not results:
                    raise RuntimeError(f"Inter-node probe {task_id} failed: No results.")

                # --- Straggler-Aware Aggregation (Worst-Case) ---
                worst_latency = max(r.latency_us for r in results)
                worst_bandwidth = min(r.bandwidth_gbps for r in results)
                worst_bus_bandwidth = min(r.bus_bandwidth_gbps for r in results)

                perf = CommPerformance(
                    latency_us=float(worst_latency),
                    bandwidth_gbps=float(worst_bandwidth),
                    bus_bandwidth_gbps=float(worst_bus_bandwidth)
                )

            if perf is None:
                raise ValueError(f"Invalid network action for task {task_id}: {action_code}")

            # 4. Construct response payload including ALL task metadata
            response_data = {
                "task_id": task_id,
                "dim": probe_dim,
                "strategy": strategy,
                "group_idx": group_idx,
                "perf": dataclasses.asdict(perf)
            }

            self._send_response(request, StatusCode.SUCCESS, response_data)

        except Exception as e:
            print(f"[Slave {self.rank}] Network probe error (Task {task_id}): {e}")
            traceback.print_exc()
            # Still include task metadata in error response for Master to handle cleanup
            error_data = {
                "task_id": task_id,
                "dim": probe_dim,
                "error": str(e)
            }
            self._send_response(request, StatusCode.ERROR, error_data)

    @staticmethod
    def _inter_probe_worker(i, local_ranks, global_ranks, rank_map, master_addr, 
                           master_port, coll, sizes_bytes, q):
        """
        Independent worker process for inter-node communication benchmarking.
        """
        my_local_gpu = local_ranks[i]
        my_global_rank = rank_map[my_local_gpu]

        try:
            prober = InterNodeProber(
                master_addr=master_addr,
                master_port=master_port,
                rank_list=global_ranks,
                coll=coll
            )
            
            perf = prober.probe(
                my_global_rank=my_global_rank,
                my_local_gpu=my_local_gpu,
                sizes_bytes=sizes_bytes
            )
            q.put(perf)
            
        except Exception as e:
            print(f"[Worker] Global Rank {my_global_rank} failed: {e}")

    def _send_response(self, request: JanusMessage, status: StatusCode, payload: Dict):
        """
        Wraps and sends the response back to the Master Orchestrator.
        """
        resp = JanusMessage(
            source_rank=self.rank,
            target=self.master_rank,
            action=request.action,
            msg_type=MessageType.RESPONSE,
            request_id=request.request_id,
            status=status,
            payload=payload
        )
        self.send_to_master(resp)

    def send_to_master(self, msg: JanusMessage):
        """
        Thread-safe socket communication for sending messages to Master.
        """
        import struct
        try:
            body = msg.to_json().encode('utf-8')
            header = struct.pack(">I", len(body))
            with self.send_lock:
                self.sock.sendall(header + body)
        except Exception as e:
            print(f"[Slave {self.rank}] Socket send error: {e}")

    def _heartbeat_loop(self):
        while not self.stop_event.is_set():
            try:
                if self.is_registered.is_set():
                    hb = JanusMessage(
                        source_rank=self.rank, target=self.master_rank,
                        action=ActionCode.HEARTBEAT, msg_type=MessageType.EVENT
                    )
                    self.send_to_master(hb)
                time.sleep(5)
            except:
                break

    def close(self):
        print(f"[Slave {self.rank}] close() called")
        self.stop_event.set()
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
            self.sock.close()
        except:
            pass