import socket
import struct
import select
import time
import threading
import queue
from collections import defaultdict
from typing import List, Optional, Dict, Set, Any

from core.common.protocol import JanusMessage, MessageType, ActionCode, StatusCode
from core.common.prober.env_prober import EnvProber
from core.common.prober.compute_prober import ComputeProber
from core.common.prober.network.probe_engine import IntraNodeProber


class MasterMessenger:
    """
    Master side communication manager for the Janus cluster.

    Responsibilities:
        - Accept and manage slave connections (handshake, liveness, disconnection)
        - Provide thread‑safe broadcast and point‑to‑point messaging
        - Aggregate responses for asynchronous task submissions
        - Maintain cluster topology (slave_conns, rank_to_conn, etc.)

    Thread safety:
        - conn_lock protects all shared connection state (slave_conns, rank_to_conn,
          conn_to_rank, all_known_ranks, last_seen, recv_buffers)
        - send_lock protects concurrent sendall operations
        - inbox_lock protects the inbox dictionary used for response aggregation
        - Polling thread runs in the background; all interaction with connection
          state happens under conn_lock to avoid data races.
    """

    MAX_MESSAGE_SIZE = 50 * 1024 * 1024  # 50MB safety limit

    def __init__(self, host: str, port: int, expected_slaves: int, cluster: Any):
        self.host, self.port = host, port
        self.expected_slaves = expected_slaves
        self.cluster = cluster

        # Topology & Identity Management                  
        self.slave_conns: List[socket.socket] = []
        self.rank_to_conn: Dict[int, socket.socket] = {}
        self.conn_to_rank: Dict[socket.socket, int] = {}
        self.all_known_ranks: Set[int] = set()
        self.last_seen: Dict[int, float] = {}
        self.conn_lock = threading.Lock()    # Protects all connection‑related structures

        # Networking
        self.recv_buffers: Dict[socket.socket, bytearray] = defaultdict(bytearray)
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind((self.host, self.port))
        self.send_lock = threading.Lock()    # Protects sendall calls

        # Message Handling
        self.inbox: Dict[str, List[JanusMessage]] = defaultdict(list)
        self.inbox_lock = threading.Lock()   # Protects inbox
        self.event_queue = queue.Queue(maxsize=2000)

        self.stop_event = threading.Event()
        self._ready_event = threading.Event()
        self.polling_thread: Optional[threading.Thread] = None
        self.master_thread: Optional[threading.Thread] = None

        print(f"[Master] Messenger started on {host}:{port}. Expecting {expected_slaves} slaves.")

    # ------------------------------------------------------------------
    # Lifecycle Control
    # ------------------------------------------------------------------

    def start(self):
        """Start the background polling thread and the slave connection acceptor."""
        self.start_polling()
        self.master_thread = threading.Thread(target=self.wait_for_slaves, daemon=False)
        self.master_thread.start()

    def start_polling(self):
        """Launch the I/O multiplexing thread (daemon)."""
        self.polling_thread = threading.Thread(target=self._run_polling, daemon=True)
        self.polling_thread.start()
        print("[Master] Polling thread started.")

    def close(self):
        """Gracefully shut down the master, closing all connections and threads."""
        self.stop_event.set()
        # Wait for the acceptor thread to finish
        if self.master_thread and self.master_thread.is_alive():
            self.master_thread.join(timeout=2.0)
        # Wait for the polling thread to finish
        if self.polling_thread and self.polling_thread.is_alive():
            self.polling_thread.join(timeout=1.0)
        # Close listening socket
        try:
            self.server_sock.close()
        except:
            pass
        # Close all slave connections
        with self.conn_lock:
            for conn in self.slave_conns:
                try:
                    conn.close()
                except:
                    pass
        print("[Master] Control plane closed.")

    # ------------------------------------------------------------------
    # Synchronization & Handshake
    # ------------------------------------------------------------------

    def wait_for_slaves(self):
        """
        Accept connections from slaves until the expected number is reached.
        Each connection must send a HANDSHAKE request; master replies with a
        HANDSHAKE response containing the slave's assigned rank and world size.
        The socket is given a short timeout to periodically check stop_event.
        """
        self.server_sock.listen(self.expected_slaves)
        self.server_sock.settimeout(1.0)      # allow periodic exit
        print(f"[Master] Waiting for {self.expected_slaves} slaves...")

        while not self.stop_event.is_set():
            with self.conn_lock:
                if len(self.rank_to_conn) >= self.expected_slaves:
                    self._ready_event.set()
                    break

            try:
                conn, addr = self.server_sock.accept()
            except socket.timeout:
                continue
            except socket.error as e:
                if self.stop_event.is_set():
                    break
                else:
                    raise

            try:
                conn.settimeout(5.0)
                header = self._recv_all(conn, 4)
                body_len = struct.unpack(">I", header)[0]
                if body_len > self.MAX_MESSAGE_SIZE:
                    raise ValueError("Handshake body too large")
                body = self._recv_all(conn, body_len)
                msg = JanusMessage.from_json(body.decode('utf-8'))

                if msg.action == ActionCode.HANDSHAKE:
                    rank = msg.source_rank
                    with self.conn_lock:
                        # Replace old connection if duplicate rank appears
                        if rank in self.rank_to_conn:
                            self._handle_disconnect_unsafe(self.rank_to_conn[rank])
                        # Register the new slave
                        self.rank_to_conn[rank] = conn
                        self.conn_to_rank[conn] = rank
                        self.all_known_ranks.add(rank)
                        self.last_seen[rank] = time.time()
                        self.slave_conns.append(conn)
                        print(f"[Master] Registered slave {rank} from {addr}")

                    # Send handshake ACK
                    ack = JanusMessage(
                        source_rank=-1, target=rank,
                        action=ActionCode.HANDSHAKE, msg_type=MessageType.RESPONSE,
                        request_id=msg.request_id, status=StatusCode.SUCCESS,
                        payload={"status": "ACCEPTED", "rank": rank, "world_size": self.expected_slaves}
                    )
                    with self.send_lock:
                        conn.sendall(self._pack_message(ack))
                    print(f"[Cluster] Slave {rank} joined from {addr}")

                conn.settimeout(None)
            except Exception as e:
                print(f"[Master] Handshake failed: {e}")
                conn.close()

        self.server_sock.settimeout(None)
        print("[Master] All slaves connected. Cluster ready.")

    def wait_until_ready(self, timeout: float) -> bool:
        """
        Block the caller until all slaves have connected or timeout occurs.
        """
        return self._ready_event.wait(timeout=timeout)

    def get_missing_ranks(self) -> List[int]:
        """Get the ranks that have not yet connected, for error diagnosis."""
        with self.conn_lock:
            connected = set(self.rank_to_conn.keys())
        all_expected = set(range(self.expected_slaves))
        return sorted(list(all_expected - connected))

    # ------------------------------------------------------------------
    # Command & Control / Aggregation APIs
    # ------------------------------------------------------------------

    def broadcast(self, msg: JanusMessage):
        """
        Send a message to every connected slave.
        The message is packed once and sent to a snapshot of the current slave list.
        """
        data = self._pack_message(msg)
        with self.conn_lock:
            targets = list(self.slave_conns)       # snapshot under lock
        with self.send_lock:
            for conn in targets:
                try:
                    conn.sendall(data)
                except socket.error:
                    self._handle_disconnect(conn)

    def send_to_rank(self, rank: int, msg: JanusMessage):
        """
        Send a message to a specific slave by rank.
        Returns immediately if the rank is not known.
        """
        with self.conn_lock:
            conn = self.rank_to_conn.get(rank)
        if not conn:
            print(f"[Master] Error: Rank {rank} is offline.")
            return
        data = self._pack_message(msg)
        with self.send_lock:
            try:
                conn.sendall(data)
            except socket.error:
                self._handle_disconnect(conn)

    def collect_responses(self, request_id: str, expected_ranks: List[int], timeout: float = 30.0) -> Dict[int, JanusMessage]:
        """
        Wait for responses from a set of ranks for a given request_id.
        Returns a dictionary mapping rank -> JanusMessage.
        The inbox entry for this request_id is cleaned up after the call.
        """
        results = {}
        start_t = time.time()
        remaining = set(expected_ranks)

        try:
            while time.time() - start_t < timeout and remaining:
                with self.inbox_lock:
                    if request_id in self.inbox:
                        msgs = self.inbox[request_id]
                        for m in list(msgs):
                            if m.source_rank in remaining:
                                results[m.source_rank] = m
                                remaining.remove(m.source_rank)
                                msgs.remove(m)
                if remaining:
                    # Check if any expected rank went offline
                    with self.conn_lock:
                        still_alive = self.all_known_ranks.copy()
                    dead_ranks = remaining - still_alive
                    if dead_ranks:
                        print(f"[Master] Ranks {dead_ranks} disconnected during collection.")
                        for r in dead_ranks:
                            remaining.remove(r)
                    time.sleep(0.1)   # prevent busy loop
        finally:
            with self.inbox_lock:
                if request_id in self.inbox:
                    del self.inbox[request_id]

        return results

    # ------------------------------------------------------------------
    # I/O Polling Engine
    # ------------------------------------------------------------------

    def _run_polling(self):
        """
        Main loop that uses select to monitor all slave sockets for incoming data.
        It reads available data, reassembles messages, and routes them to the
        appropriate handler (heartbeat, event queue, or inbox).
        """
        while not self.stop_event.is_set():
            with self.conn_lock:
                current_slaves = list(self.slave_conns)
            if not current_slaves:
                time.sleep(0.1)
                continue

            self._check_liveness()

            try:
                readable, _, exceptional = select.select(current_slaves, [], current_slaves, 0.2)
            except (select.error, ValueError):
                continue

            for s in readable:
                messages_to_route = []
                try:
                    data = s.recv(16384)
                    if not data:
                        self._handle_disconnect(s)
                        continue
                    with self.conn_lock:
                        # Append to buffer and extract complete messages
                        self.recv_buffers[s].extend(data)
                        messages_to_route = self._extract_messages_unsafe(s)
                except Exception:
                    self._handle_disconnect(s)
                    continue

                # Route messages outside the lock to avoid deadlock
                for msg in messages_to_route:
                    self._route_message(msg)

            for s in exceptional:
                self._handle_disconnect(s)

            if time.time() % 10 < 0.2:
                self._cleanup_inbox()

    def _extract_messages_unsafe(self, conn: socket.socket) -> List[JanusMessage]:
        """
        Parse the receive buffer for complete messages.
        Assumes caller holds conn_lock. Returns a list of fully parsed messages.
        """
        extracted = []
        buf = self.recv_buffers[conn]
        while len(buf) >= 4:
            body_len = struct.unpack(">I", buf[:4])[0]
            if body_len > self.MAX_MESSAGE_SIZE:
                self._handle_disconnect_unsafe(conn)
                break
            if len(buf) < 4 + body_len:
                break
            body = buf[4:4 + body_len]
            del buf[:4 + body_len]
            try:
                msg = JanusMessage.from_json(body.decode('utf-8'))
                extracted.append(msg)
            except Exception as e:
                print(f"[Master] Message parse error: {e}")
        return extracted

    def _route_message(self, msg: JanusMessage):
        """
        Dispatch a received message to its proper destination:
            - Heartbeat: only updates liveness timestamp
            - Event: placed into event_queue
            - Response: stored in inbox under request_id
        """
        rank = msg.source_rank
        with self.conn_lock:
            self.last_seen[rank] = time.time()

        if msg.action == ActionCode.HEARTBEAT:
            return
        if msg.msg_type == MessageType.EVENT:
            try:
                self.event_queue.put_nowait(msg)
            except queue.Full:
                self.event_queue.get()
                self.event_queue.put(msg)
        else:
            with self.inbox_lock:
                self.inbox[msg.request_id].append(msg)

    # ------------------------------------------------------------------
    # Maintenance & Utilities
    # ------------------------------------------------------------------

    def _check_liveness(self, timeout: float = 30.0):
        """
        Detect slaves that have not been heard from for more than `timeout` seconds.
        Such slaves are evicted.
        """
        now = time.time()
        to_evict = []
        with self.conn_lock:
            for rank, last_t in self.last_seen.items():
                if now - last_t > timeout:
                    to_evict.append(rank)
            for rank in to_evict:
                conn = self.rank_to_conn.get(rank)
                if conn:
                    self._handle_disconnect_unsafe(conn)

    def _handle_disconnect(self, conn: socket.socket):
        """Thread‑safe external entry for disconnection."""
        with self.conn_lock:
            self._handle_disconnect_unsafe(conn)

    def _handle_disconnect_unsafe(self, conn: socket.socket):
        """
        Remove all references to a socket and close it.
        Must be called with conn_lock held.
        """
        rank = self.conn_to_rank.pop(conn, None)
        if rank is not None:
            self.rank_to_conn.pop(rank, None)
            self.all_known_ranks.discard(rank)
            self.last_seen.pop(rank, None)
            print(f"[Cluster] Slave {rank} disconnected.")
        if conn in self.slave_conns:
            self.slave_conns.remove(conn)
        self.recv_buffers.pop(conn, None)
        try:
            conn.close()
        except:
            pass

    def _cleanup_inbox(self, ttl: float = 60.0):
        """
        Periodically remove stale entries from the inbox to prevent unbounded growth.
        """
        now = time.time()
        with self.inbox_lock:
            for req_id in list(self.inbox.keys()):
                msgs = self.inbox[req_id]
                if msgs and now - msgs[0].timestamp > ttl:
                    del self.inbox[req_id]

    def _recv_all(self, conn: socket.socket, length: int) -> bytes:
        """Read exactly `length` bytes from the socket, raising an error on failure."""
        data = b''
        while len(data) < length:
            chunk = conn.recv(length - len(data))
            if not chunk:
                raise ConnectionError("Remote closed")
            data += chunk
        return data

    def _pack_message(self, msg: JanusMessage) -> bytes:
        """Encode a message as a length‑prefixed JSON string."""
        body = msg.to_json().encode('utf-8')
        return struct.pack(">I", len(body)) + body