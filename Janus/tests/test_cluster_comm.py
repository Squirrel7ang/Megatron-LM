#!/usr/bin/env python3
"""
Test script for MasterMessenger and SlaveMessenger with DEBUG logs.
Run with: python -m unittest test_cluster_comm.py
"""

import unittest
import threading
import time
import sys
import os
import socket
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

# --- Adjust path to import the real classes ---
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.common.protocol import JanusMessage, ActionCode, StatusCode, MessageType
from core.master.orchestrator.messenger import MasterMessenger
from core.slave.messenger import SlaveMessenger

# --- Mock classes (same as before) ---
@dataclass
class NodeTopology:
    gpu_gpu_dist: Dict = field(default_factory=dict)
    gpu_nic_dist: Dict = field(default_factory=dict)
    cpu_affinity: Dict = field(default_factory=dict)
    numa_affinity: Dict = field(default_factory=dict)

@dataclass
class GPUInfo:
    local_id: int
    pci_bus_id: str = ""
    memory_capacity_gb: float = 0.0
    available_mem_gb: float = 0.0
    temperature_celsius: float = 0.0
    peak_gemm_tflops: float = 0.0
    best_matmul_size: int = 0
    memory_bandwidth_gbps: float = 0.0
    gemm_efficiency: float = 0.0
    ridge_point_flops_per_byte: float = 0.0

@dataclass
class NodeInfo:
    node_id: int
    hostname: str
    ip: str
    gpus: List[GPUInfo] = field(default_factory=list)
    topology: Optional[NodeTopology] = field(default_factory=NodeTopology)
    cpu_cores: int = 0
    sys_mem_gb: float = 0.0
    nic_type: str = ""
    has_rdma: bool = False
    intra_node_comm: Dict = field(default_factory=dict)
    intra_node_topology: Dict = field(default_factory=dict)

class MockClusterContext:
    def __init__(self, num_nodes=0):
        self.nodes = {}
        for i in range(1, num_nodes+1):
            self.nodes[i] = NodeInfo(i, f"node-{i}", "127.0.0.1")
        self.ranks = list(self.nodes.keys())

def tcp_encode(msg: JanusMessage) -> bytes:
    body = msg.to_json().encode('utf-8')
    return len(body).to_bytes(4, 'big') + body

class TestClusterCommunication(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = "127.0.0.1"
        cls.port = 8888
        cls.num_slaves = 2

    def setUp(self):
        self.cluster = MockClusterContext(self.num_slaves)
        for attempt in range(3):
            try:
                self.master = MasterMessenger(
                    host=self.host,
                    port=self.port,
                    rank=0,
                    expected_slaves=self.num_slaves,
                    cluster=self.cluster
                )
                break
            except OSError as e:
                if "Address already in use" in str(e) and attempt < 2:
                    time.sleep(0.5)
                else:
                    raise
        self.master.start()  # starts polling and wait_for_slaves in separate threads
        time.sleep(0.2)
        self.slaves = []

    def tearDown(self):
        for s in self.slaves:
            s.close()
        if hasattr(self, 'master'):
            self.master.close()
        time.sleep(0.2)

    def test_cluster_formation(self):
        for i in range(1, self.num_slaves + 1):
            node_info = NodeInfo(i, f"node-{i}", "127.0.0.1")
            slave = SlaveMessenger(
                master_host=self.host,
                master_port=self.port,
                rank=i,
                node_info=node_info
            )
            slave.connect_to_master()
            self.slaves.append(slave)

        timeout = 5
        start = time.time()
        while len(self.master.rank_to_conn) < self.num_slaves and time.time() - start < timeout:
            time.sleep(0.1)

        self.assertEqual(len(self.master.rank_to_conn), self.num_slaves)

        for slave in self.slaves:
            self.assertTrue(slave.is_registered.wait(timeout=1))
            self.assertIsNotNone(slave.master_rank)
            self.assertEqual(slave.master_rank, 0)

    def test_broadcast_and_collect(self):
        for i in range(1, self.num_slaves + 1):
            node_info = NodeInfo(i, f"node-{i}", "127.0.0.1")
            slave = SlaveMessenger(
                master_host=self.host,
                master_port=self.port,
                rank=i,
                node_info=node_info
            )
            slave.connect_to_master()
            self.slaves.append(slave)

        timeout = 5
        start = time.time()
        while len(self.master.rank_to_conn) < self.num_slaves and time.time() - start < timeout:
            time.sleep(0.1)
        self.assertEqual(len(self.master.rank_to_conn), self.num_slaves)

        req_id = "test_req_1"
        msg = JanusMessage(
            source_rank=0,
            target=-1,
            action=ActionCode.PROBE_ENV,
            msg_type=MessageType.REQUEST,
            request_id=req_id,
            payload={}
        )
        self.master.broadcast(msg)
        time.sleep(0.5)  # Give time for processing

        expected_ranks = list(range(1, self.num_slaves + 1))
        responses = self.master.collect_responses(req_id, expected_ranks, timeout=10.0)

        self.assertEqual(len(responses), self.num_slaves)
        for rank in expected_ranks:
            self.assertIn(rank, responses)
            resp_msg = responses[rank]
            self.assertEqual(resp_msg.action, ActionCode.PROBE_ENV)
            self.assertEqual(resp_msg.status, StatusCode.SUCCESS)
            self.assertEqual(resp_msg.payload.get("node_id"), rank)

    def test_targeted_send(self):
        for i in range(1, self.num_slaves + 1):
            node_info = NodeInfo(i, f"node-{i}", "127.0.0.1")
            slave = SlaveMessenger(
                master_host=self.host,
                master_port=self.port,
                rank=i,
                node_info=node_info
            )
            slave.connect_to_master()
            self.slaves.append(slave)

        timeout = 5
        start = time.time()
        while len(self.master.rank_to_conn) < self.num_slaves and time.time() - start < timeout:
            time.sleep(0.1)

        req_id = "targeted_req"
        msg = JanusMessage(
            source_rank=0,
            target=1,
            action=ActionCode.PROBE_COMPUTE,
            msg_type=MessageType.REQUEST,
            request_id=req_id,
            payload={}
        )
        self.master.send_to_rank(1, msg)
        time.sleep(0.5)

        responses = self.master.collect_responses(req_id, [1], timeout=10.0)
        self.assertEqual(len(responses), 1)
        self.assertIn(1, responses)
        self.assertEqual(responses[1].action, ActionCode.PROBE_COMPUTE)
        self.assertEqual(responses[1].status, StatusCode.SUCCESS)

    def test_timeout_handling(self):
        node_info = NodeInfo(1, "node-1", "127.0.0.1")
        slave = SlaveMessenger(
            master_host=self.host,
            master_port=self.port,
            rank=1,
            node_info=node_info
        )
        slave.connect_to_master()
        self.slaves.append(slave)

        timeout = 5
        start = time.time()
        while len(self.master.rank_to_conn) < 1 and time.time() - start < timeout:
            time.sleep(0.1)

        req_id = "timeout_test"
        msg = JanusMessage(
            source_rank=0,
            target=-1,
            action=ActionCode.PROBE_ENV,
            msg_type=MessageType.REQUEST,
            request_id=req_id,
            payload={}
        )
        self.master.broadcast(msg)
        time.sleep(0.5)

        expected_ranks = [1, 2]
        responses = self.master.collect_responses(req_id, expected_ranks, timeout=2.0)

        self.assertEqual(len(responses), 1)
        self.assertIn(1, responses)
        self.assertEqual(responses[1].action, ActionCode.PROBE_ENV)

        with self.master.inbox_lock:
            self.assertNotIn(req_id, self.master.inbox)

    def test_disconnect_during_collection(self):
        for i in range(1, 3):
            node_info = NodeInfo(i, f"node-{i}", "127.0.0.1")
            slave = SlaveMessenger(
                master_host=self.host,
                master_port=self.port,
                rank=i,
                node_info=node_info
            )
            slave.connect_to_master()
            self.slaves.append(slave)

        timeout = 5
        start = time.time()
        while len(self.master.rank_to_conn) < 2 and time.time() - start < timeout:
            time.sleep(0.1)

        req_id = "disconnect_test"
        msg = JanusMessage(
            source_rank=0,
            target=-1,
            action=ActionCode.PROBE_ENV,
            msg_type=MessageType.REQUEST,
            request_id=req_id,
            payload={}
        )
        self.master.broadcast(msg)

        def delayed_disconnect():
            time.sleep(0.5)
            self.slaves[1].close()

        threading.Thread(target=delayed_disconnect, daemon=True).start()

        expected_ranks = [1, 2]
        responses = self.master.collect_responses(req_id, expected_ranks, timeout=3.0)

        self.assertEqual(len(responses), 2)
        self.assertIn(1, responses)

    def test_large_message_rejection(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.host, self.port))
        handshake = JanusMessage(
            source_rank=99,
            target=0,
            action=ActionCode.HANDSHAKE,
            msg_type=MessageType.REQUEST,
            payload={"version": "test"}
        )
        sock.sendall(tcp_encode(handshake))
        time.sleep(0.1)
        bogus_len = (100 * 1024 * 1024).to_bytes(4, 'big')
        sock.sendall(bogus_len)
        time.sleep(0.5)
        with self.master.conn_lock:
            self.assertNotIn(99, self.master.rank_to_conn)
        sock.close()

if __name__ == "__main__":
    unittest.main(verbosity=2)