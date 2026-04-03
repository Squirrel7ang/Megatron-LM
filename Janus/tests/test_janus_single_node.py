import threading
import time
import json
import os
import sys
from unittest.mock import patch

# ------------------------------------------------------------------
# Environment Setup
# ------------------------------------------------------------------
# Ensure the project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.master.orchestrator.scheduler import JanusOrchestrator
from core.slave.messenger import SlaveMessenger
from core.master.parser.cluster_parser import ClusterParser
from core.common.protocol import ActionCode, JanusMessage, MessageType
from core.common.cluster_context import ClusterContext

# ------------------------------------------------------------------
# Slave Process Simulation (Rank 1)
# ------------------------------------------------------------------

def run_slave_process(model_path, cluster_path):
    """Simulates the startup logic of a Slave process."""
    # Give the Master a moment to bind its socket and enter the listening loop
    time.sleep(2)
    
    print("\n" + "-"*30)
    print("[Slave] Initializing Slave Messenger (Rank 1)...")
    print("-"*30)
    
    parser = ClusterParser(cluster_path)
    cluster_ctx = parser.parse()
    
    # Since we are on a single machine, Slave Rank 1 maps to Node 0 metadata
    node_info = cluster_ctx.nodes[0]
    
    slave = SlaveMessenger(
        master_host=cluster_ctx.master_addr,
        master_port=cluster_ctx.master_port,
        rank=0,
        node_info=node_info
    )
    
    # Inject ClusterContext for global rank mapping during network probing
    slave.cluster_context = cluster_ctx
    
    try:
        # Robust connection logic with retries to prevent [Errno 111]
        connected = False
        for attempt in range(5):
            try:
                slave.connect_to_master()
                connected = True
                break
            except ConnectionRefusedError:
                print(f"[Slave] Master not ready (attempt {attempt+1}/5), retrying...")
                time.sleep(2)
        
        if not connected:
            print("[Slave] Failed to connect after multiple retries. Exiting.")
            return

        print(f"[Slave] Successfully connected to Master at {cluster_ctx.master_addr}:{cluster_ctx.master_port}")
        
        # Keep Slave alive until explicitly closed by Master or test ends
        while not slave.stop_event.is_set():
            time.sleep(1)
    except Exception as e:
        print(f"[Slave] Fatal Error: {e}")
    finally:
        slave.close()
        print("[Slave] Connection closed.")

# ------------------------------------------------------------------
# Main Orchestration Test (Rank 0)
# ------------------------------------------------------------------
def main():
    model_spec = "configs/model_spec.json"
    cluster_spec = "configs/cluster_spec.json"
    
    # 1. Initialize Master Orchestrator (Rank 0)
    print("\n" + "="*70)
    print(" [MASTER] Initializing Janus Orchestrator (Rank -1)...")
    print("="*70)
    orchestrator = JanusOrchestrator(model_spec, cluster_spec)
    
    # Adjust timeouts for local test speed
    orchestrator.SLAVE_CONNECT_TIMEOUT_S = 15.0

    try:
        # 2. START MASTER MESSENGER FIRST (Crucial: port must be open before slave connects)
        orchestrator.messenger.start()
        print(f"[Master] Listening on {orchestrator.messenger.host}:{orchestrator.messenger.port}")

        # 3. Start Slave Thread
        slave_thread = threading.Thread(
            target=run_slave_process, 
            args=(model_spec, cluster_spec), 
            daemon=True
        )
        slave_thread.start()
        
        # Phase A: Handshake
        print("\n[Phase A] Waiting for all Slaves to handshake...")
        # Note: Ensure Master's own rank (0) is considered 'online' internally
        orchestrator._wait_for_all_slaves()
        
        # Phase B: Environmental Probing
        print("\n[Phase B] Broadcasting PROBE_ENV...")
        orchestrator._probe_env()
        
        # Phase C: Compute Capability Probing
        print("\n[Phase C] Broadcasting PROBE_COMPUTE...")
        orchestrator._probe_compute()
        
        # Phase D: Parallel Strategy Generation & Analysis
        print("\n[Phase D] Generating Plausible Parallel Strategies...")
        strategies = orchestrator.cluster_context.get_plausible_strategies()
        
        print(f"\n>>> Identified {len(strategies)} viable strategies based on Model & Cluster constraints:")
        for idx, s in enumerate(strategies):
            tp, dp, pp = s
            print(f"    Strategy {idx:02d}: [TP={tp}, DP={dp}, PP={pp}]")
            
            # Briefly inspect communication groups for this specific strategy
            orchestrator.cluster_context.init_parallel_strategy(tp, dp, pp)
            matrix = orchestrator.cluster_context.strategy_matrix[s]
            for dim in ["tp", "dp", "pp"]:
                groups = matrix.communication_groups.get(dim, [])
                if groups:
                    print(f"       |_ Dim {dim.upper()}: Found {len(groups)} comm_groups. Representative: {groups[0]}")

        # Phase E: Topology-Aware Network Probing
        print("\n[Phase E] Executing Topology-Aware Network Probing...")
        orchestrator._probe_network()
        
        # --------------------------------------------------------------
        # FINAL REPORT: ClusterContext Data Summary
        # --------------------------------------------------------------
        print("\n" + "#"*70)
        print(" FINAL CLUSTER CONTEXT REPORT")
        print("#"*70)
        ctx = orchestrator.cluster_context
        print(f"Cluster: {ctx.cluster_name} | Master: {ctx.master_addr}:{ctx.master_port}")
        print(f"Inventory: {len(ctx.nodes)} Node(s) registered.")
        
        for node_id, node in enumerate(ctx.nodes):
            print(f"\n[Node {node_id}] {node.hostname} ({node.ip})")
            print(f"  - Resource: {node.cpu_cores} Cores, {node.sys_mem_gb}GB RAM")
            print(f"  - Network:  {node.nic_type} (RDMA Active: {node.has_rdma})")
            print(f"  - Accelerators: {len(node.gpus)} GPUs detected")
            for gpu in node.gpus:
                print(f"    |_ GPU {gpu.local_id}: {gpu.type} | {gpu.memory_capacity_gb}GB VRAM | {gpu.peak_gemm_tflops} TFLOPS")

        print("\n" + "="*70)
        print(" COMMUNICATION PERFORMANCE MATRIX (TOP STRATEGIES)")
        print("="*70)
        for s in strategies[:2]:
            perf_matrix = ctx.strategy_matrix.get(s)
            if not perf_matrix: continue
            
            print(f"\nTarget Strategy: TP={s[0]}, DP={s[1]}, PP={s[2]}")
            for dim in ["tp", "dp", "pp"]:
                groups = perf_matrix.communication_groups.get(dim, [])
                if not groups: continue
                
                samples = ctx.get_strategic_samples(groups, dim)
                print(f"  Dimension {dim.upper()}:")
                print(f"    - Total Comm Groups: {len(groups)}")
                print(f"    - Strategic Samples: {samples}")
                
                dim_perf = perf_matrix.dimension_performance.get(dim, {})
                for coll_type, data in dim_perf.items():
                    print(f"    - Measured Result ({coll_type}):")
                    print(f"      Latency: {data.latency_us:.2f} us | Bandwidth: {data.bandwidth_gbps:.2f} Gbps")

    except Exception as e:
        print(f"\n[Master] Pipeline interrupted by Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n" + "="*70)
        print(" [SHUTDOWN] Cleaning up system resources...")
        orchestrator.messenger.close()
        print(" Orchestration Test Complete.")
        print("="*70)

if __name__ == "__main__":
    main()