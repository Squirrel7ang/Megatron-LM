import unittest
import sys
import os
import json
import tempfile

# Ensure the core module is discoverable
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import core modules
from core.common.cluster_context import ClusterContext, NodeInfo, GPUInfo
from core.common.model_context import ModelContext
from core.master.parser.model_parser import ModelParser

class TestClusterContextStrategies(unittest.TestCase):
    
    def setUp(self):
        """
        Sets up a mocked 8x8 cluster (64 GPUs) and a 175B Model Context.
        """
        # 1. Cluster Setup (8 Nodes, 8 GPUs per node = 64 A100-80GB)
        self.total_nodes = 8
        self.gpus_per_node = 8
        self.total_gpus = self.total_nodes * self.gpus_per_node
        
        nodes = []
        global_id = 0
        for node_id in range(self.total_nodes):
            gpus = []
            for local_id in range(self.gpus_per_node):
                gpus.append(GPUInfo(
                    global_id=global_id,
                    local_id=local_id,
                    pci_bus_id=f"0000:0{node_id}:0{local_id}.0",
                    type="NVIDIA-A100-80GB",
                    memory_capacity_gb=80.0
                ))
                global_id += 1
            
            nodes.append(NodeInfo(
                node_id=node_id,
                hostname=f"worker-{node_id}",
                ip=f"10.0.0.{node_id+1}",
                gpus=gpus
            ))
            
        self.cluster = ClusterContext(
            cluster_name="Mock-8x8-Cluster",
            total_nodes=self.total_nodes,
            master_addr="10.0.0.1",
            master_port=12345,
            gpus_per_node=self.gpus_per_node,
            nodes=nodes
        )

        # 2. Mock 175B Model Config Setup (GPT-3 175B Architecture)
        # Layers: 96, Hidden: 12288, Heads: 96
        self.mock_config = {
            "model_metadata": {
                "name": "GiantModel",
                "framework": "Megatron-LM",
                "pretrain_path": "/home/test/Megatron-LM/pretrain_175b.py",
                "model_type": "decoder-only"
            },
            "decoder": {
                "architecture": {
                    "num_layers": 48,
                    "hidden_size": 12288,
                    "ffn_hidden_size": 49152, # 4 * hidden_size
                    "num_attention_heads": 96,
                    "num_query_groups": 96,   # MHA
                    "vocab_size": 50257,
                    "max_position_embeddings": 2048,
                    "seq_length": 2048,
                    "untie_embeddings_and_output_weights": True
                },
                "operator_features": {
                    "activation_func": "gelu",
                    "position_embedding_type": "learned-absolute",
                    "bias_linear": True,
                    "attention_backend": "flash",
                    "use_flash_attn": True,
                    "normalization": "layernorm"
                }
            },
            "training_hyperparams": {
                "precision": "bf16",
                "distributed_backend": "nccl",
                "num_layers_per_virtual_pipeline_stage": None,
                "data_path": "data/corpus",
                "tokenizer_type": "GPT2BPETokenizer",
                "train_iters": 100000,
                "use_distributed_optimizer": True
            }
        }

        # Create a temp file for ModelParser to read
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            json.dump(self.mock_config, tmp)
            self.config_path = tmp.name

        # 3. Parse Model Context
        parser = ModelParser(self.config_path)
        self.model_context = parser.parse()
        os.unlink(self.config_path)

    def test_get_plausible_strategies(self):
        """
        Validates the generation of strategies WITHOUT model context (pure topology).
        """
        print("\n" + "="*70)
        print(" [LOG] Testing: get_plausible_strategies (Topology Only)")
        print("="*70)
        
        strategies = self.cluster.get_plausible_strategies(preferred_tp=4)
        print(f"Discovered {len(strategies)} topology-valid strategies.")
        self.assertTrue(len(strategies) > 0)

    def test_plausible_strategies_with_model_pruning(self):
        """
        Validates the strategy generation WITH model context pruning.
        This test checks if the 14B model correctly filters out OOM or invalid PP/TP configs.
        """
        print("\n" + "="*70)
        print(" [LOG] Testing: get_plausible_strategies (With Model Pruning)")
        print("="*70)

        # MBS=1 is usually the baseline for memory estimation
        mbs = 1
        strategies = self.cluster.get_plausible_strategies(
            preferred_tp=4, 
            model_ctx=self.model_context,
            mbs=mbs
        )

        print(f"\nModel: {self.model_context.name} ")
        print(f"Discovered {len(strategies)} feasible strategies (after pruning):")
        
        for i, (tp, dp, pp) in enumerate(strategies):
            # Check memory feasibility for each (mocking the internal check results)
            print(f"  #{i+1:02d} -> TP: {tp}, DP: {dp}, PP: {pp}")

        # Assertions: 
        # 1. Total number of strategies should be <= topology-only strategies
        all_topo_strategies = self.cluster.get_plausible_strategies(preferred_tp=4)
        self.assertLessEqual(len(strategies), len(all_topo_strategies))
        
        # 2. Basic validity of 14B model with 16 GPUs
        # For a 14B model on 16x80GB GPUs, most configurations should pass, 
        # but TP=1/PP=1 might be tight depending on distributed optimizer settings.
        self.assertTrue(len(strategies) > 0)

    def test_megatron_grouping_logic(self):
        """
        Validates grouping logic for a specific pruned strategy.
        """
        tp, dp, pp = 4, 2, 8
        print("\n" + "="*70)
        print(f" [LOG] Testing: Megatron Grouping Logic (TP={tp}, DP={dp}, PP={pp})")
        print("="*70)
        
        # We can now link the performance object to the model context
        strategy_perf = self.cluster.init_parallel_strategy(tp=tp, pp=pp, dp=dp)
        
        # Mocking the memory evaluation result that Janus would perform
        if self.model_context:
            mem_info = self.model_context.evaluate_strategy_memory(tp, pp, dp, gpus_per_node=self.cluster.gpus_per_node, gpu_mem_limit_gb=80, mbs=1)
            print(f"[Model Memory] Estimated Peak: {mem_info.get('peak_mem_gb', 'N/A')} GB per GPU")

        self.assertEqual(strategy_perf.communication_groups["tp"][0], [0, 1, 2, 3])

    def test_parser_accuracy(self):
        """
        Directly validates the ModelParser IR construction.
        """
        print("\n" + "="*70)
        print(" [LOG] Testing: ModelParser IR Construction")
        print("="*70)
        
        ctx = self.model_context
        print(f"Model Name: {ctx.name}")
        print(f"Model Type: {ctx.model_type}")
        
        # Validate backbone (decoder-only)
        self.assertIn("backbone", ctx.sections)
        backbone = ctx.sections["backbone"]
        
        # Check Layer counts: 16 layers + 1 embedding + 1 lm_head = 18 layers in IR
        expected_layers = self.mock_config["decoder"]["architecture"]["num_layers"] + 2
        print(f"Layers in IR: {len(backbone.layers)} (Expected {expected_layers})")
        self.assertEqual(len(backbone.layers), expected_layers)
        

if __name__ == '__main__':
    unittest.main(verbosity=1)