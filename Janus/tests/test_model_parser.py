import unittest
import json
import os
import sys

# Ensure the core module is discoverable
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.master.parser.model_parser import ModelParser

class TestModelParser(unittest.TestCase):
    def setUp(self):
        """Initialize temporary directory for test configurations."""
        self.test_dir = "tests/temp_configs"
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        """Clean up all temporary files created during testing."""
        if os.path.exists(self.test_dir):
            for f in os.listdir(self.test_dir):
                os.remove(os.path.join(self.test_dir, f))
            os.rmdir(self.test_dir)

    def _create_config(self, name: str, model_type: str, arch_overrides: dict = None) -> str:
        """
        Helper to generate model configuration files aligned with the FM9G structure.
        The 'architecture' and 'operator_features' are nested under the specific model type node.
        """
        # Default FM9G-style architecture settings
        architecture = {
            "num_layers": 16,
            "hidden_size": 3584,
            "ffn_hidden_size": 18944,
            "num_attention_heads": 28,
            "num_query_groups": 4,
            "vocab_size": 73448,
            "max_position_embeddings": 32768,
            "seq_length": 4096,
            "untie_embeddings_and_output_weights": True
        }
        
        operator_features = {
            "activation_func": "swiglu",
            "position_embedding_type": "rotary",
            "bias_linear": False,
            "normalization": "rmsnorm"
        }

        # Apply specific overrides if provided
        if arch_overrides:
            architecture.update(arch_overrides.get("architecture", {}))
            operator_features.update(arch_overrides.get("operator_features", {}))

        config = {
            "model_metadata": {
                "name": name,
                "framework": "Megatron-LM",
                "pretrain_path": f"/home/test/Megatron-LM/pretrain_{name.lower()}.py",
                "model_type": model_type
            },
            "training_hyperparams": {
                "precision": "bf16",
                "distributed_backend": "nccl",
                "num_layers_per_virtual_pipeline_stage": 4,
                "data_path": "data/wikitext-2/corpus",
                "tokenizer_type": "SentencePieceTokenizer",
                "train_iters": 1000,
                "use_distributed_optimizer": True
            }
        }

        # Nesting architecture based on model type to align with user's structure
        if model_type == "decoder-only":
            config["decoder"] = {"architecture": architecture, "operator_features": operator_features}
        elif model_type == "encoder-only":
            config["encoder"] = {"architecture": architecture, "operator_features": operator_features}
        elif model_type == "encoder-decoder":
            # For T5-style, we apply the same to both for testing simplicity
            config["encoder"] = {"architecture": architecture, "operator_features": operator_features}
            config["decoder"] = {"architecture": architecture, "operator_features": operator_features}

        file_path = os.path.join(self.test_dir, f"{name.lower()}_config.json")
        with open(file_path, "w") as f:
            json.dump(config, f)
        return file_path

    def test_fm9g_parameter_count(self):
        """
        Verify that the FM9G model configuration yields approximately 4.2B parameters.
        Calculation: Embedding + 16 Layers (GQA + SwiGLU) + Untied LM Head.
        """
        path = self._create_config("FM9G", "decoder-only")
        parser = ModelParser(path)
        ctx = parser.parse()
        
        mem_info = parser.estimate_model_static_memory(ctx)
        params_b = mem_info["total_params_billions"]
        
        print(f"\n[Test] FM9G Estimated Parameters: {params_b:.3f} B")
        
        # Expected value is ~4.25-4.26B based on the provided dimensions
        self.assertAlmostEqual(params_b, 4.26, delta=0.1)

    def test_decoder_only_architecture(self):
        """Verify parsing of GPT-style (decoder-only) models."""
        # Use a smaller override to keep the test focused on structure
        overrides = {"architecture": {"num_layers": 12}}
        path = self._create_config("GPT_Small", "decoder-only", arch_overrides=overrides)
        parser = ModelParser(path)
        ctx = parser.parse()
        
        self.assertEqual(ctx.model_type, "decoder-only")
        self.assertIn("backbone", ctx.sections)
        
        # Topology: 1 (Embed) + 12 (Blocks) + 1 (Head) = 14
        flat_topology = ctx.get_flat_topology()
        self.assertEqual(len(flat_topology), 14)

    def test_encoder_only_architecture(self):
        """Verify parsing of BERT-style (encoder-only) models."""
        overrides = {"architecture": {"num_layers": 12}}
        path = self._create_config("BERT_Base", "encoder-only", arch_overrides=overrides)
        parser = ModelParser(path)
        ctx = parser.parse()
        
        self.assertEqual(ctx.model_type, "encoder-only")
        self.assertIn("backbone", ctx.sections)
        
        # Topology: 1 (Embed) + 12 (Blocks) = 13
        flat_topology = ctx.get_flat_topology()
        self.assertEqual(len(flat_topology), 13)

    def test_encoder_decoder_architecture(self):
        """Verify complex T5-style (encoder-decoder) parsing."""
        overrides = {"architecture": {"num_layers": 6}}
        path = self._create_config("T5_Small", "encoder-decoder", arch_overrides=overrides)
        parser = ModelParser(path)
        ctx = parser.parse()
        
        self.assertEqual(ctx.model_type, "encoder-decoder")
        self.assertIn("encoder", ctx.sections)
        self.assertIn("decoder", ctx.sections)
        
        # Encoder: 1 (Embed) + 6 (Blocks) = 7
        # Decoder: 1 (Embed) + 6 (Blocks) + 1 (Head) = 8
        # Total: 15
        flat_topology = ctx.get_flat_topology()
        self.assertEqual(len(flat_topology), 15)

    def test_memory_and_flops_across_types(self):
        """Ensure performance estimation works regardless of multi-section topology."""
        path = self._create_config("Benchmark_Model", "decoder-only")
        parser = ModelParser(path)
        ctx = parser.parse()
        
        mem_info = parser.estimate_model_static_memory(ctx)
        flops = parser.estimate_model_total_flops(ctx)
        
        self.assertGreater(mem_info["total_params_billions"], 0)
        self.assertGreater(flops, 0)

    def test_validation_mismatch(self):
        """Verify the parser detects invalid section configurations."""
        bad_config_path = self._create_config("Broken_Config", "decoder-only")
        
        with open(bad_config_path, "r") as f:
            data = json.load(f)
        
        # Remove a mandatory hyperparameter to trigger validation error
        del data["training_hyperparams"]["precision"]
        
        with open(bad_config_path, "w") as f:
            json.dump(data, f)
            
        with self.assertRaises(ValueError):
            ModelParser(bad_config_path)

if __name__ == "__main__":
    unittest.main()