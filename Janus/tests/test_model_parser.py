import unittest
import json
import os
import sys
import tempfile

# Ensure the core module is discoverable
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.master.parser.model_parser import ModelParser
from core.common.model_context import ModelContext

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
        Helper to generate model configuration files aligned with the Janus IR structure.
        The architecture and operator_features are nested to match FM9G's real-world JSON.
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
            if "architecture" in arch_overrides:
                architecture.update(arch_overrides["architecture"])
            if "operator_features" in arch_overrides:
                operator_features.update(arch_overrides["operator_features"])

        config = {
            "model_metadata": {
                "name": name,
                "framework": "Megatron-LM",
                "pretrain_path": f"/home/test/Megatron-LM/pretrain_{name.lower()}.py",
                "model_type": model_type
            },
            "training_hyperparams": {
                "precision": "bf16",
                "data_path": "data/wikitext-2/corpus",
                "tokenizer_type": "SentencePieceTokenizer",
                "tokenizer_model": "tokenizer.model",
                "train_iters": 1000,
                "use_distributed_optimizer": True
            }
        }

        # Nesting architecture based on model type to align with ModelParser's logic
        if model_type == "decoder-only":
            config["decoder"] = {"architecture": architecture, "operator_features": operator_features}
        elif model_type == "encoder-only":
            config["encoder"] = {"architecture": architecture, "operator_features": operator_features}
        elif model_type == "encoder-decoder":
            config["encoder"] = {"architecture": architecture, "operator_features": operator_features}
            config["decoder"] = {"architecture": architecture, "operator_features": operator_features}

        file_path = os.path.join(self.test_dir, f"{name.lower()}_config.json")
        with open(file_path, "w") as f:
            json.dump(config, f)
        return file_path

    def test_fm9g_parameter_count(self):
        """
        Verify that the FM9G model configuration yields the expected parameter scale.
        Logic: Summing up params from all sections in ModelContext.
        """
        path = self._create_config("FM9G", "decoder-only")
        parser = ModelParser(path)
        ctx = parser.parse()
        
        # Calculate total params from all sections (in Billions)
        total_params = sum(section.total_params for section in ctx.sections.values())
        params_b = total_params / 1e9
        
        print(f"\n[Test] FM9G Estimated Parameters: {params_b:.3f} B")
        
        # Expected value for FM9G-14B architecture with 16 layers is ~4.26B
        self.assertAlmostEqual(params_b, 4.26, delta=0.1)

    def test_decoder_only_topology(self):
        """Verify parsing of GPT-style (decoder-only) Multi-Section IR."""
        overrides = {"architecture": {"num_layers": 12}}
        path = self._create_config("GPT_Small", "decoder-only", arch_overrides=overrides)
        parser = ModelParser(path)
        ctx = parser.parse()
        
        self.assertEqual(ctx.model_type, "decoder-only")
        self.assertIn("backbone", ctx.sections)
        
        # Layers: 1 (Embed) + 12 (TransformerBlocks) + 1 (LM Head) = 14
        layers = ctx.sections["backbone"].layers
        self.assertEqual(len(layers), 14)
        self.assertEqual(layers[0].type, "Embedding")
        self.assertEqual(layers[-1].type, "Linear")
        self.assertEqual(layers[1].type, "TransformerBlock")

    def test_encoder_decoder_dual_stream(self):
        """Verify T5-style (encoder-decoder) parsing creates two distinct sections."""
        overrides = {"architecture": {"num_layers": 6}}
        path = self._create_config("T5_Small", "encoder-decoder", arch_overrides=overrides)
        parser = ModelParser(path)
        ctx = parser.parse()
        
        self.assertEqual(ctx.model_type, "encoder-decoder")
        self.assertIn("encoder", ctx.sections)
        self.assertIn("decoder", ctx.sections)
        
        # Encoder: 1 (Embed) + 6 (Blocks) = 7
        # Decoder: 1 (Embed) + 6 (Blocks) + 1 (Head) = 8
        self.assertEqual(len(ctx.sections["encoder"].layers), 7)
        self.assertEqual(len(ctx.sections["decoder"].layers), 8)
        
        # Verify Cross-Attention presence in Decoder but not in Encoder
        dec_block = ctx.sections["decoder"].layers[1]
        roles = [comp.role for comp in dec_block.sub_components]
        self.assertIn("cross_attn_qkv", roles)
        
        enc_block = ctx.sections["encoder"].layers[1]
        roles_enc = [comp.role for comp in enc_block.sub_components]
        self.assertNotIn("cross_attn_qkv", roles_enc)

    def test_subcomponent_details(self):
        """Verify that SubComponents carry correct parallel and operator metadata."""
        path = self._create_config("FM9G_Detail", "decoder-only")
        ctx = ModelParser(path).parse()
        
        # Pick the first TransformerBlock
        block = ctx.sections["backbone"].layers[1]
        self.assertEqual(block.type, "TransformerBlock")
        
        # Check QKV component (should be TP split on dim 0 for column parallel)
        qkv = next(c for c in block.sub_components if c.role == "self_attn_qkv")
        self.assertEqual(qkv.tp_split_dim, 0)
        self.assertEqual(qkv.attention_type, "gqa") # 28 heads / 4 groups
        
        # Check MLP component (SwiGLU should have 2x multiplier in weight_shape)
        mlp_gate = next(c for c in block.sub_components if c.role == "mlp_h_to_4h")
        self.assertEqual(mlp_gate.fused_activation, "swiglu")
        # ffn_hidden_size (18944) * 2 = 37888
        self.assertEqual(mlp_gate.weight_shape[1], 37888)

    def test_validation_missing_metadata(self):
        """Verify the parser detects missing mandatory metadata fields."""
        path = self._create_config("Broken_Meta", "decoder-only")
        with open(path, "r") as f:
            data = json.load(f)
        
        del data["model_metadata"]["framework"]
        
        with open(path, "w") as f:
            json.dump(data, f)
            
        with self.assertRaisesRegex(ValueError, "Metadata field 'framework' is mandatory"):
            ModelParser(path)

    def test_validation_type_mismatch(self):
        """Verify the parser detects mismatch between model_type and existing sections."""
        # Specifying decoder-only but providing an 'encoder' node (via helper)
        path = self._create_config("Mismatch", "decoder-only")
        with open(path, "r") as f:
            data = json.load(f)
        
        # Force add an encoder section
        data["encoder"] = data["decoder"]
        
        with open(path, "w") as f:
            json.dump(data, f)
            
        with self.assertRaisesRegex(ValueError, "decoder-only.*encoder.*found"):
            ModelParser(path)

if __name__ == "__main__":
    unittest.main(verbosity=2)