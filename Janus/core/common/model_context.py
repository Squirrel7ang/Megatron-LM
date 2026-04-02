from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class SubComponent:
    """
    Represents internal operators within a complex layer (e.g., QKV projection).
    Fine-grained breakdown for performance estimation and TP (Tensor Parallel) strategy.
    """
    role: str                # e.g., "attn_qkv", "mlp_h_to_4h", "cross_attn_qkv"
    op_type: str             # e.g., "Linear", "LayerNorm", "Softmax"
    weight_shape: List[int]
    tp_split_dim: Optional[int] = None 
    
    # --- Attention Mechanism Metadata ---
    # Defines the visibility: "causal" (GPT), "full" (BERT/Encoder), "cross" (T5 Decoder Cross-Attn)
    attention_mask_type: Optional[str] = None 
    
    # Architecture-specific flags
    attention_type: Optional[str] = None   # e.g., "gqa", "mha"
    num_groups: Optional[int] = None       # Required if attention_type is "gqa"
    fused_activation: Optional[str] = None # e.g., "swiglu", "gelu"

@dataclass
class LayerIR:
    """
    Intermediate Representation of a single model layer or transformer block.
    Used by the Janus Search Engine to map layers to hardware resources.
    """
    layer_id: int
    name: str
    type: str               # e.g., "Embedding", "TransformerBlock", "Linear"
    params: int             # Total parameter count for this layer
    weight_shape: List[int] # Primary weight dimension
    
    # Parallelism hints for Module B (Strategy Search)
    # e.g., {"tp_strategy": "column", "pp_splittable": True}
    parallel_capability: Dict[str, Any] = field(default_factory=dict)
    
    # Detailed list of operators inside this layer
    sub_components: List[SubComponent] = field(default_factory=list)

@dataclass
class SectionContext:
    """
    Represents a logically independent part of the model (e.g., Encoder or Decoder).
    Allows Janus to support heterogeneous architectures like Encoder-Decoder (T5).
    """
    num_layers: int
    hidden_size: int
    ffn_hidden_size: int
    num_attention_heads: int
    num_query_groups: int
    vocab_size: int
    seq_length: int 
    max_position_embeddings: int
    
    # Section-specific features (e.g., Encoder uses LayerNorm while Decoder uses RMSNorm)
    operator_features: Dict[str, Any] = field(default_factory=dict)
    
    # Sequence of LayerIRs generated for this specific section
    layers: List[LayerIR] = field(default_factory=list)

@dataclass
class ModelContext:
    """
    The root container for the model specification.
    Holds global training metadata and multiple architectural sections.
    """
    name: str
    framework: str         # e.g., "Megatron-LM"
    pretrain_path: str
    model_type: str        # Valid: "decode-only", "encoder-only", "encoder-decoder"

    # --- Multi-Section Topology ---
    # decode-only:     {"backbone": SectionContext}
    # encoder-only:    {"backbone": SectionContext}
    # encoder-decoder: {"encoder": SectionContext, "decoder": SectionContext}
    sections: Dict[str, SectionContext] = field(default_factory=dict)
    
    # Global training configurations shared across all sections
    # e.g., {"precision": "bf16", "micro_batch_size": 4}
    training_hyperparams: Dict[str, Any] = field(default_factory=dict)

    def get_flat_topology(self) -> List[LayerIR]:
        """
        Returns a linearized sequence of all layers in logical execution order.
        Crucial for Pipeline Parallelism (PP) split-point calculations.
        """
        flat_layers = []
        # Execution order: Encoder -> Decoder (if exists) -> Backbone (if single-stream)
        for key in ["encoder", "decoder", "backbone"]:
            if key in self.sections:
                flat_layers.extend(self.sections[key].layers)
        return flat_layers