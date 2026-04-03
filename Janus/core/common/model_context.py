from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

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
    attention_mask_type: Optional[str] = None 
    attention_type: Optional[str] = None   # e.g., "gqa", "mha"
    num_groups: Optional[int] = None       
    fused_activation: Optional[str] = None 

@dataclass
class LayerIR:
    """
    Intermediate Representation of a single model layer or transformer block.
    """
    layer_id: int
    name: str
    type: str               # e.g., "Embedding", "TransformerBlock", "Linear"
    params: int             # Total parameter count
    weight_shape: List[int] 
    parallel_capability: Dict[str, Any] = field(default_factory=dict)
    sub_components: List[SubComponent] = field(default_factory=list)

@dataclass
class SectionContext:
    """
    Represents a logically independent part of the model (Encoder/Decoder).
    """
    num_layers: int
    hidden_size: int
    ffn_hidden_size: int
    num_attention_heads: int
    num_query_groups: int
    vocab_size: int
    seq_length: int 
    max_position_embeddings: int
    operator_features: Dict[str, Any] = field(default_factory=dict)
    layers: List[LayerIR] = field(default_factory=list)

    @property
    def total_params(self) -> int:
        """
        Calculates the total parameter count for this section 
        by summing up all constituent layers in the IR.
        """
        return sum(layer.params for layer in self.layers)

    def estimate_activation_memory(self, mbs: int, tp_size: int, bytes_per_elem: int, sp_enabled: bool = False) -> float:
        """
        Precise Attention & TP-sharding Model with FlashAttention support.
        """
        s = self.seq_length
        b = mbs
        h = self.hidden_size
        num_heads = self.num_attention_heads
        
        # 1. Base Activations (Linear/MLP inputs)
        # Some activations are sharded (TP), but residuals and LayerNorm inputs often remain replicated.
        tp_sharded_ratio = 0.7  
        # Empirical standard for transformer block (around 34 tensors per layer)
        base_act_per_layer = s * b * h * 34 * bytes_per_elem
        
        if sp_enabled:
            # Sequence Parallel shards almost all activations along sequence dim
            base_mem = base_act_per_layer / tp_size
        else:
            base_mem = (base_act_per_layer * tp_sharded_ratio / tp_size) + (base_act_per_layer * (1 - tp_sharded_ratio))
            
        # 2. Attention Score Maps
        # flash_attn significantly reduces memory from O(S^2) to O(S)
        use_flash_attn = self.operator_features.get("use_flash_attn", True)
        
        if use_flash_attn:
            # FlashAttention: Memory is O(S) per head (storing logsumexp, etc.)
            # Factor 8 is a heuristic for additional metadata stored per token per head
            attn_map_mem = (b * num_heads * s * bytes_per_elem * 8) / tp_size
        else:
            # Naive: O(S^2) - The main OOM culprit for long sequences
            attn_map_mem = (b * num_heads * (s ** 2) * bytes_per_elem) / tp_size 
        
        return base_mem + attn_map_mem

    def estimate_block_params(self, has_cross_attn: bool = False) -> int:
        """
        Calculates the parameter count of a single Transformer block 
        based on the section's architectural dimensions.
        """
        total_params = 0
        h = self.hidden_size
        f = self.ffn_hidden_size
        n_heads = self.num_attention_heads
        n_groups = self.num_query_groups
        head_dim = h // n_heads
        
        # 1. Self-Attention
        # QKV projection: hidden_size * (num_heads + 2 * num_groups) * head_dim
        # Note: (n_heads * head_dim) is Q, (2 * n_groups * head_dim) is K and V
        qkv_dim = (n_heads + 2 * n_groups) * head_dim
        total_params += h * qkv_dim
        # Output projection: hidden_size * hidden_size
        total_params += h * h
        
        # 2. Cross-Attention (Optional, e.g., for Decoder in Encoder-Decoder)
        if has_cross_attn:
            # Typically Cross-Attn uses full MHA, but we follow the provided logic
            total_params += h * qkv_dim
            total_params += h * h
            # Extra LayerNorm for cross-attention
            total_params += h 
        
        # 3. MLP (Feed-Forward Network)
        act_func = self.operator_features.get("activation_func", "gelu").lower()
        # SwiGLU requires an extra gate projection (h -> ffn_h)
        mlp_multiplier = 2 if act_func == "swiglu" else 1
        total_params += h * (f * mlp_multiplier)
        total_params += f * h
        
        # 4. Layer Normalizations (Input norm and Post-attn norm)
        # Assuming weight only (or weight + bias, but standard IR focuses on weights)
        total_params += 2 * h
        
        return int(total_params)

@dataclass
class ModelContext:
    """
    Root container for model specification and parallel strategy evaluation.
    """
    name: str
    framework: str
    pretrain_path: str
    model_type: str
    sections: Dict[str, SectionContext] = field(default_factory=dict)
    training_hyperparams: Dict[str, Any] = field(default_factory=dict)

    def get_bytes_per_elem(self) -> int:
        """Returns 2 for mixed-precision (bf16/fp16) and 4 for fp32."""
        precision = self.training_hyperparams.get("precision", "bf16").lower()
        if precision in ["bf16", "fp16", "mixed"]:
            return 2
        return 4

    def get_flat_topology(self) -> List[LayerIR]:
        flat_layers = []
        for key in ["encoder", "decoder", "backbone"]:
            if key in self.sections:
                flat_layers.extend(self.sections[key].layers)
        return flat_layers

    @property
    def total_params(self) -> int:
        """
        Sums up parameters across all sections (Encoder, Decoder, Backbone).
        """
        return sum(section.total_params for section in self.sections.values())

    def get_total_params_billions(self) -> float:
        """
        Returns the total parameter count formatted in Billions (B) for 
        readability and high-level benchmarking.
        """
        return self.total_params / 1e9

    def _estimate_max_stage_cost(self, pp: int, tp: int, mbs: int) -> Tuple[float, int]:
        """
        Stage Cost = Alpha * Params + Beta * Activations.
        Balanced PP slicing based on real memory pressure.
        """
        flat_layers = self.get_flat_topology()
        sp_enabled = self.training_hyperparams.get("sequence_parallel", False)
        bytes_per_elem = self.get_bytes_per_elem()
        
        layer_costs = []
        for layer in flat_layers:
            parent_sec = list(self.sections.values())[0] 
            for sec in self.sections.values():
                if layer in sec.layers:
                    parent_sec = sec
                    break
            
            p_cost = (layer.params * bytes_per_elem) / (1024**3)
            a_cost = parent_sec.estimate_activation_memory(mbs, tp, bytes_per_elem, sp_enabled) / (1024**3)
            layer_costs.append(p_cost + a_cost)

        if pp == 1:
            return sum(layer_costs), len(flat_layers)

        target_cost = sum(layer_costs) / pp
        max_cost, max_layers = 0.0, 0
        current_cost, current_layers = 0.0, 0

        for cost in layer_costs:
            current_cost += cost
            current_layers += 1
            if current_cost >= target_cost:
                max_cost = max(max_cost, current_cost)
                max_layers = max(max_layers, current_layers)
                current_cost, current_layers = 0.0, 0
        
        if current_cost > 0:
            max_cost = max(max_cost, current_cost)
            max_layers = max(max_layers, current_layers)

        return max_cost, max_layers

    def evaluate_strategy_memory(self, tp: int, pp: int, dp: int, mbs: int, gpus_per_node: int, gpu_mem_limit_gb: float) -> Dict[str, Any]:
        """
        High-fidelity memory model with structural constraints and precision awareness.
        """
        # --- 1. Strict Structural Constraints (Fail-fast) ---
        if tp > gpus_per_node:
            return {"feasible": False, "risk_level": "OOM", "reason": f"TP ({tp}) crosses node boundary"}
        
        for name, sec in self.sections.items():
            if sec.hidden_size % tp != 0:
                return {"feasible": False, "risk_level": "OOM", "reason": f"{name}: hidden_size ({sec.hidden_size}) % TP != 0"}
            
            if sec.num_attention_heads % tp != 0:
                return {"feasible": False, "risk_level": "OOM", "reason": f"{name}: num_heads ({sec.num_attention_heads}) % TP != 0"}
            
            is_gqa = sec.operator_features.get("attention_type", "").lower() == "gqa" or sec.num_query_groups < sec.num_attention_heads
            if is_gqa and sec.num_query_groups % tp != 0:
                return {"feasible": False, "risk_level": "OOM", "reason": f"{name}: GQA query_groups % TP != 0"}
        
        bytes_per_elem = self.get_bytes_per_elem()

        # --- 2. Parameter & Weight Memory ---
        total_params = sum(l.params for l in self.get_flat_topology())
        avg_params_per_stage = total_params / pp
        
        tp_overhead_ratio = 0.05 
        sharded_params = (avg_params_per_stage * (1 - tp_overhead_ratio)) / tp
        replicated_params = avg_params_per_stage * tp_overhead_ratio
        effective_params_per_gpu = sharded_params + replicated_params
        
        weight_mem_gb = (effective_params_per_gpu * bytes_per_elem) / (1024**3)

        # --- 3. Optimizer Memory ---
        use_dist_opt = self.training_hyperparams.get("use_distributed_optimizer", True)
        # Optimizer states are typically fp32 (4 bytes). 
        # Standard Adam: 4 (param) + 4 (grad) + 4 (m1) + 4 (m2) = 16 bytes per sharded param? 
        # Simplified multiplier for Distributed Optimizer:
        opt_multiplier = 12 / dp if use_dist_opt else 12
        opt_mem_gb = (effective_params_per_gpu * opt_multiplier) / (1024**3)

        # --- 4. Activation Memory ---
        _, max_stage_layers = self._estimate_max_stage_cost(pp, tp, mbs)
        rep_section = list(self.sections.values())[0]
        sp_enabled = self.training_hyperparams.get("sequence_parallel", False)
        
        act_per_layer = rep_section.estimate_activation_memory(mbs, tp, bytes_per_elem, sp_enabled)
        num_inflight = min(pp + 1, mbs) 
        total_act_mem_gb = (max_stage_layers * act_per_layer * num_inflight * 1.3) / (1024**3)
        
        if self.training_hyperparams.get("activation_checkpointing", False):
            total_act_mem_gb *= 0.55 

        # --- 5. Workspace & Feasibility ---
        workspace_gb = 2.0 + (rep_section.hidden_size / 4096) * (tp * 0.2) + (dp * 0.05)
        estimated_peak_gb = weight_mem_gb + opt_mem_gb + total_act_mem_gb + workspace_gb
        utilization = estimated_peak_gb / gpu_mem_limit_gb

        if utilization > 1.0: risk = "OOM"; feasible = False
        elif utilization > 0.90: risk = "dangerous"; feasible = True
        elif utilization > 0.75: risk = "tight"; feasible = True
        else: risk = "safe"; feasible = True

        mem_breakdown = {
            "weights": round(weight_mem_gb, 2),
            "optimizer": round(opt_mem_gb, 2),
            "activation": round(total_act_mem_gb, 2),
            "workspace": round(workspace_gb, 2)
        }

        return {
            "feasible": feasible,
            "risk_level": risk,
            "bottleneck": max(mem_breakdown, key=mem_breakdown.get),
            "utilization": round(utilization, 3),
            "estimated_peak_gb": round(estimated_peak_gb, 2),
            "memory_breakdown_gb": mem_breakdown
        }