import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from core.common.model_context import SubComponent, LayerIR, SectionContext, ModelContext


class ModelParser:
    """
    Parses model specifications from a structured JSON and constructs 
    the Multi-Section ModelContext IR for the Janus search engine.
    """
    REQUIRED_METADATA = ["name", "framework", "pretrain_path", "model_type"]
    REQUIRED_ARCH = [
        "num_layers", "hidden_size", "ffn_hidden_size", 
        "num_attention_heads", "vocab_size"
    ]
    REQUIRED_TRAINING = ["precision", "data_path", "tokenizer_type", "train_iters"]

    def __init__(self, config_path: str):
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.raw_config = json.load(f)
        self._validate_config()

    def _validate_config(self):
        """
        Comprehensive validation of model type and section consistency.
        Raises ValueError for architectural mismatches.
        """
        if "model_metadata" not in self.raw_config:
            raise ValueError("[Config Error] Missing required section: model_metadata")
            
        meta = self.raw_config["model_metadata"]
        for field_name in self.REQUIRED_METADATA:
            if not meta.get(field_name):
                raise ValueError(f"[Config Error] Metadata field '{field_name}' is mandatory.")

        m_type = meta["model_type"].lower()
        valid_types = ["decoder-only", "encoder-only", "encoder-decoder"]
        if m_type not in valid_types:
            raise ValueError(f"[Config Error] 'model_type' must be one of {valid_types}.")

        has_encoder = "encoder" in self.raw_config
        has_decoder = "decoder" in self.raw_config

        if m_type == "decoder-only" and has_encoder:
            raise ValueError("[Config Error] 'decoder-only' type specified, but an 'encoder' section was found.")
        if m_type == "encoder-only" and has_decoder:
            raise ValueError("[Config Error] 'encoder-only' type specified, but a 'decoder' section was found.")
        if m_type == "encoder-decoder" and not (has_encoder and has_decoder):
            raise ValueError("[Config Error] 'encoder-decoder' type requires BOTH 'encoder' and 'decoder' sections.")

        train = self.raw_config.get("training_hyperparams", {})
        for field_name in self.REQUIRED_TRAINING:
            if field_name not in train or train[field_name] is None:
                raise ValueError(f"[Config Error] Training field '{field_name}' is mandatory.")

        t_type = train.get("tokenizer_type")
        if "SentencePiece" in str(t_type) and not train.get("tokenizer_model"):
            print("--- WARNING ---")
            print(f"Tokenizer type '{t_type}' usually requires 'tokenizer_model' in Megatron-LM.")
            print("----------------")

        if "use_distributed_optimizer" not in train:
            print("[Warning] 'use_distributed_optimizer' not specified. Defaulting to True for memory estimation.")

    def _get_arch_val(self, key: str, default: Any = None, config_node: Dict = None) -> Any:
        """Helper to safely extract architecture values, handling hyphens and underscores."""
        if config_node is None:
            config_node = self.raw_config
            
        arch = config_node.get("architecture", {})
        val = arch.get(key.replace("-", "_"))
        if val is None:
            val = arch.get(key.replace("_", "-"))
        return val if val is not None else default

    def _create_section_context(self, config_node: Dict, default_vocab: int) -> SectionContext:
        """Initializes a SectionContext from a specific node in the config dict."""
        ops = config_node.get("operator_features", self.raw_config.get("operator_features", {}))
        heads = self._get_arch_val("num_attention_heads", config_node=config_node)
        
        # Extract sequence length and max position embeddings directly into the section context
        seq_len = self._get_arch_val("seq_length", default=2048, config_node=config_node)
        max_pos = self._get_arch_val("max_position_embeddings", default=seq_len, config_node=config_node)
        
        return SectionContext(
            num_layers=self._get_arch_val("num_layers", config_node=config_node),
            hidden_size=self._get_arch_val("hidden_size", config_node=config_node),
            ffn_hidden_size=self._get_arch_val("ffn_hidden_size", config_node=config_node),
            num_attention_heads=heads,
            num_query_groups=self._get_arch_val("num_query_groups", default=heads, config_node=config_node),
            vocab_size=self._get_arch_val("vocab_size", default=default_vocab, config_node=config_node),
            seq_length=seq_len,
            max_position_embeddings=max_pos,
            operator_features=ops
        )

    def parse(self) -> ModelContext:
        """Constructs the root ModelContext and populates independent SectionContexts."""
        meta = self.raw_config["model_metadata"]
        training = self.raw_config["training_hyperparams"]
        m_type = meta["model_type"].lower()
        
        context = ModelContext(
            name=meta["name"],
            framework=meta["framework"],
            pretrain_path=meta["pretrain_path"],
            model_type=m_type,
            training_hyperparams=training
        )

        global_vocab = self._get_arch_val("vocab_size", config_node=self.raw_config)
        layer_counter = 0

        # 1. Single-Stream Architecture (Backbone only)
        if m_type in ["decoder-only", "encoder-only"]:
            node_key = "decoder" if m_type == "decoder-only" else "encoder"
            target_node = self.raw_config.get(node_key, self.raw_config)
            section_ctx = self._create_section_context(target_node, global_vocab)
            is_decoder = (m_type == "decoder-only")
            
            section_ctx.layers, layer_counter = self._build_section_topology(
                section=section_ctx,
                config_node=target_node,
                prefix="backbone",
                mask_type="causal" if is_decoder else "full",
                has_cross_attn=False,
                start_layer_id=layer_counter,
                include_embedding=True,
                include_lm_head=is_decoder
            )
            context.sections["backbone"] = section_ctx

        # 2. Dual-Stream Architecture (Encoder + Decoder)
        elif m_type == "encoder-decoder":
            # Encoder Section
            enc_node = self.raw_config.get("encoder", self.raw_config)
            enc_ctx = self._create_section_context(enc_node, global_vocab)
            
            enc_ctx.layers, layer_counter = self._build_section_topology(
                section=enc_ctx,
                config_node=enc_node,
                prefix="encoder",
                mask_type="full",
                has_cross_attn=False,
                start_layer_id=layer_counter,
                include_embedding=True,
                include_lm_head=False
            )
            context.sections["encoder"] = enc_ctx
            
            # Decoder Section
            dec_node = self.raw_config.get("decoder", self.raw_config)
            dec_ctx = self._create_section_context(dec_node, global_vocab)
            
            dec_ctx.layers, layer_counter = self._build_section_topology(
                section=dec_ctx,
                config_node=dec_node,
                prefix="decoder",
                mask_type="causal",
                has_cross_attn=True, 
                start_layer_id=layer_counter,
                include_embedding=True,
                include_lm_head=True
            )
            context.sections["decoder"] = dec_ctx

        return context

    def _build_section_topology(self, section: SectionContext, config_node: Dict, prefix: str, mask_type: str, 
                                has_cross_attn: bool, start_layer_id: int, 
                                include_embedding: bool, include_lm_head: bool):
        """Builds the sequential LayerIR list for a specific section."""
        layers_ir = []
        current_id = start_layer_id

        # 01. Embedding Layer
        if include_embedding:
            layers_ir.append(LayerIR(
                layer_id=current_id,
                name=f"{prefix}_embedding",
                type="Embedding",
                params=section.vocab_size * section.hidden_size,
                weight_shape=[section.vocab_size, section.hidden_size],
                parallel_capability={"tp_strategy": "vocab_parallel", "pp_splittable": False}
            ))
            current_id += 1

        # 02. Transformer Blocks
        for i in range(1, section.num_layers + 1):
            block_params = section.estimate_block_params(has_cross_attn=has_cross_attn)
            block = LayerIR(
                layer_id=current_id,
                name=f"{prefix}_layer_{i}",
                type="TransformerBlock",
                params=block_params,
                weight_shape=[section.hidden_size, section.hidden_size],
                sub_components=self._generate_transformer_components(section, mask_type, has_cross_attn)
            )
            layers_ir.append(block)
            current_id += 1

        # 03. LM Head (Output Layer)
        if include_lm_head:
            untie = self._get_arch_val("untie_embeddings_and_output_weights", False, config_node=config_node)
            is_tied = not untie
            
            layers_ir.append(LayerIR(
                layer_id=current_id,
                name=f"{prefix}_lm_head",
                type="Linear",
                params=section.hidden_size * section.vocab_size if is_tied is False else 0,
                weight_shape=[section.hidden_size, section.vocab_size],
                parallel_capability={
                    "tp_strategy": "vocab_parallel",
                    "pp_splittable": False,
                    "is_tied": is_tied,
                    "shared_with_id": start_layer_id if is_tied else None
                }
            ))
            current_id += 1

        return layers_ir, current_id

    def _generate_transformer_components(self, section: SectionContext, mask_type: str, has_cross_attn: bool) -> List[SubComponent]:
        """Generates dynamic sub-components utilizing the explicit dataclass fields."""
        components = []
        head_dim = section.hidden_size // section.num_attention_heads
        act_func = section.operator_features.get("activation_func", "gelu").lower()
        norm_type = section.operator_features.get("normalization", "layernorm").lower()

        # A. Input Norm
        components.append(SubComponent(
            role="input_norm", op_type=norm_type, weight_shape=[section.hidden_size]
        ))

        # B. Self Attention
        is_gqa = section.num_query_groups < section.num_attention_heads
        qkv_out_dim = (section.num_attention_heads + 2 * section.num_query_groups) * head_dim
        
        components.append(SubComponent(
            role="self_attn_qkv", op_type="Linear", weight_shape=[section.hidden_size, qkv_out_dim],
            tp_split_dim=0,
            attention_mask_type=mask_type,
            attention_type="gqa" if is_gqa else "mha",
            num_groups=section.num_query_groups
        ))
        
        components.append(SubComponent(
            role="self_attn_out", op_type="Linear", weight_shape=[section.hidden_size, section.hidden_size],
            tp_split_dim=1
        ))

        # B2. Cross Attention (for Decoders in T5-style architectures)
        if has_cross_attn:
            components.append(SubComponent(
                role="cross_attn_norm", op_type=norm_type, weight_shape=[section.hidden_size]
            ))
            
            components.append(SubComponent(
                role="cross_attn_qkv", op_type="Linear", weight_shape=[section.hidden_size, qkv_out_dim],
                tp_split_dim=0,
                attention_mask_type="cross",
                attention_type="mha"
            ))
            
            components.append(SubComponent(
                role="cross_attn_out", op_type="Linear", weight_shape=[section.hidden_size, section.hidden_size],
                tp_split_dim=1
            ))

        # C. Post-Attention Norm
        components.append(SubComponent(
            role="post_attn_norm", op_type=norm_type, weight_shape=[section.hidden_size]
        ))

        # D. MLP
        mlp_multiplier = 2 if act_func == "swiglu" else 1
        
        components.append(SubComponent(
            role="mlp_h_to_4h", op_type="Linear", weight_shape=[section.hidden_size, section.ffn_hidden_size * mlp_multiplier],
            tp_split_dim=0,
            fused_activation=act_func
        ))
        
        components.append(SubComponent(
            role="mlp_4h_to_h", op_type="Linear", weight_shape=[section.ffn_hidden_size, section.hidden_size],
            tp_split_dim=1
        ))

        return components
        