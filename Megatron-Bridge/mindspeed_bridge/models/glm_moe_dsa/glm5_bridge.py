import logging

import torch
from megatron.core import parallel_state

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
)
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.utils.common_utils import extract_expert_number_from_param
from megatron.core.models.gpt.gpt_model import GPTModel

from mindspeed_bridge.models.glm_moe_dsa.glm_moe_mappings import (
    GLMExpertDownProjMapping,
    GLMExpertGateUpProjMapping,
    GLMKVUpProjMapping,
)
from mindspeed_bridge.models.glm_moe_dsa.glm5_provider import GLM5ModelProvider

logger = logging.getLogger(__name__)

_GLM5_HF_CLASS_NAME = "GlmMoeDsaForCausalLM"


@MegatronModelBridge.register_bridge(
    source=_GLM5_HF_CLASS_NAME,
    target=GPTModel,
    provider=GLM5ModelProvider,
    model_type="glm5",
)
class GLM5Bridge(MegatronModelBridge):
    """
    Megatron Bridge for GLM-5 (MoE + MLA + DSA).

    This bridge handles the conversion between HuggingFace GlmMoeDsaForCausalLM
    and Megatron-Core GPTModel formats.

    GLM-5 uses:
    - Multi-Latent Attention (MLA): Compresses KV cache using low-rank projections
    - Dynamic Sparse Attention (DSA): Selects top-k relevant tokens for long context
    - Mixture-of-Experts (MoE): Routed experts with shared expert overlap
    - Gated Linear Units (GLU): SiLU activation with gate projection

    The weight mappings handle:
    - Language model embeddings and output layer
    - MLA weights (q_down_proj, q_up_proj, kv_down_proj, k_up_proj, v_up_proj)
    - DSA indexer weights
    - MoE router and expert MLPs (routed and shared experts)
    - Layer norms (input, post-attention, q_a_layernorm, kv_a_layernorm)

    Example:
         >>> from megatron.bridge import AutoBridge
         >>> bridge = AutoBridge.from_hf_pretrained("zai-org/GLM-5")
         >>> provider = bridge.to_megatron_provider()
    """

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> GLM5ModelProvider:
        """
        Create a GLM5ModelProvider from a HuggingFace pretrained model.

        Extracts configuration from the HuggingFace config and maps it to
        Megatron provider parameters.

        Args:
            hf_pretrained: HuggingFace pretrained model

        Returns:
            GLM5ModelProvider configured with the HF model's parameters
        """
        hf_config = hf_pretrained.config

        # Use base class utility to extract common config fields
        provider_kwargs = self.hf_config_to_provider_kwargs(hf_config)

        # Create provider with common parameters
        provider = GLM5ModelProvider(**provider_kwargs)

        # --- Common GLM5 LLM settings ---
        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.add_bias_linear = False
        provider.add_qkv_bias = False
        provider.qk_layernorm = True  # GLM5 MLA has q_a_layernorm and kv_a_layernorm
        provider.hidden_dropout = 0.0
        provider.attention_dropout = 0.0
        provider.attention_softmax_in_fp32 = False

        # --- Position embedding settings ---
        provider.rope_type = "yarn"
        provider.rotary_base = getattr(hf_config, "rope_theta", 1000000)
        provider.yarn_original_max_position_embeddings = 202752
        provider.yarn_rotary_scaling_factor = 40
        provider.yarn_beta_fast = 32
        provider.yarn_beta_slow = 1
        provider.yarn_mscale = 1.0
        provider.yarn_mscale_all_dim = 1.0
        provider.yarn_correction_range_round_to_int = True

        # --- Embedding settings ---
        provider.share_embeddings_and_output_weights = False
        provider.make_vocab_size_divisible_by = 1

        # --- MLA (Multi-Latent Attention) parameters ---
        provider.multi_latent_attention = True
        provider.q_lora_rank = getattr(hf_config, "q_lora_rank", 1536)
        provider.kv_lora_rank = getattr(hf_config, "kv_lora_rank", 512)
        provider.layernorm_epsilon = getattr(hf_config, "layernorm_epsilon", 1e-5)
        provider.qk_head_dim = getattr(hf_config, "qk_nope_head_dim", 128)
        provider.qk_pos_emb_head_dim = getattr(hf_config, "qk_rope_head_dim", 64)
        provider.v_head_dim = getattr(hf_config, "v_head_dim", 128)
        provider.mtp_num_layers = 0

        # --- DSA (Dynamic Sparse Attention) parameters ---
        provider.experimental_attention_variant = "dsa"
        provider.dsa_indexer_head_dim = getattr(hf_config, "index_head_dim", 128)
        provider.dsa_indexer_n_heads = getattr(hf_config, "index_n_heads", 8)
        provider.dsa_indexer_topk = getattr(hf_config, "index_topk", 8)
        provider.dsa_indexer_loss_coeff = 0.001
        provider.dsa_indexer_use_sparse_loss = True

        # --- MoE parameters ---
        provider.num_moe_experts = getattr(hf_config, "n_routed_experts", 256)
        provider.moe_router_topk = getattr(hf_config, "num_experts_per_tok", 6)
        provider.moe_shared_expert_intermediate_size = getattr(hf_config, "moe_intermediate_size", 2048)
        provider.moe_router_pre_softmax = True
        provider.moe_router_score_function = "sigmoid"
        provider.moe_router_enable_expert_bias = True
        provider.moe_router_dtype = "fp32"
        provider.moe_router_load_balancing_type = "seq_aux_loss"
        provider.moe_router_num_groups = getattr(hf_config, "n_group", 1)
        provider.moe_router_group_topk = getattr(hf_config, "topk_group", 1)
        provider.moe_router_topk_scaling_factor = getattr(hf_config, "routed_scaling_factor", 2.5)
        provider.moe_grouped_gemm = True
        provider.moe_token_dispatcher_type = "alltoall"  # nosec B105
        provider.moe_permute_fusion = True
        provider.moe_shared_expert_overlap = True

        provider.max_position_embeddings = provider.yarn_original_max_position_embeddings
        provider.use_dsa_absorb = True
        provider.use_fused_lightning_indexer = True
        provider.use_fused_sparse_flash_attention = True
        provider.use_fused_lightning_indexer_kl_loss = True
        provider.multi_latent_attention = True
        if hasattr(hf_config, "aux_loss_alpha"):
            provider.moe_aux_loss_coeff = hf_config.aux_loss_alpha

        # MoE layer frequency (mix of dense and MoE layers)
        first_k_dense_replace = getattr(hf_config, "first_k_dense_replace", 0)
        if provider.num_layers is not None:
            provider.moe_layer_freq = [0] * first_k_dense_replace + [1] * (provider.num_layers - first_k_dense_replace)

        return provider

    def build_conversion_tasks(self, hf_pretrained, megatron_model):
        """Override to store config before mapping_registry is called."""
        # pylint: disable=attribute-defined-outside-init
        self._hf_config = hf_pretrained.config
        self._hf_state_source = hf_pretrained.state.source
        self._hf_keys = list(self._hf_state_source.get_all_keys())

        from megatron.bridge.models.conversion.param_mapping import MegatronParamMapping

        if not hasattr(MegatronParamMapping, '_orig_broadcast_from_pp_rank'):
            MegatronParamMapping._orig_broadcast_from_pp_rank = MegatronParamMapping.broadcast_from_pp_rank

        def _safe_broadcast(mapping_self, tensor, cache_key=None):
            if mapping_self.pp_size > 1:
                presence = [None] * mapping_self.pp_size
                torch.distributed.all_gather_object(presence, tensor is not None, group=mapping_self.pp_group)
                if sum(presence) > 1 and mapping_self.pp_rank != presence.index(True):
                    tensor = None
            return mapping_self._orig_broadcast_from_pp_rank(tensor, cache_key)

        MegatronParamMapping.broadcast_from_pp_rank = _safe_broadcast

        return super().build_conversion_tasks(hf_pretrained, megatron_model)

    def mapping_registry(self) -> MegatronMappingRegistry:
        """
        Return MegatronMappingRegistry containing parameter mappings for GLM-5.

        The weight mappings handle:
        1. Language model embeddings and output layer
        2. MLA weights (low-rank projections for Q and KV)
        3. DSA indexer weights
        4. MoE router and expert MLPs (both routed and shared experts)
        5. Layer norms (input, post-attention, q_a_layernorm, kv_a_layernorm)

        Naming Convention:
        - Megatron params are prefixed with "decoder." (no "language_model." prefix for pure LLM)
        - HF params are prefixed with "model."
        - Layer-specific params use ".*" wildcard

        Returns:
            MegatronMappingRegistry with all parameter mappings
        """

        # =====================================================================
        # Simple 1:1 parameter mappings
        # =====================================================================
        param_mappings = {
            # =================================================================
            # Embeddings and output
            # =================================================================
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "output_layer.weight": "lm_head.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            # =================================================================
            # Layer norms
            # =================================================================
            # Input layernorm (before attention)
            "decoder.layers.*.input_layernorm.weight": "model.layers.*.input_layernorm.weight",
            # Post-attention layernorm (before MLP)
            "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
            # =================================================================
            # Attention output
            # =================================================================
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            # =================================================================
            # MLA (Multi-Latent Attention) weights
            # =================================================================
            # Query low-rank projections
            "decoder.layers.*.self_attention.linear_q_down_proj.weight": "model.layers.*.self_attn.q_a_proj.weight",
            "decoder.layers.*.self_attention.linear_q_up_proj.weight": "model.layers.*.self_attn.q_b_proj.weight",
            # Query layernorm (after q_up_proj)
            "decoder.layers.*.self_attention.q_layernorm.weight": "model.layers.*.self_attn.q_a_layernorm.weight",
            # KV low-rank projections
            "decoder.layers.*.self_attention.linear_kv_down_proj.weight": "model.layers.*.self_attn.kv_a_proj_with_mqa.weight",
            # KV up projections (DSA splits K and V for AbsorbedDSA)
            "decoder.layers.*.self_attention.linear_k_up_proj.weight": "model.layers.*.self_attn.kv_b_proj.weight",
            "decoder.layers.*.self_attention.linear_v_up_proj.weight": "model.layers.*.self_attn.kv_b_proj.weight",
            "decoder.layers.*.self_attention.kv_layernorm.weight": "model.layers.*.self_attn.kv_a_layernorm.weight",
            # =================================================================
            # DSA (Dynamic Sparse Attention) indexer weights
            # =================================================================
            "decoder.layers.*.self_attention.core_attention.indexer.linear_wq_b.weight": "model.layers.*.self_attn.indexer.wq_b.weight",
            "decoder.layers.*.self_attention.core_attention.indexer.linear_wk.weight": "model.layers.*.self_attn.indexer.wk.weight",
            "decoder.layers.*.self_attention.core_attention.indexer.k_norm.weight": "model.layers.*.self_attn.indexer.k_norm.weight",
            "decoder.layers.*.self_attention.core_attention.indexer.k_norm.bias": "model.layers.*.self_attn.indexer.k_norm.bias",
            "decoder.layers.*.self_attention.core_attention.indexer.linear_weights_proj.weight": "model.layers.*.self_attn.indexer.weights_proj.weight",
            # =================================================================
            # Dense MLP (down projection)
            # =================================================================
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
            # =================================================================
            # MoE router
            # =================================================================
            "decoder.layers.*.mlp.router.weight": "model.layers.*.mlp.gate.weight",
            "decoder.layers.*.mlp.router.expert_bias": "model.layers.*.mlp.gate.e_score_correction_bias",
            # =================================================================
            # MoE shared experts (down projection)
            # =================================================================
            "decoder.layers.*.mlp.shared_experts.linear_fc2.weight": "model.layers.*.mlp.shared_experts.down_proj.weight",
        }

        mapping_list = []

        for megatron_param, hf_param in param_mappings.items():
            if "linear_k_up_proj" in megatron_param or "linear_v_up_proj" in megatron_param:
                mapping_list.append(GLMKVUpProjMapping(megatron_param=megatron_param, hf_param=hf_param))
            else:
                mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        AutoMapping.register_module_type("SharedExpertMLP", "column")
        AutoMapping.register_module_type("TELinear", "replicated")
        AutoMapping.register_module_type("ColumnParallelLinear", "column")
        AutoMapping.register_module_type("RowParallelLinear", "row")
        AutoMapping.register_module_type("TEColumnParallelGroupedLinear", "column")
        AutoMapping.register_module_type("TERowParallelGroupedLinear", "row")

        use_fused_experts = self._uses_fused_experts()
        gate_up_suffix = self._hf_expert_suffix("mlp.experts.gate_up_proj")
        down_suffix = self._hf_expert_suffix("mlp.experts.down_proj")

        mapping_list.extend(
            [
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                    gate="model.layers.*.mlp.gate_proj.weight",
                    up="model.layers.*.mlp.up_proj.weight",
                ),
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
                    gate="model.layers.*.mlp.shared_experts.gate_proj.weight",
                    up="model.layers.*.mlp.shared_experts.up_proj.weight",
                ),
            ]
        )

        if use_fused_experts:
            mapping_list.extend(
                [
                    GLMExpertGateUpProjMapping(
                        megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
                        hf_param=f"model.layers.*.mlp.experts.gate_up_proj{gate_up_suffix}",
                    ),
                    GLMExpertDownProjMapping(
                        megatron_param="decoder.layers.*.mlp.experts.linear_fc2.weight*",
                        hf_param=f"model.layers.*.mlp.experts.down_proj{down_suffix}",
                    ),
                ]
            )
        else:
            mapping_list.extend(
                [
                    GatedMLPMapping(
                        megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
                        gate="model.layers.*.mlp.experts.*.gate_proj.weight",
                        up="model.layers.*.mlp.experts.*.up_proj.weight",
                    ),
                    AutoMapping(
                        megatron_param="decoder.layers.*.mlp.experts.linear_fc2.weight*",
                        hf_param="model.layers.*.mlp.experts.*.down_proj.weight",
                    ),
                ]
            )

        # =====================================================================
        # MTP (Multi-Token Prediction) layer mappings
        # =====================================================================
        if not hasattr(self, "_hf_config"):
            logger.warning("No HF config found, skipping MTP mappings.")
            return MegatronMappingRegistry(*mapping_list)

        hf_config = self._hf_config
        num_mtp_layers = getattr(hf_config, "num_nextn_predict_layers", 0)
        num_transformer_layers = hf_config.num_hidden_layers

        for mtp_layer in range(num_mtp_layers):
            hf_layer_idx = mtp_layer + num_transformer_layers

            mtp_layer_mappings = {
                f"mtp.layers.{mtp_layer}.transformer_layer.input_layernorm.weight": f"model.layers.{hf_layer_idx}.input_layernorm.weight",
                f"mtp.layers.{mtp_layer}.transformer_layer.pre_mlp_layernorm.weight": f"model.layers.{hf_layer_idx}.post_attention_layernorm.weight",
                f"mtp.layers.{mtp_layer}.transformer_layer.mlp.linear_fc1.layer_norm_weight": f"model.layers.{hf_layer_idx}.post_attention_layernorm.weight",
                f"mtp.layers.{mtp_layer}.transformer_layer.self_attention.linear_proj.weight": f"model.layers.{hf_layer_idx}.self_attn.o_proj.weight",
                f"mtp.layers.{mtp_layer}.transformer_layer.self_attention.linear_q_down_proj.weight": f"model.layers.{hf_layer_idx}.self_attn.q_a_proj.weight",
                f"mtp.layers.{mtp_layer}.transformer_layer.self_attention.linear_q_up_proj.weight": f"model.layers.{hf_layer_idx}.self_attn.q_b_proj.weight",
                f"mtp.layers.{mtp_layer}.transformer_layer.self_attention.q_layernorm.weight": f"model.layers.{hf_layer_idx}.self_attn.q_a_layernorm.weight",
                f"mtp.layers.{mtp_layer}.transformer_layer.self_attention.linear_kv_down_proj.weight": f"model.layers.{hf_layer_idx}.self_attn.kv_a_proj_with_mqa.weight",
                f"mtp.layers.{mtp_layer}.transformer_layer.self_attention.linear_k_up_proj.weight": f"model.layers.{hf_layer_idx}.self_attn.kv_b_proj.weight",
                f"mtp.layers.{mtp_layer}.transformer_layer.self_attention.linear_v_up_proj.weight": f"model.layers.{hf_layer_idx}.self_attn.kv_b_proj.weight",
                f"mtp.layers.{mtp_layer}.transformer_layer.self_attention.kv_layernorm.weight": f"model.layers.{hf_layer_idx}.self_attn.kv_a_layernorm.weight",
                f"mtp.layers.{mtp_layer}.transformer_layer.self_attention.core_attention.indexer.linear_wq_b.weight": f"model.layers.{hf_layer_idx}.self_attn.indexer.wq_b.weight",
                f"mtp.layers.{mtp_layer}.transformer_layer.self_attention.core_attention.indexer.linear_wk.weight": f"model.layers.{hf_layer_idx}.self_attn.indexer.wk.weight",
                f"mtp.layers.{mtp_layer}.transformer_layer.self_attention.core_attention.indexer.k_norm.weight": f"model.layers.{hf_layer_idx}.self_attn.indexer.k_norm.weight",
                f"mtp.layers.{mtp_layer}.transformer_layer.self_attention.core_attention.indexer.k_norm.bias": f"model.layers.{hf_layer_idx}.self_attn.indexer.k_norm.bias",
                f"mtp.layers.{mtp_layer}.transformer_layer.self_attention.core_attention.indexer.linear_weights_proj.weight": f"model.layers.{hf_layer_idx}.self_attn.indexer.weights_proj.weight",
                f"mtp.layers.{mtp_layer}.transformer_layer.mlp.router.weight": f"model.layers.{hf_layer_idx}.mlp.gate.weight",
                f"mtp.layers.{mtp_layer}.transformer_layer.mlp.router.expert_bias": f"model.layers.{hf_layer_idx}.mlp.gate.e_score_correction_bias",
                f"mtp.layers.{mtp_layer}.transformer_layer.mlp.shared_experts.linear_fc2.weight": f"model.layers.{hf_layer_idx}.mlp.shared_experts.down_proj.weight",
            }

            for megatron_param, hf_param in mtp_layer_mappings.items():
                if "linear_k_up_proj" in megatron_param or "linear_v_up_proj" in megatron_param:
                    mapping_list.append(GLMKVUpProjMapping(megatron_param=megatron_param, hf_param=hf_param))
                else:
                    mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

            mapping_list.extend(
                [
                    AutoMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.enorm.weight",
                        hf_param=f"model.layers.{hf_layer_idx}.enorm.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.hnorm.weight",
                        hf_param=f"model.layers.{hf_layer_idx}.hnorm.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.eh_proj.weight",
                        hf_param=f"model.layers.{hf_layer_idx}.eh_proj.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.final_layernorm.weight",
                        hf_param=f"model.layers.{hf_layer_idx}.shared_head.norm.weight",
                    ),
                    GatedMLPMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.transformer_layer.mlp.shared_experts.linear_fc1.weight",
                        gate=f"model.layers.{hf_layer_idx}.mlp.shared_experts.gate_proj.weight",
                        up=f"model.layers.{hf_layer_idx}.mlp.shared_experts.up_proj.weight",
                    ),
                ]
            )

            if use_fused_experts:
                mapping_list.extend(
                    [
                        GLMExpertGateUpProjMapping(
                            megatron_param=f"mtp.layers.{mtp_layer}.transformer_layer.mlp.experts.linear_fc1.weight*",
                            hf_param=f"model.layers.{hf_layer_idx}.mlp.experts.gate_up_proj{gate_up_suffix}",
                        ),
                        GLMExpertDownProjMapping(
                            megatron_param=f"mtp.layers.{mtp_layer}.transformer_layer.mlp.experts.linear_fc2.weight*",
                            hf_param=f"model.layers.{hf_layer_idx}.mlp.experts.down_proj{down_suffix}",
                        ),
                    ]
                )
            else:
                mapping_list.extend(
                    [
                        GatedMLPMapping(
                            megatron_param=f"mtp.layers.{mtp_layer}.transformer_layer.mlp.experts.linear_fc1.weight*",
                            gate=f"model.layers.{hf_layer_idx}.mlp.experts.*.gate_proj.weight",
                            up=f"model.layers.{hf_layer_idx}.mlp.experts.*.up_proj.weight",
                        ),
                        AutoMapping(
                            megatron_param=f"mtp.layers.{mtp_layer}.transformer_layer.mlp.experts.linear_fc2.weight*",
                            hf_param=f"model.layers.{hf_layer_idx}.mlp.experts.*.down_proj.weight",
                        ),
                    ]
                )

        return MegatronMappingRegistry(*mapping_list)

    def _uses_fused_experts(self) -> bool:
        """Check if HF model uses fused expert weights (gate_up_proj, down_proj)."""
        hf_keys = getattr(self, "_hf_keys", None)
        if hf_keys:
            if any("mlp.experts.gate_up_proj" in key for key in hf_keys) or any(
                "mlp.experts.down_proj" in key for key in hf_keys
            ):
                return True

        hf_source = getattr(self, "_hf_state_source", None)
        if hf_source is not None:
            return hf_source.has_glob("*mlp.experts.gate_up_proj*") or hf_source.has_glob("*mlp.experts.down_proj*")

        return False

    def _hf_expert_suffix(self, base_name: str) -> str:
        """Determine the suffix for HF expert weight names (.weight or empty)."""
        hf_keys = getattr(self, "_hf_keys", None) or []
        if any(f"{base_name}.weight" in key for key in hf_keys):
            return ".weight"

        hf_source = getattr(self, "_hf_state_source", None)
        if hf_source is not None and hf_source.has_glob(f"*{base_name}.weight"):
            return ".weight"

        return ""

    def stream_weights_megatron_to_hf(
        self,
        megatron_model,
        hf_pretrained,
        cpu: bool = True,
        show_progress: bool = True,
        conversion_tasks=None,
        merge_adapter_weights: bool = True,
    ):
        """Override to reset GLMKVUpProjMapping cache before conversion.

        This ensures that the class-level cache used for coordinating k_up and v_up
        parts is cleared at the beginning of each conversion session.
        """
        # Reset the cache for GLMKVUpProjMapping before starting conversion
        GLMKVUpProjMapping.reset_cache()

        # Call parent implementation
        return super().stream_weights_megatron_to_hf(
            megatron_model,
            hf_pretrained,
            cpu=cpu,
            show_progress=show_progress,
            conversion_tasks=conversion_tasks,
            merge_adapter_weights=merge_adapter_weights,
        )

    def maybe_modify_converted_hf_weight(
        self,
        task,
        converted_weights_dict: dict[str, torch.Tensor],
        hf_state_dict,
    ) -> dict[str, torch.Tensor]:
        """Handle fused expert weight conversion for Expert Parallelism.

        When using Expert Parallelism (EP), expert weights are sharded across ranks.
        This method collects and merges the sharded weights from all ranks before
        conversion to ensure correct weight loading.

        Args:
            task: The conversion task being processed
            converted_weights_dict: Dictionary of converted weights
            hf_state_dict: The HuggingFace state dict

        Returns:
            Merged weight dictionary, or empty dict if still collecting
        """
        if not isinstance(task.mapping, (GLMExpertGateUpProjMapping, GLMExpertDownProjMapping)):
            return converted_weights_dict

        if not converted_weights_dict:
            return {}

        num_experts = self._hf_config.n_routed_experts
        ep_size = parallel_state.get_expert_model_parallel_world_size()
        experts_per_rank = num_experts // ep_size

        try:
            local_expert_number = extract_expert_number_from_param(task.param_name) % experts_per_rank
        except ValueError:
            return converted_weights_dict

        if not hasattr(self, "hf_weights_cache"):
            # pylint: disable=attribute-defined-outside-init
            self.hf_weights_cache = {}

        for key, value in converted_weights_dict.items():
            if key not in self.hf_weights_cache:
                self.hf_weights_cache[key] = {}

            if ep_size == 1:
                self.hf_weights_cache[key][local_expert_number] = value
            else:
                if value.shape[0] != ep_size:
                    raise ValueError(f"Expected EP dim {ep_size} for {key}, got {value.shape}.")
                for i, exp_val in enumerate(value):
                    global_expert_number = local_expert_number + (i * experts_per_rank)
                    self.hf_weights_cache[key][global_expert_number] = exp_val

            if len(self.hf_weights_cache[key]) == num_experts:
                merged = torch.stack([self.hf_weights_cache[key][i] for i in range(num_experts)], dim=0)
                if key in hf_state_dict:
                    expected = hf_state_dict[key].shape
                    if merged.shape != expected and merged.transpose(-1, -2).shape == expected:
                        merged = merged.transpose(-1, -2).contiguous()
                del self.hf_weights_cache[key]
                return {key: merged}

            return {}

        return {}
