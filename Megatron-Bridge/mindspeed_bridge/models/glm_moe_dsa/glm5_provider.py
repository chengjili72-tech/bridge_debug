from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Union

from megatron.core.models.gpt import GPTModel as MCoreGPTModel

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.transformer_config import MLATransformerConfig

try:
    import transformer_engine  # noqa: F401

    HAVE_TE = True
except (ImportError, ModuleNotFoundError):
    HAVE_TE = False

if TYPE_CHECKING:
    from megatron.core.transformer import ModuleSpec

if HAVE_TE:
    pass

from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
)


@dataclass
class GLM5ModelProvider(GPTModelProvider, MLATransformerConfig):
    """
    Model provider for GLM-5 (MoE + MLA + DSA).

    GLM-5 combines Multi-Latent Attention (MLA) with Dynamic Sparse Attention (DSA)
    and Mixture-of-Experts (MoE) to achieve efficient long-context modeling and
    high throughput inference.

    Key Architecture Details:
    - Multi-Latent Attention (MLA): Compresses KV cache using low-rank projections
    - Dynamic Sparse Attention (DSA): Selects top-k relevant tokens for long context
    - Mixture-of-Experts: Routed experts with shared expert overlap
    - Gated Linear Units: SiLU activation with gate projection

    MLA Parameters:
    - q_lora_rank: Rank for query low-rank projection
    - kv_lora_rank: Rank for KV low-rank projection
    - qk_head_dim: Dimension for QK attention heads (non-rotary part)
    - qk_pos_emb_head_dim: Dimension for QK rotary position embeddings
    - v_head_dim: Dimension for value heads

    DSA Parameters:
    - dsa_indexer_head_dim: Dimension for DSA indexer
    - dsa_indexer_n_heads: Number of DSA indexer heads
    - dsa_indexer_topk: Top-k selection for sparse attention
    - dsa_indexer_loss_coeff: Coefficient for DSA sparse loss

    MoE Parameters:
    - num_moe_experts: Total number of experts
    - moe_router_topk: Number of routed experts per token
    - moe_shared_expert_overlap: Whether shared expert overlaps with routed experts
    - moe_layer_freq: Frequency of MoE layers (vs dense layers)
    """

    # =========================================================================
    # Layer Specification - Use DSA MLA spec
    # =========================================================================
    transformer_layer_spec: Union["ModuleSpec", Callable[["GPTModelProvider"], "ModuleSpec"]] = (
        get_transformer_block_with_experimental_attention_variant_spec
    )

    # =========================================================================
    # Multi-Latent Attention (MLA) Configuration
    # =========================================================================
    multi_latent_attention: bool = True

    # MLA low-rank projection ranks
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512

    # MLA head dimensions
    qk_head_dim: int = 128
    qk_pos_emb_head_dim: int = 64
    v_head_dim: int = 128

    # =========================================================================
    # Dynamic Sparse Attention (DSA) Configuration
    # =========================================================================
    experimental_attention_variant: str = "dsa"

    # DSA indexer parameters
    dsa_indexer_head_dim: int = 128
    dsa_indexer_n_heads: int = 8
    dsa_indexer_topk: int = 8
    dsa_indexer_loss_coeff: float = 0.001
    dsa_indexer_use_sparse_loss: bool = True

    # =========================================================================
    # Mixture-of-Experts (MoE) Configuration
    # =========================================================================
    num_moe_experts: int = 256
    moe_router_topk: int = 6
    moe_router_pre_softmax: bool = True
    moe_router_score_function: str = "sigmoid"
    moe_router_enable_expert_bias: bool = True
    moe_router_dtype: str = "fp32"
    moe_router_load_balancing_type: str = "seq_aux_loss"
    moe_router_num_groups: int = 1
    moe_router_group_topk: int = 1
    moe_router_topk_scaling_factor: float = 1.0
    moe_grouped_gemm: bool = True
    moe_token_dispatcher_type: str = "alltoall"
    moe_permute_fusion: bool = True
    moe_shared_expert_overlap: bool = True
    moe_aux_loss_coeff: float = 0.0001

    # MoE layer frequency (mix of dense and MoE layers)
    moe_layer_freq: list[int] | None = None

    # =========================================================================
    # Common LLM Parameters
    # =========================================================================
    normalization: str = "RMSNorm"
    layernorm_epsilon: float = 1e-6
    gated_linear_unit: bool = True
    add_bias_linear: bool = False
    add_qkv_bias: bool = False
    qk_layernorm: bool = True  # GLM5 has q_a_layernorm and kv_a_layernorm
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    attention_softmax_in_fp32: bool = False

    # Position embedding (RoPE)
    position_embedding_type: str = "rope"
    rope_type: str = "rope"
    rotary_percent: float = 1.0
    rotary_scaling_factor: float = 1.0
    mscale: float = 1.0
    mscale_all_dim: float = 1.0
    apply_rope_fusion: bool = False

    max_position_embeddings: int = 202752
    yarn_original_max_position_embeddings: int = 202752
    yarn_rotary_scaling_factor: float = 40.0
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0
    yarn_mscale: float = 1.0
    yarn_mscale_all_dim: float = 1.0
    yarn_correction_range_round_to_int: bool = True

    # DSA & Fused Ops Flags
    use_dsa_absorb: bool = True
    use_fused_lightning_indexer: bool = True
    use_fused_sparse_flash_attention: bool = True
    use_fused_lightning_indexer_kl_loss: bool = True

    # MTP
    mtp_num_layers: int = 1

    # =========================================================================
    # Other Parameters
    # =========================================================================
    share_embeddings_and_output_weights: bool = False
    make_vocab_size_divisible_by: int = 1

    # =========================================================================
    # NPU-specific Parameters
    # =========================================================================
    # Disable gradient accumulation fusion on NPU (requires APEX with CUDA extensions)
    gradient_accumulation_fusion: bool = False

    def __post_init__(self):
        """Initialize the provider with proper validation.

        This method:
        1. Validates MLA parameters are consistent
        2. Sets default MoE layer frequency if not specified
        3. Calls parent __post_init__ to ensure proper initialization
        """
        # Validate MLA parameters
        if self.multi_latent_attention:
            if self.q_lora_rank <= 0:
                raise ValueError(f"q_lora_rank must be positive, got {self.q_lora_rank}")
            if self.kv_lora_rank <= 0:
                raise ValueError(f"kv_lora_rank must be positive, got {self.kv_lora_rank}")
            if self.qk_head_dim <= 0:
                raise ValueError(f"qk_head_dim must be positive, got {self.qk_head_dim}")
            if self.v_head_dim <= 0:
                raise ValueError(f"v_head_dim must be positive, got {self.v_head_dim}")

        # Set default MoE layer frequency (all layers use MoE)
        if self.moe_layer_freq is None and self.num_layers is not None:
            self.moe_layer_freq = [1] * self.num_layers

        # Call parent __post_init__ to ensure proper initialization
        super().__post_init__()

    def finalize(self) -> None:
        """Finalize the provider configuration.

        This method:
        1. Validates parallelism settings
        2. Calls parent finalize() to compute derived fields
        """
        self.validate_parallelism()
        super().finalize()

    def validate_parallelism(self):
        """Validate that parallelism settings are compatible with this model's architecture.

        Call this after mutating parallelism attributes (e.g. tensor_model_parallel_size)
        on an already-constructed provider, since finalize() only runs once before provide().
        """
        # MLA-specific validation
        if self.multi_latent_attention:
            # For MLA, TP size should be compatible with head dimensions
            if self.qk_head_dim < self.tensor_model_parallel_size:
                raise ValueError(
                    f"TP size {self.tensor_model_parallel_size} should be less than or equal to "
                    f"qk_head_dim {self.qk_head_dim}. Please use a smaller TP size."
                )
            if self.v_head_dim < self.tensor_model_parallel_size:
                raise ValueError(
                    f"TP size {self.tensor_model_parallel_size} should be less than or equal to "
                    f"v_head_dim {self.v_head_dim}. Please use a smaller TP size."
                )

        # MoE-specific validation
        if self.num_moe_experts is not None and self.num_moe_experts > 0:
            if self.moe_router_topk > self.num_moe_experts:
                raise ValueError(
                    f"moe_router_topk {self.moe_router_topk} should be less than or equal to "
                    f"num_moe_experts {self.num_moe_experts}."
                )

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> MCoreGPTModel:
        """Provide a GLM-5 model instance.

        This method creates a GPTModel with GLM-5 specific configurations
        including MLA, DSA, and MoE.

        Args:
            pre_process: Whether this rank should process the first pipeline stage
            post_process: Whether this rank should process the last pipeline stage
            vp_stage: Virtual pipeline stage index

        Returns:
            MCoreGPTModel instance configured for GLM-5
        """
        # Delegate to parent GPTModelProvider.provide()
        # The transformer_layer_spec will handle DSA MLA variant
        return GPTModelProvider.provide(self, pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)

    def provide_language_model(self, pre_process=None, post_process=None, vp_stage=None) -> MCoreGPTModel:
        """Provide just the language model component.

        This is an alias for provide() for consistency with other providers.
        """
        return self.provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)


__all__ = ["GLM5ModelProvider"]
