from typing import Dict, Optional, Tuple

import torch
import torch.distributed
from torch import nn

from mindspeed_bridge.models.conversion.utils import remove_non_pickleables

from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    MegatronParamMapping,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.bridge.models.conversion.param_mapping import (
    merge_gdn_linear_weights,
    split_gdn_linear_weights,
)


class GDNLinearMappingSeparate(MegatronParamMapping[Dict[str, torch.Tensor]]):
    """GDN input projection mapping for models with separate QKV, Z, B, A HF weights.

    Unlike :class:`GDNLinearMapping` which expects two fused tensors (``in_proj_qkvz``
    and ``in_proj_ba`` in Qwen3-Next's head-grouped layout), this mapping handles
    models that store each projection component separately:

    * ``in_proj_qkv`` - fused Q, K, V projection  (flat ``[Q; K; V]``)
    * ``in_proj_z``   - Z (gate) projection
    * ``in_proj_b``   - B (beta) projection
    * ``in_proj_a``   - A (alpha) projection

    Used by **Qwen3.5** whose GDN layers expose four distinct weight matrices.

    The class converts between the 4-tensor HF layout and Megatron's single
    ``in_proj`` tensor by first assembling the head-grouped ``qkvz`` / ``ba``
    intermediates expected by the existing :func:`merge_gdn_linear_weights` and
    :func:`split_gdn_linear_weights` helpers, keeping the TP-sharding logic
    unchanged.
    """

    def __init__(self, megatron_param: str, qkv: str, z: str, b: str, a: str):
        """Initialise GDN separate-component mapping.

        Args:
            megatron_param: Megatron ``in_proj`` parameter name pattern.
            qkv: HF weight pattern for the fused Q/K/V projection.
            z:   HF weight pattern for the Z (gate) projection.
            b:   HF weight pattern for the B (beta) projection.
            a:   HF weight pattern for the A (alpha) projection.
        """
        super().__init__(megatron_param, {"qkv": qkv, "z": z, "b": b, "a": a})
        self._tp_mapping = AutoMapping(megatron_param, megatron_param)

    # --------------------------------------------------------------------- #
    # HF → Megatron
    # --------------------------------------------------------------------- #
    def hf_to_megatron(
        self,
        hf_weights: Dict[str, torch.Tensor],
        megatron_module: nn.Module,
    ) -> torch.Tensor:
        """Merge four separate HF tensors into Megatron's single ``in_proj``."""
        if self.tp_rank == 0:
            config = self._get_config(megatron_module)
            qkvz, ba = _fuse_gdn_separate_to_grouped(
                config,
                hf_weights["qkv"],
                hf_weights["z"],
                hf_weights["b"],
                hf_weights["a"],
            )
            merged = merge_gdn_linear_weights(config, qkvz, ba, tp_size=self.tp_size)
        else:
            merged = None

        return self._tp_mapping.hf_to_megatron(merged, megatron_module)

    # --------------------------------------------------------------------- #
    # Megatron → HF
    # --------------------------------------------------------------------- #
    def megatron_to_hf(
        self,
        megatron_weights: Optional[torch.Tensor],
        megatron_module: Optional[nn.Module],
    ) -> Dict[str, torch.Tensor]:
        """Gather shards and split into the four separate HF tensors."""
        if megatron_weights is not None:
            megatron_weights = self.maybe_dequantize(megatron_weights)

        # Broadcast config across PP ranks (mirrors GDNLinearMapping).
        if megatron_module is None:
            config = self.broadcast_obj_from_pp_rank(None)
        else:
            config = self._get_config(megatron_module)
            config = remove_non_pickleables(config, max_depth=3)
            config = self.broadcast_obj_from_pp_rank(config)

        packed_dict = self._tp_mapping.megatron_to_hf(megatron_weights, megatron_module)
        if not packed_dict:
            return {}

        packed_in_proj = next(iter(packed_dict.values()))
        qkvz, ba = split_gdn_linear_weights(config, packed_in_proj, tp_size=self.tp_size)
        qkv, z, b, a = _split_gdn_grouped_to_separate(config, qkvz, ba)

        return {
            self.hf_param["qkv"]: qkv,
            self.hf_param["z"]: z,
            self.hf_param["b"]: b,
            self.hf_param["a"]: a,
        }

    # --------------------------------------------------------------------- #
    # Pattern resolution
    # --------------------------------------------------------------------- #
    def resolve(self, captures: Tuple[str, ...]) -> "MegatronParamMapping":
        resolved_megatron_param, resolved_hf_param = self._resolve_names(captures)
        return type(self)(
            resolved_megatron_param,
            resolved_hf_param["qkv"],
            resolved_hf_param["z"],
            resolved_hf_param["b"],
            resolved_hf_param["a"],
        )


def _fuse_gdn_separate_to_grouped(
    config: TransformerConfig,
    qkv: torch.Tensor,
    z: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert four separate (flat) GDN projection tensors into the head-grouped
    ``qkvz`` and ``ba`` format expected by :func:`merge_gdn_linear_weights`.

    Args:
        config: Transformer configuration with GDN head dimensions.
        qkv: Flat ``[Q; K; V]`` tensor of shape ``(qk_dim*2 + v_dim, hidden)``.
        z:   Z projection of shape ``(v_dim, hidden)``.
        b:   B projection of shape ``(num_v_heads, hidden)``.
        a:   A projection of shape ``(num_v_heads, hidden)``.

    Returns:
        Tuple of (qkvz, ba) in head-grouped layout that
        :func:`merge_gdn_linear_weights` can consume directly.
    """
    hidden_size = config.hidden_size
    qk_head_dim = config.linear_key_head_dim
    v_head_dim = config.linear_value_head_dim
    num_qk_heads = config.linear_num_key_heads
    num_v_heads = config.linear_num_value_heads
    qk_dim = qk_head_dim * num_qk_heads
    v_dim = v_head_dim * num_v_heads
    v_per_group = num_v_heads // num_qk_heads

    expected_qkv = (qk_dim * 2 + v_dim, hidden_size)
    expected_z = (v_dim, hidden_size)
    expected_ba = (num_v_heads, hidden_size)
    if tuple(qkv.shape) != expected_qkv:
        raise ValueError(f"qkv shape mismatch: expected {expected_qkv}, got {tuple(qkv.shape)}")
    if tuple(z.shape) != expected_z:
        raise ValueError(f"z shape mismatch: expected {expected_z}, got {tuple(z.shape)}")
    if tuple(b.shape) != expected_ba:
        raise ValueError(f"b shape mismatch: expected {expected_ba}, got {tuple(b.shape)}")
    if tuple(a.shape) != expected_ba:
        raise ValueError(f"a shape mismatch: expected {expected_ba}, got {tuple(a.shape)}")

    # --- Split flat QKV into individual components ---
    q_flat, k_flat, v_flat = torch.split(qkv, [qk_dim, qk_dim, v_dim], dim=0)

    # --- Reshape every component to (num_qk_heads, per_group_dim, hidden) ---
    q_g = q_flat.reshape(num_qk_heads, qk_head_dim, hidden_size)
    k_g = k_flat.reshape(num_qk_heads, qk_head_dim, hidden_size)
    v_g = v_flat.reshape(num_qk_heads, v_per_group * v_head_dim, hidden_size)
    z_g = z.reshape(num_qk_heads, v_per_group * v_head_dim, hidden_size)
    b_g = b.reshape(num_qk_heads, v_per_group, hidden_size)
    a_g = a.reshape(num_qk_heads, v_per_group, hidden_size)

    # --- Assemble grouped qkvz and ba ---
    qkvz = torch.cat([q_g, k_g, v_g, z_g], dim=1).reshape(-1, hidden_size)
    ba = torch.cat([b_g, a_g], dim=1).reshape(-1, hidden_size)

    return qkvz, ba


def _split_gdn_grouped_to_separate(
    config: TransformerConfig,
    qkvz: torch.Tensor,
    ba: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert head-grouped ``qkvz`` and ``ba`` tensors (as produced by
    :func:`split_gdn_linear_weights`) back into four flat tensors.

    Returns:
        Tuple of (qkv, z, b, a) where each tensor has a flat per-component layout.
    """
    hidden_size = config.hidden_size
    qk_head_dim = config.linear_key_head_dim
    v_head_dim = config.linear_value_head_dim
    num_qk_heads = config.linear_num_key_heads
    num_v_heads = config.linear_num_value_heads
    v_per_group = num_v_heads // num_qk_heads

    expected_qkvz_dim0 = num_qk_heads * (qk_head_dim * 2 + v_per_group * v_head_dim * 2)
    expected_ba_dim0 = num_qk_heads * v_per_group * 2
    if qkvz.ndim != 2 or qkvz.shape[0] != expected_qkvz_dim0 or qkvz.shape[1] != hidden_size:
        raise ValueError(
            f"qkvz shape mismatch: expected ({expected_qkvz_dim0}, {hidden_size}), got {tuple(qkvz.shape)}"
        )
    if ba.ndim != 2 or ba.shape[0] != expected_ba_dim0 or ba.shape[1] != hidden_size:
        raise ValueError(f"ba shape mismatch: expected ({expected_ba_dim0}, {hidden_size}), got {tuple(ba.shape)}")

    # --- Split grouped QKVZ ---
    qkvz_g = qkvz.reshape(num_qk_heads, -1, hidden_size)
    q_g, k_g, v_g, z_g = torch.split(
        qkvz_g,
        [qk_head_dim, qk_head_dim, v_per_group * v_head_dim, v_per_group * v_head_dim],
        dim=1,
    )
    q_flat = q_g.reshape(-1, hidden_size)
    k_flat = k_g.reshape(-1, hidden_size)
    v_flat = v_g.reshape(-1, hidden_size)
    z_flat = z_g.reshape(-1, hidden_size)
    qkv = torch.cat([q_flat, k_flat, v_flat], dim=0)

    # --- Split grouped BA ---
    ba_g = ba.reshape(num_qk_heads, -1, hidden_size)
    b_g, a_g = torch.split(ba_g, [v_per_group, v_per_group], dim=1)
    b_flat = b_g.reshape(-1, hidden_size)
    a_flat = a_g.reshape(-1, hidden_size)

    return qkv, z_flat, b_flat, a_flat
