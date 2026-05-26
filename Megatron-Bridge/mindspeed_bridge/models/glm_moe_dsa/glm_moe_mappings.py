"""GLM MoE mapping helpers for fused expert weights in transformers 5.0+."""

from typing import Dict, Optional

import torch
from torch import nn

from megatron.bridge.models.conversion.param_mapping import AutoMapping, GatedMLPMapping, ColumnParallelMapping
from megatron.bridge.models.conversion.utils import get_module_and_param_from_name
from megatron.bridge.utils.common_utils import extract_expert_number_from_param


def _select_expert_weight(hf_weights: torch.Tensor, expert_idx: int) -> torch.Tensor:
    if hf_weights.ndim >= 3:
        return hf_weights[expert_idx]
    return hf_weights


def _align_weight_to_shape(weight: torch.Tensor, target_shape: torch.Size, name: str) -> torch.Tensor:
    if tuple(weight.shape) == tuple(target_shape):
        return weight
    if weight.ndim == 2 and tuple(weight.t().shape) == tuple(target_shape):
        return weight.t().contiguous()
    raise ValueError(f"Unexpected {name} shape {tuple(weight.shape)}; expected {tuple(target_shape)}.")


class _LooseGatedMLPMapping(GatedMLPMapping):
    def _validate_patterns(self, *args, **kwargs):
        # Allow mismatched wildcard counts for fused expert mappings.
        pass


class GLMExpertGateUpProjMapping(AutoMapping):
    """Mapping for fused expert gate+up projection weights."""

    def __init__(self, megatron_param: str, hf_param: str, permute_dims=None):
        super().__init__(megatron_param, hf_param, permute_dims)
        self._gated_mapping = _LooseGatedMLPMapping(
            megatron_param=self.megatron_param,
            gate=f"{self.hf_param}.gate",
            up=f"{self.hf_param}.up",
        )

    def hf_to_megatron(self, hf_weights: torch.Tensor, megatron_module: torch.nn.Module) -> torch.Tensor:
        global_expert_number = extract_expert_number_from_param(self.megatron_param)
        expert_weight = _select_expert_weight(hf_weights, global_expert_number)

        normalized_param = self._normalize_expert_param_name(self.megatron_param)
        _, target_param = get_module_and_param_from_name(megatron_module, normalized_param)
        target_shape = target_param.shape
        gate_target_shape = (target_shape[0] // 2, target_shape[1])

        if target_shape[0] % 2 != 0:
            raise ValueError(f"Expected even fused dim for {self.megatron_param}, got {target_shape}.")

        if expert_weight.ndim == 3 and expert_weight.shape[0] == 2:
            gate = _align_weight_to_shape(expert_weight[0], gate_target_shape, "gate")
            up = _align_weight_to_shape(expert_weight[1], gate_target_shape, "up")
        else:
            expert_weight = _align_weight_to_shape(expert_weight, target_shape, "gate_up")
            gate, up = torch.chunk(expert_weight, 2, dim=0)

        return self._gated_mapping.hf_to_megatron({"gate": gate, "up": up}, megatron_module)

    def megatron_to_hf(
        self, megatron_weights: torch.Tensor, megatron_module: torch.nn.Module
    ) -> Dict[str, torch.Tensor]:
        converted = self._gated_mapping.megatron_to_hf(megatron_weights, megatron_module)
        if not converted:
            return {}

        fused: Dict[str, torch.Tensor] = {}
        for name, tensor in converted.items():
            if not name.endswith(".gate"):
                continue
            base_name = name[: -len(".gate")]
            up_tensor = converted.get(f"{base_name}.up")
            if up_tensor is None:
                continue
            concat_dim = 0 if tensor.ndim == 2 else 1
            fused[base_name] = torch.cat([tensor, up_tensor], dim=concat_dim)
        return fused

    def _validate_patterns(self, *args, **kwargs):
        # Allow number of wildcards to mismatch in this mapping.
        pass


class GLMExpertDownProjMapping(AutoMapping):
    """Mapping for fused expert down projection weights."""

    def hf_to_megatron(self, hf_weights: torch.Tensor, megatron_module: torch.nn.Module) -> torch.Tensor:
        global_expert_number = extract_expert_number_from_param(self.megatron_param)
        expert_weight = _select_expert_weight(hf_weights, global_expert_number)

        normalized_param = self._normalize_expert_param_name(self.megatron_param)
        _, target_param = get_module_and_param_from_name(megatron_module, normalized_param)
        expert_weight = _align_weight_to_shape(expert_weight, target_param.shape, "down_proj")
        return super().hf_to_megatron(expert_weight, megatron_module)

    def _validate_patterns(self, *args, **kwargs):
        # Allow number of wildcards to mismatch in this mapping.
        pass


class GLMKVUpProjMapping(ColumnParallelMapping):
    """
    Independent mapping: Specifically handles dimension splitting for HF kv_b_proj -> Mcore k_up/v_up.
    Does not affect any common logic.
    """

    # Class-level cache for coordinating k_up and v_up parts
    _kv_cache = {}

    def __init__(self, megatron_param: str, hf_param: str):
        super().__init__(megatron_param, hf_param)
        # Determine if current parameter is k_up or v_up
        self.is_k_up = "linear_k_up_proj" in megatron_param
        self.is_v_up = "linear_v_up_proj" in megatron_param

    @classmethod
    def reset_cache(cls):
        """Reset the class-level cache.

        This should be called at the beginning of each conversion session
        to prevent stale cache data from previous conversions.
        """
        cls._kv_cache = {}

    def hf_to_megatron(
        self,
        hf_weights: torch.Tensor,
        megatron_module: nn.Module,
        # param_name is not needed, determined internally
    ) -> torch.Tensor:
        # ======================== Original core logic ========================
        # Only process k_up / v_up, skip all other parameters
        if self.is_k_up or self.is_v_up:
            # Reshape -> split -> extract the required part
            hf_weights = hf_weights.reshape(64, 448, -1)
            if self.is_k_up:
                hf_weights = hf_weights.split([192, 256], dim=1)[0]
            else:
                hf_weights = hf_weights.split([192, 256], dim=1)[1]
            hf_weights = hf_weights.view(-1, 512)
        # ====================================================================

        # Finally call the original ColumnParallel for TP splitting
        if self.tp_size == 1:
            return hf_weights
        return super().hf_to_megatron(hf_weights, megatron_module)

    def megatron_to_hf(
        self,
        megatron_weights: Optional[torch.Tensor],
        megatron_module: Optional[nn.Module],
    ) -> Dict[str, torch.Tensor]:
        """Convert from Megatron format back to HF format with dimension merging."""
        # First call parent class to handle TP gathering and PP broadcast
        result = super().megatron_to_hf(megatron_weights, megatron_module)

        # If no result (e.g., not on TP rank 0), return empty
        if not result:
            return result

        # Get the gathered weight if available
        if str(self.hf_param) not in result:
            return result

        weight = result[str(self.hf_param)]

        # Only process k_up and v_up mappings
        if not (self.is_k_up or self.is_v_up):
            return result

        # Create a cache key based on the HF parameter name and layer index
        # We need to extract the layer index from the megatron parameter name
        import re

        # Try to extract layer index from different patterns
        layer_match = None
        patterns = [
            r'layers\.(\d+)',  # decoder.layers.0.self_attention.linear_k_up_proj.weight
            r'model\.layers\.(\d+)',  # model.layers.0.self_attn.kv_b_proj.weight (HF side)
        ]

        for pattern in patterns:
            layer_match = re.search(pattern, self.megatron_param) or re.search(pattern, str(self.hf_param))
            if layer_match:
                break

        if not layer_match:
            # If no layer index found, use a generic key (less likely but possible)
            cache_key = f"{self.hf_param}"
        else:
            layer_idx = layer_match.group(1)
            cache_key = f"{self.hf_param}_layer_{layer_idx}"

        # Reshape weight based on whether this is k_up or v_up
        # The weight from ColumnParallelMapping is 2D: (output_dim, input_dim)
        # Based on hf_to_megatron logic, the weight was reshaped to (-1, 512)
        # So input_dim should be 512
        input_dim = 512

        # Check if weight shape is compatible
        total_elements = weight.numel()
        if self.is_k_up:
            # k_up: original was (64, 192, 512) before flattening to (12288, 512)
            expected_elements = 64 * 192 * input_dim  # 64*192*512 = 6291456
            if total_elements != expected_elements:
                # Try to infer shape dynamically
                if total_elements % (64 * 192) != 0:
                    # Fallback: return as is if shape doesn't match expected pattern
                    return result
                input_dim = total_elements // (64 * 192)
            reshaped_weight = weight.reshape(64, 192, input_dim)
            part_type = "k_up"
        else:  # self.is_v_up
            # v_up: original was (64, 256, 512) before flattening to (16384, 512)
            expected_elements = 64 * 256 * input_dim  # 64*256*512 = 8388608
            if total_elements != expected_elements:
                # Try to infer shape dynamically
                if total_elements % (64 * 256) != 0:
                    # Fallback: return as is if shape doesn't match expected pattern
                    return result
                input_dim = total_elements // (64 * 256)
            reshaped_weight = weight.reshape(64, 256, input_dim)
            part_type = "v_up"

        # Initialize cache for this key if needed
        if cache_key not in self._kv_cache:
            self._kv_cache[cache_key] = {}

        # Store the reshaped weight in cache
        self._kv_cache[cache_key][part_type] = reshaped_weight

        # Check if both k_up and v_up parts are available
        if "k_up" in self._kv_cache[cache_key] and "v_up" in self._kv_cache[cache_key]:
            # Both parts are ready, merge them along dimension 1
            k_up_part = self._kv_cache[cache_key]["k_up"]
            v_up_part = self._kv_cache[cache_key]["v_up"]

            # Verify input dimensions match
            if k_up_part.shape[-1] != v_up_part.shape[-1]:
                # Clean up cache and return original weight as fallback
                del self._kv_cache[cache_key]
                return result

            # Concatenate along dimension 1: (64, 192+256=448, input_dim)
            merged_weight = torch.cat([k_up_part, v_up_part], dim=1)

            # Reshape back to 2D for HF format: (64*448=28672, input_dim)
            # But wait, the original HF weight might have different shape
            # Let's check: original HF weight was reshaped to (64, 448, -1) in hf_to_megatron
            # So we need to flatten first two dimensions
            final_weight = merged_weight.reshape(-1, merged_weight.shape[-1])

            # Clear cache for this key
            del self._kv_cache[cache_key]

            return {str(self.hf_param): final_weight}
        else:
            # Only one part is ready, return empty dict
            # This prevents duplicate entries in the final HF state dict
            # The other mapping will return the merged result when both parts are ready
            return {}
