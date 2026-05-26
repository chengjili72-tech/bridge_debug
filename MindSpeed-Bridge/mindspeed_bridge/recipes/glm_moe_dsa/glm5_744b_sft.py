import os
import torch

from megatron.bridge.recipes.common import _sft_common
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.recipes.utils.finetune_utils import default_squad_config
from mindspeed_bridge.recipes.glm_moe_dsa.glm_moe_dsa import (
    _glm_moe_dsa_apply_common,
    _glm_moe_dsa_apply_moe,
    _glm_moe_dsa_enable_recompute,
)


# =============================================================================
# MindSpeed global patch
# =============================================================================
def _apply_mindspeed_args_patch(patch_dict: dict):
    """
    Dynamically inject missing or overridden parameters into Megatron
    and MindSpeed global arguments by intercepting lower-level argument getters.
    """
    import mindspeed.args_utils as ms_args_utils

    if hasattr(ms_args_utils, 'get_mindspeed_args') and ms_args_utils.get_mindspeed_args.__name__ != 'safe_ms_get_args':
        orig_ms_get_args = ms_args_utils.get_mindspeed_args

        def safe_ms_get_args():
            args = orig_ms_get_args()
            if args is not None:
                # Force override argument values
                for key, value in patch_dict.items():
                    setattr(args, key, value)
            return args

        ms_args_utils.get_mindspeed_args = safe_ms_get_args


# =============================================================================
# GLM5 744B SFT Configuration (MoE)
# =============================================================================
def glm5_744b_sft_config(
    hf_path: str = "GLM/GLM5-744B/",
) -> ConfigContainer:
    """Return a full SFT config for GLM5 744B(MoE).

    Default configuration: 2 nodes, 16 GPUs
    - TP=2, PP=1, EP=16
    - LR=2e-5 (full SFT)
    - Sequence length: 4096

    Args:
        hf_path: HuggingFace model ID or local path to model directory.
    """
    cfg = _sft_common()
    hf_path = os.getenv("HF_PATH", hf_path)
    _glm_moe_dsa_apply_common(cfg, hf_path, tp=2, pp=2, max_lr=2e-5, min_lr=2e-6)
    _glm_moe_dsa_apply_moe(cfg, ep=4)
    _glm_moe_dsa_enable_recompute(cfg)
    # Parallel settings
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.mtp_num_layers = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = True
    cfg.model.calculate_per_token_loss = False
    cfg.model.use_flash_attn = True

    # MOE settings
    cfg.model.moe_router_load_balancing_type = "aux_loss"
    cfg.model.moe_permute_fusion = True

    # Training config
    cfg.train.train_iters = 10
    cfg.train.global_batch_size = 8
    cfg.train.micro_batch_size = 1

    # VLM Model freeze config
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = True
    cfg.model.freeze_vision_projection = False

    # Optim featrue config
    cfg.model.cross_entropy_loss_fusion = False
    # Log settings
    cfg.logger.log_interval = 1

    # Load and save config
    cfg.save_interval = cfg.train.train_iters
    # Dataset config
    cfg.dataset = default_squad_config(
        seq_length=cfg.dataset.seq_length,
        packed_sequence=False,
        pad_seq_to_mult=1,
    )
    # NPU config
    cfg.model.max_position_embeddings = cfg.model.yarn_original_max_position_embeddings
    cfg.model.use_dsa_absorb = True
    cfg.model.use_fused_lightning_indexer = True
    cfg.model.use_fused_sparse_flash_attention = True
    cfg.model.use_fused_lightning_indexer_kl_loss = True
    cfg.model.multi_latent_attention = True

    mindspeed_missing_args = {
        'reset_position_ids': False,
    }
    _apply_mindspeed_args_patch(mindspeed_missing_args)
    return cfg
