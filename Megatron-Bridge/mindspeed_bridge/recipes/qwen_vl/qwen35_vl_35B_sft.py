import os
import torch

from megatron.bridge.recipes.common import _sft_common
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.data.vlm_datasets.preloaded_provider import (
    PreloadedVLMConversationProvider,
)
from mindspeed_bridge.recipes.qwen_vl.qwen35_vl import (
    _qwen35_vl_apply_common,
    _qwen35_vl_apply_moe,
    _qwen35_vl_enable_recompute,
)


# =============================================================================
# Qwen3.5-VL 35B-A3B SFT Configuration (MoE)
# =============================================================================
def qwen35_vl_35b_a3b_sft_config(
    hf_path: str = "Qwen/Qwen3.5-35B-A3B/",
) -> ConfigContainer:
    """Return a full SFT config for Qwen3.5-VL 35B-A3B (MoE).

    Default configuration: 2 nodes, 16 GPUs
    - TP=2, PP=1, EP=16
    - LR=2e-5 (full SFT)
    - Sequence length: 4096

    Args:
        hf_path: HuggingFace model ID or local path to model directory.
    """
    cfg = _sft_common()
    hf_path = os.getenv("HF_PATH", hf_path)
    _qwen35_vl_apply_common(cfg, hf_path, tp=1, pp=1, max_lr=2e-5, min_lr=2e-6)
    _qwen35_vl_apply_moe(cfg, ep=8)
    _qwen35_vl_enable_recompute(cfg)
    # Parallel settings
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.mtp_num_layers = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.model.calculate_per_token_loss = False
    cfg.model.use_flash_attn = True

    # MOE settings
    cfg.model.moe_router_fusion = False
    cfg.model.moe_permute_fusion = False
    cfg.model.moe_router_load_balancing_type = "aux_loss"

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
    dataset_cfg = PreloadedVLMConversationProvider(
        seq_length=cfg.model.seq_length,
        hf_processor_path=hf_path,
        train_data_path="train.jsonl",
        image_folder="train images",
        num_workers=4,
        dataloader_type="single",
        data_sharding=True,
        pin_memory=True,
        persistent_workers=False,
        pack_sequences_in_batch=False,
    )
    cfg.dataset = dataset_cfg

    # NPU featrue config
    cfg.model.use_triton_gdn = False
    cfg.model.use_ascend_gdn = False

    return cfg
