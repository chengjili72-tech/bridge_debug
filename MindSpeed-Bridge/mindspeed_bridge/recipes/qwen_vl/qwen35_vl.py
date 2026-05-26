# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Qwen3.5-VL recipes.

This module provides pretrain, SFT, and PEFT configurations for Qwen3.5-VL models:

- **Dense**: 800M, 2B, 4B, 9B, 27B
- **MoE**: 35B-A3B, 122B-A10B, 397B-A17B
"""

from __future__ import annotations

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.peft.base import PEFT
from megatron.bridge.recipes.utils.finetune_utils import default_peft_config
from megatron.bridge.recipes.utils.optimizer_utils import (
    distributed_fused_adam_with_cosine_annealing,
)
from megatron.bridge.training.config import ConfigContainer


# =============================================================================
# Internal helpers — shared configuration for all Qwen3.5-VL recipes
# =============================================================================


def _qwen35_vl_apply_common(
    cfg: ConfigContainer,
    hf_path: str,
    *,
    tp: int,
    pp: int,
    max_lr: float,
    min_lr: float,
    gbs: int = 32,
) -> None:
    """Apply settings shared across all Qwen3.5-VL SFT and PEFT recipes.

    Sets model, parallelism (except EP/SP for MoE), VLM freeze, MTP, TE,
    CUDA graphs, kernels, memory-saving defaults, training, optimizer,
    dataset, DDP, and mixed-precision options.
    """
    # Model configuration
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider()
    cfg.tokenizer.tokenizer_model = hf_path
    cfg.model.seq_length = 4096

    # Parallel settings (dense defaults; MoE overrides EP/SP via _apply_moe)
    cfg.model.tensor_model_parallel_size = tp
    cfg.model.pipeline_model_parallel_size = pp
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # VLM-specific settings
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = False
    cfg.model.freeze_vision_projection = False

    # MTP (Multi-Token Prediction) — auto-detected from HF config (mtp_num_hidden_layers=1).
    # Set to None to finetune without MTP loss.
    cfg.model.mtp_num_layers = 1
    cfg.model.mtp_loss_scaling_factor = 0.1

    # TE / Transformer implementation
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph settings
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = "auto"
    cfg.model.gradient_accumulation_fusion = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # Memory saving (disabled by default)
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Training config
    cfg.train.train_iters = 300000
    cfg.train.global_batch_size = gbs
    cfg.train.micro_batch_size = 4  # tested on Blackwell GPUs; reduce for smaller VRAM
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100

    # Optimizer
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=200,
        lr_decay_iters=300000,
        max_lr=max_lr,
        min_lr=min_lr,
    )
    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg

    # Optimizer precision settings
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Dataset configuration
    cfg.dataset.seq_length = 4096
    cfg.dataset.hf_processor_path = hf_path
    cfg.dataset.pack_sequences_in_batch = False

    # DDP settings
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"

    # Norm settings
    cfg.layernorm_zero_centered_gamma = True
    cfg.rmsnorm_zerocenter = True

    # Mixed precision
    cfg.mixed_precision = "bf16_mixed"


def _qwen35_vl_apply_moe(cfg: ConfigContainer, *, ep: int, etp: int = 1) -> None:
    """Apply MoE-specific settings on top of the common configuration.

    Enables expert parallelism, sequence parallelism, token dispatcher,
    MoE kernels, and sets MoE-specific overlap / balance / FP8-padding defaults.
    """
    cfg.model.expert_model_parallel_size = ep
    cfg.model.expert_tensor_parallel_size = etp
    cfg.model.sequence_parallel = True

    # MoE dispatcher (alltoall is the standard choice for EP>1)
    cfg.model.moe_token_dispatcher_type = "alltoall"  # nosec B105

    # MoE kernel selections
    cfg.model.moe_router_fusion = True
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True

    # MoE overlap
    cfg.model.moe_shared_expert_overlap = False

    # MoE force balance
    cfg.model.moe_router_force_load_balancing = False

    # MoE FP8 padding
    cfg.model.moe_router_padding_for_fp8 = False

    # Comm overlap settings
    cfg.comm_overlap = None


def _qwen35_vl_enable_recompute(cfg: ConfigContainer) -> None:
    """Enable activation recomputation for large models."""
    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_method = "uniform"
    cfg.model.recompute_num_layers = 1


def _qwen35_vl_apply_peft_scheme(cfg: ConfigContainer, peft_scheme: str | PEFT) -> None:
    """Resolve and apply the PEFT scheme (LoRA, DoRA, or custom)."""
    if isinstance(peft_scheme, str) and peft_scheme.lower() in ["lora", "dora"]:
        cfg.peft = default_peft_config(peft_scheme)
    else:
        cfg.peft = peft_scheme
