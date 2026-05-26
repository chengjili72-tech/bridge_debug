#!/bin/bash
set -euo pipefail

WORKSPACE="$(pwd)"
export PYTHONPATH=${WORKSPACE}/src:${WORKSPACE}/mindspeed_bridge:$PYTHONPATH
export HF_PATH="your hf path"

NPUS_PER_NODE=16
MASTER_ADDR=localhost
MASTER_PORT=6015
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

mkdir -p ./logs

TP=2
PP=2
EP=8
CP=1
CP_TYPE="kvallgather_cp_algo"
NUM_LAYERS=40
MTP_LAYERS=1

RECIPE="qwen35_vl_35b_a3b_sft_config"
SEQ_LENGTH=4096
TRAIN_ITERS=2000
SAVE_INTERVAL=2000
GLOBAL_BATCH_SIZE=16
MICRO_BATCH_SIZE=1
LOG_INTERVAL=1
TRAIN_JSON="your train.jsonl"
TRAIN_IMAGES="your images"

LR=1e-6
MIN_LR=1e-7
LR_WARMUP_ITERS=500

CLI_OVERRIDES="\
    model.seq_length=$SEQ_LENGTH \
    train.train_iters=$TRAIN_ITERS \
    train.global_batch_size=$GLOBAL_BATCH_SIZE \
    train.micro_batch_size=$MICRO_BATCH_SIZE \
    checkpoint.save=${WORKSPACE}/results/${RECIPE}_sft \
    checkpoint.save_interval=${SAVE_INTERVAL} \
    dataset.train_data_path=${TRAIN_JSON} \
    dataset.image_folder=${TRAIN_IMAGES} \
    logger.log_interval=$LOG_INTERVAL \
    model.tensor_model_parallel_size=${TP} \
    model.sequence_parallel=True \
    model.pipeline_model_parallel_size=${PP} \
    model.expert_model_parallel_size=${EP} \
    model.context_parallel_size=${CP} \
    model.context_parallel_algo=${CP_TYPE} \
    model.num_layers=${NUM_LAYERS} \
    model.mtp_num_layers=${MTP_LAYERS} \
    optimizer.lr=${LR} \
    optimizer.min_lr=${MIN_LR} \
    scheduler.lr_warmup_iters=${LR_WARMUP_ITERS} \
    ++model.use_ascend_gdn=True \
    "


torchrun $DISTRIBUTED_ARGS scripts/training/run_recipe.py \
    --recipe $RECIPE \
    --step_func qwen3_vl_step \
    $CLI_OVERRIDES | tee ./logs/qwen35_vl_35b_sft_TP${TP}_PP${PP}_EP${EP}_CP${CP}_NUM_LAYERS${NUM_LAYERS}.log
