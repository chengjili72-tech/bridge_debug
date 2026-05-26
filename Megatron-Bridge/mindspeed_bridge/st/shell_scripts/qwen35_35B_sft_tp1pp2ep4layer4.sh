#!/bin/bash
set -euo pipefail

WORKSPACE="$(pwd)"
export PYTHONPATH=${WORKSPACE}/src:${WORKSPACE}/mindspeed_bridge:$PYTHONPATH
export HF_PATH="/home/hf_weights/Qwen3.5-35B-A3B"

NPUS_PER_NODE=8
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

TP=1
PP=2
EP=4
CP=1
CP_TYPE="kvallgather_cp_algo"
NUM_LAYERS=4

RECIPE="qwen35_vl_35b_a3b_sft_config"
SEQ_LENGTH=4096
TRAIN_ITERS=15
GLOBAL_BATCH_SIZE=16
MICRO_BATCH_SIZE=1
LOG_INTERVAL=1
TRAIN_JSON="/home/ci_dataset/cord_v2_data/train.jsonl"
TRAIN_IMAGES="/home/ci_dataset/cord_v2_data/images"


CLI_OVERRIDES="\
    model.seq_length=$SEQ_LENGTH \
    train.train_iters=$TRAIN_ITERS \
    train.global_batch_size=$GLOBAL_BATCH_SIZE \
    train.micro_batch_size=$MICRO_BATCH_SIZE \
    checkpoint.save=${WORKSPACE}/results/${RECIPE}_sft \
    dataset.train_data_path=${TRAIN_JSON} \
    dataset.image_folder=${TRAIN_IMAGES} \
    logger.log_interval=$LOG_INTERVAL \
    model.tensor_model_parallel_size=${TP} \
    model.pipeline_model_parallel_size=${PP} \
    model.expert_model_parallel_size=${EP} \
    model.context_parallel_size=${CP} \
    model.context_parallel_algo=${CP_TYPE} \
    model.num_layers=${NUM_LAYERS}  \
"
torchrun $DISTRIBUTED_ARGS scripts/training/run_recipe.py \
    --recipe $RECIPE \
    --step_func qwen3_vl_step \
    $CLI_OVERRIDES