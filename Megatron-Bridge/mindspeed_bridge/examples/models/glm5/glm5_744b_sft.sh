#!/bin/bash
set -euo pipefail

WORKSPACE="$(pwd)"
export PYTHONPATH=${WORKSPACE}/src:${WORKSPACE}/mindspeed_bridge:$PYTHONPATH
export HF_PATH="your hf path"

NPUS_PER_NODE=16
MASTER_ADDR=localhost
MASTER_PORT=6015
NNODES=32
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
PP=8
EP=64
CP=1
CP_TYPE="kvallgather_cp_algo"
NUM_LAYERS=80
MTP_LAYERS=1

RECIPE="glm5_744b_sft_config"
SEQ_LENGTH=4096
TRAIN_ITERS=2000
GLOBAL_BATCH_SIZE=2048
MICRO_BATCH_SIZE=1
LOG_INTERVAL=1


CLI_OVERRIDES="\
    model.seq_length=$SEQ_LENGTH \
    train.train_iters=$TRAIN_ITERS \
    train.eval_iters=0 \
    dataset.seq_length=$SEQ_LENGTH \
    train.global_batch_size=$GLOBAL_BATCH_SIZE \
    train.micro_batch_size=$MICRO_BATCH_SIZE \
    checkpoint.save=${WORKSPACE}/results/${RECIPE}_sft \
    logger.log_interval=$LOG_INTERVAL \
    model.tensor_model_parallel_size=${TP} \
    model.pipeline_model_parallel_size=${PP} \
    model.expert_model_parallel_size=${EP} \
    model.context_parallel_size=${CP} \
    model.context_parallel_algo=${CP_TYPE} \
    model.num_layers=${NUM_LAYERS} \
    model.mtp_num_layers=${MTP_LAYERS}"


torchrun $DISTRIBUTED_ARGS scripts/training/run_recipe.py \
    --recipe $RECIPE \
    --step_func gpt_step \
    $CLI_OVERRIDES | tee ./logs/glm5_744b_sft.log
