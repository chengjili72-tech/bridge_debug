#!/bin/bash
set -euo pipefail

WORKSPACE="$(pwd)"
export PYTHONPATH=${WORKSPACE}/src:${WORKSPACE}/mindspeed_bridge:$PYTHONPATH
export HF_PATH="/home/hf_weights/glm5_4ceng_mtp1_hf"

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
export BRIDGE_DETERMINISM_DEBUG=${BRIDGE_DETERMINISM_DEBUG:-1}
export BRIDGE_DETERMINISM_DEBUG_INTERVAL=${BRIDGE_DETERMINISM_DEBUG_INTERVAL:-1}
# For sequence parallel path, keeping max connections to 1 improves scheduling stability.
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

TP=4
PP=1
EP=8
CP=1
CP_TYPE="ulysses_cp_algo"
NUM_LAYERS=4
MTP_LAYERS=1

RECIPE="glm5_744b_sft_config"
SEQ_LENGTH=1024
TRAIN_ITERS=15
GLOBAL_BATCH_SIZE=8
MICRO_BATCH_SIZE=1
LOG_INTERVAL=1
DATA_FILE_PATH=/home/ci_dataset/squad/squad_train_flat.jsonl
DATA_CACHE_PATH="${DATA_FILE_PATH%/*}"


CLI_OVERRIDES="\
    model.seq_length=$SEQ_LENGTH \
    rng.seed=5678 \
    rng.data_parallel_random_init=false \
    dataset.seed=5678 \
    dataset.dataset_name=json \
    dataset.dataloader_type=single \
    dataset.seq_length=$SEQ_LENGTH \
    ++dataset.hf_kwargs.data_files.train=${DATA_FILE_PATH} \
    dataset.split_val_from_train=true \
    dataset.dataset_root=${DATA_CACHE_PATH} \
    train.train_iters=$TRAIN_ITERS \
    train.eval_iters=0 \
    train.global_batch_size=$GLOBAL_BATCH_SIZE \
    train.micro_batch_size=$MICRO_BATCH_SIZE \
    checkpoint.save=null \
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
