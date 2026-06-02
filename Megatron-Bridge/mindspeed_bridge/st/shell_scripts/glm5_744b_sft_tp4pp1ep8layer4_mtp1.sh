#!/bin/bash
set -euo pipefail

WORKSPACE="$(pwd)"
export PYTHONPATH=${WORKSPACE}/src:${WORKSPACE}/mindspeed_bridge:${PYTHONPATH:-}:/home/c00842080/Megatron-LM:/home/c00842080/MindSpeed
export HF_PATH="/home/g00679814/bridge/MindSpeed-LLM/glm5_4ceng_mtp1_hf"
#export CUDA_DEVICE_MAX_CONNECTIONS=1
#export HCCL_DETERMINISTIC=true
#export CUDA_HOME=/usr/local/cuda-12.2
#export PATH=${CUDA_HOME}/bin:${PATH}
#export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}

####
#export BRIDGE_DEBUG_CONVERSION=1
#export BRIDGE_DEBUG_CONVERSION_MAX=300
#export BRIDGE_STRICT_NONE_TASK=0
###


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
DATA_FILE_PATH=/home/g00679814/bridge/glm5_bridge/squad/squad_train_flat.jsonl
DATA_CACHE_PATH="${DATA_FILE_PATH%/*}"


CLI_OVERRIDES="\
    model.seq_length=$SEQ_LENGTH \
    rng.seed=5678 \
    rng.data_parallel_random_init=false \
    dataset.seed=5678 \
    dataset.dataset_name=json \
    dataset.dataloader_type=batch \
    dataset.seq_length=$SEQ_LENGTH \
    ++dataset.hf_kwargs.data_files.train=${DATA_FILE_PATH} \
    dataset.split_val_from_train=true \
    ++dataset.data_kwargs.pad_to_max_length=true \
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
    model.gradient_accumulation_fusion=false \
    model.moe_permute_fusion=false \
    model.use_fused_lightning_indexer=false \
    model.use_fused_sparse_flash_attention=false \
    model.use_fused_lightning_indexer_kl_loss=false \
    model.mtp_num_layers=${MTP_LAYERS}"

logfile=$(date +%Y%m%d)_$(date +%H%M%S)

torchrun $DISTRIBUTED_ARGS scripts/training/run_recipe.py \
    --recipe $RECIPE \
    --step_func gpt_step \
    $CLI_OVERRIDES \
    2>&1 | tee ./logs/glm5_744b_sft_${logfile}.log
