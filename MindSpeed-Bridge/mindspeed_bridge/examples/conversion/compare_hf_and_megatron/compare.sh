export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONPATH=./Megatron-Bridge/src/:$PYTHONPATH
export TORCHDYNAMO_DISABLE=1
export TORCHDYNAMO_SUPPRESS_ERRORS=1
export TORCH_DEVICE_CUDA_IGNORE_PARITY_ERROR=1


mkdir -p logs
torchrun --master_port=29501 --nproc-per-node=4 examples/conversion/compare_hf_and_megatron/compare.py \
    --hf_model_path "/mnt/hf_weights/Qwen3.5-122B-A10B" \
    --model_class "Qwen3_5MoeForConditionalGeneration" \
    --prompt "hello, please describe this picture ? haha" \
    --image_path  "demo.jpeg" \
    --enable_debug_hooks \
    --tp 1 \
    --pp 2 \
    --ep 2 \
    2>&1 | tee logs/compare_$(date +%Y%m%d_%H%M).log