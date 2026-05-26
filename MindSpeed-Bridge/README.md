# MindSpeed-Bridge

[Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) 是 NeMo Framework 下的 PyTorch 原生库，为主流 LLM/VLM 模型提供预训练、SFT 和 LoRA 能力。在训练方面，基于 Megatron Core 构建高性能可扩展的训练循环，支持张量并行、流水线并行及混合精度训练（FP8、BF16 等）；在模型转换方面，作为 Hugging Face 与 Megatron Core 之间的桥接与验证层，支持双向 checkpoint 转换，便于其他项目接入 Megatron Core 的并行能力或将模型导出至各类推理引擎，同时内置验证机制保障跨格式转换的精度与完整性。

**MindSpeed-Bridge** 在 Megatron-Bridge 基础上，面向昇腾平台额外提供：

- **更多开源模型支持**：扩展昇腾平台上的模型覆盖范围
- **精度验证**：经过昇腾 NPU 平台的精度对齐与验证
- **亲和性能优化**：针对昇腾硬件特性的性能调优

## 最新消息

- 🔥 **2026-05**：MindSpeed-Bridge 仓库创建，提供 Qwen3.5-VL 模型的训练与适配支持


## 安装

[安装指南](docs/zh/install.md)
