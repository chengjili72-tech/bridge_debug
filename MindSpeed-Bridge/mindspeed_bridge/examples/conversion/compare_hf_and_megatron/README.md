# HuggingFace 与 Megatron 模型对比工具

本工具用于比较 HuggingFace 模型与其 Megatron 等价模型的单步生成输出，支持纯文本 LLM 和视觉语言模型。

## 功能特性

- 对比 HF 和 Megatron 模型的前向传播输出
- 支持纯文本 LLM 和视觉语言模型（VL）
- 多 GPU 支持：张量并行（TP）、流水线并行（PP）、专家并行（EP）
- Debug 钩子支持详细的前向/反向传播检查
- 自动检测模型类型（VL 或纯文本）

## 环境依赖

```bash
pip install torch transformers pandas numpy requests Pillow
pip install qwen_vl_utils  # 视觉语言模型必需
```

## 快速开始

### 纯文本 LLM 对比

```bash
python compare.py \
    --hf_model_path "Qwen/Qwen3-1.7B" \
    --prompt "你好，你好吗？"
```

### 视觉语言模型对比

```bash
python compare.py \
    --hf_model_path "Qwen/Qwen2.5-VL-3B-Instruct" \
    --model_class "Qwen2_5_VLForConditionalGeneration" \
    --image_path "demo.jpeg" \
    --prompt "描述这张图片。"
```

### 多 GPU 对比

```bash
torchrun --nproc_per_node=4 compare.py \
    --hf_model_path "/path/to/model" \
    --prompt "Hello world" \
    --tp 2 \
    --pp 2 \
    --ep 2
```

完整示例参见 [compare.sh](compare.sh)。

## 参数说明

| 参数 | 是否必需 | 默认值 | 说明                                                     |
|------|----------|---------|--------------------------------------------------------|
| `--hf_model_path` | 必需 | - | HuggingFace 模型路径或名称                                    |
| `--prompt` | 必需 | - | 输入文本提示                                                 |
| `--image_path` | 可选 | None | VL 模型的图片路径或 URL                                        |
| `--model_class` | 可选 | AutoModelForCausalLM | VL 模型需要指定 HF 模型类（如 `Qwen2_5_VLForConditionalGeneration`） |
| `--megatron_model_path` | 可选 | None | 已转换的 Megatron checkpoint 路径                            |
| `--tp` | 可选 | 1 | 张量并行大小                                                 |
| `--pp` | 可选 | 1 | 流水线并行大小                                                |
| `--ep` | 可选 | 1 | 专家并行大小                                                 |
| `--etp` | 可选 | 1 | 专家张量并行大小                                               |
| `--enable_debug_hooks` | 可选 | False | 启用前向/反向传播日志                                            |
| `--roundtrip_hf` | 可选 | False | 从 Megatron 导出 HF 权重后对比                                 |
| `--exported_hf_dir` | 可选 | 当前目录 | 导出 HF 模型的保存目录                                          |
| `--trust_remote_code` | 可选 | False | 信任 HF 模型的远程代码                                          |

## 输出指标

工具输出以下对比指标：

- **Token 预测**：两个模型的下一个 token
- **Logits 统计**：均值和标准差
- **Top-5 tokens**：置信度最高的 5 个预测
- **余弦相似度**：logits 之间的相似度（阈值：98%）
- **绝对差异**：logits 的最大和平均绝对差异
- **全量输出对比**：完整张量统计（HF output vs Megatron output）

## Debug 模式

启用 `--enable_debug_hooks` 时，会生成 JSONL 日志文件：

- `hf_debug_fwd_log_<world_size>_rank_<rank>.jsonl`：HF 前向传播日志
- `megatron_debug_component_<i>_fwd_log_<world_size>_rank_<rank>.jsonl`：Megatron 各组件日志

每个日志条目包含：
- 模块名称和类型
- 输入张量的形状和统计信息
- 输出张量的形状和统计信息
- 权重摘要

## 使用示例

### 纯文本 LLM（单 GPU）

```bash
python compare.py \
    --hf_model_path "Qwen/Qwen3-1.7B" \
    --prompt "什么是机器学习？"
```

### VL 模型（图片 URL）

```bash
python compare.py \
    --hf_model_path "Qwen/Qwen2.5-VL-3B-Instruct" \
    --model_class "Qwen2_5_VLForConditionalGeneration" \
    --image_path "https://example.com/image.jpg" \
    --prompt "描述这张图片。"
```

### VL 模型（本地图片）

```bash
python compare.py \
    --hf_model_path "Qwen/Qwen2.5-VL-3B-Instruct" \
    --model_class "Qwen2_5_VLForConditionalGeneration" \
    --image_path "/path/to/local/image.jpg" \
    --prompt "你看到了什么？"
```

### 多 GPU（TP 和 PP）

```bash
torchrun --nproc_per_node=2 compare.py \
    --hf_model_path "Qwen/Qwen3-1.7B" \
    --prompt "Hello world" \
    --tp 2 \
    --pp 1
```

### MoE 模型（专家并行）

```bash
torchrun --nproc_per_node=4 compare.py \
    --hf_model_path "/path/to/moe_model" \
    --prompt "你好" \
    --tp 1 \
    --pp 2 \
    --ep 2
```

### 使用已转换的 Megatron checkpoint

```bash
python compare.py \
    --hf_model_path "Qwen/Qwen3-1.7B" \
    --megatron_model_path "/path/to/megatron/checkpoint" \
    --prompt "Hello world"
```

### 回环导出对比

```bash
python compare.py \
    --hf_model_path "Qwen/Qwen3-1.7B" \
    --prompt "Hello world" \
    --roundtrip_hf \
    --exported_hf_dir "/path/to/export/dir"
```

## 自定义 Bridge 支持

对于自定义模型，运行前需导入对应的 bridge：

```python
from mindspeed_bridge.models.qwen_vl.qwen35_vl_bridge import Qwen35VLMoEBridge
```

## 注意事项

1. 工具会根据模型名称/配置自动检测 VL 模型
2. VL 模型未指定 `--model_class` 时会警告并回退到 AutoModelForCausalLM
3. 余弦相似度阈值为 98%（容差 2%）
4. Megatron 可能会为 GPU kernel 效率填充 vocab_size，对比时会截断到 HF vocab_size
5. 使用 `--enable_debug_hooks` 可调试模型转换问题

## 文件说明

- `compare.py`：主对比脚本
- `compare.sh`：多 GPU 运行示例脚本
- `debugger.py`：前向/反向传播日志钩子模块