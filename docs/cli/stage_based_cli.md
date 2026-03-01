# Stage Based CLI - 单 Stage 部署指南

本文档介绍如何使用 vLLM-Omni 的 Stage Based CLI 进行单 Stage 部署。

## 前置条件

- Python 3.10+
- CUDA 12.0+（GPU 推理）
- vLLM 0.16.0+

### 安装

```bash
pip install vllm-omni
```

## 概述

Stage Based CLI 是 vLLM-Omni 提供的多阶段推理命令行工具。对于大多数用户来说，单 Stage 部署是最简单直接的使用方式：

- **Diffusion 模型**（如 Qwen-Image、Wan 2.2）：自动识别为单 Stage
- **单卡 LLM 模型**：默认单 Stage 运行
- **分布式部署**：通过 `--stage-id` 指定运行特定 Stage

## 快速开始

### 1. Diffusion 模型（图像/视频生成）

```bash
vllm serve Qwen/Qwen-Image --omni --port 8091
```

Diffusion 模型会自动识别为单 Stage 架构，无需额外配置。

### 2. Omni LLM 模型（文本/音频）

```bash
vllm serve Qwen/Qwen2.5-Omni-7B --omni --port 8091
```

对于多模态 LLM 模型，默认会在单卡上启动所有 Stage（如果显存足够）。

> **支持版本**：vLLM-Omni 支持 Qwen2.5-Omni 和 Qwen3-Omni 系列。示例中使用 Qwen2.5-Omni-7B，如需使用 Qwen3-Omni 请替换模型名称。

### 3. 使用自定义 Stage 配置

```bash
vllm serve Qwen/Qwen2.5-Omni-7B --omni \
  --stage-configs-path ./my_config.yaml \
  --port 8091
```

### 4. 验证部署

启动服务后，可以通过以下方式验证：

```bash
# 查看可用模型
curl http://localhost:8091/v1/models

# 发送测试请求
curl http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-Omni-7B",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## CLI 参数详解

### 核心参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--omni` | flag | - | 启用 vLLM-Omni 模式（必需） |
| `--stage-configs-path` | str | None | Stage 配置文件路径，未指定则使用模型默认配置 |
| `--stage-id` | int | None | 指定启动的 Stage ID（用于分布式或调试） |
| `--init-timeout` | int | 600 | 所有 Stage 初始化超时时间（秒） |
| `--stage-init-timeout` | int | 300 | 单个 Stage 初始化超时时间（秒） |
| `--shm-threshold-bytes` | int | 65536 | 共享内存传输阈值（字节） |

### 分布式参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--headless` | flag | - | 启用无头模式（Worker 节点） |
| `--omni-master-address` | str | None | Master 节点 IP 地址 |
| `--omni-master-port` | int | None | Master 节点端口 |
| `--worker-backend` | str | 自动选择 | Worker 后端。普通模式默认自动选择；headless 模式需设为 `multi_process` |

### Diffusion 模型参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--vae-use-slicing` | flag | - | 启用 VAE slicing，减少显存占用 |
| `--vae-use-tiling` | flag | - | 启用 VAE tiling，适用于大分辨率图像 |
| `--enable-cpu-offload` | flag | - | 启用 CPU offload，将部分计算移至 CPU |
| `--enable-layerwise-offload` | flag | - | 启用逐层 offload，进一步节省显存 |
| `--cfg-parallel-size` | int | 1 | CFG 并行数（1 或 2），用于加速生成 |
| `--boundary-ratio` | float | None | DiT 层分割比例（视频模型，如 Wan 2.2） |
| `--flow-shift` | float | None | 调度器 flow_shift 参数（视频模型） |
| `--max-generated-image-size` | int | None | 最大生成图像尺寸（宽 × 高） |

### 采样参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--default-sampling-params` | str | None | 默认采样参数（JSON 格式，详见下方说明） |
| `--tts-max-instructions-length` | int | 500 | TTS 语音风格指令最大长度 |

#### `--default-sampling-params` 说明

该参数接收 JSON 格式字符串，**键为 stage_id**，值为该 Stage 的采样参数：

```bash
--default-sampling-params '{"0": {"num_inference_steps": 50, "guidance_scale": 7.5}}'
```

- `"0"`：表示 Stage ID 为 0（Diffusion 模型通常只有一个 Stage）
- `num_inference_steps`：推理步数，越多质量越高但速度越慢
- `guidance_scale`：引导系数，控制生成结果与提示词的匹配程度

## 使用场景

### 场景 1：单卡运行 Diffusion 模型

```bash
vllm serve Qwen/Qwen-Image --omni \
  --port 8091 \
  --gpu-memory-utilization 0.9
```

### 场景 2：显存不足时启用 Offload

```bash
vllm serve Qwen/Qwen-Image --omni \
  --port 8091 \
  --enable-cpu-offload \
  --vae-use-slicing
```

### 场景 3：大分辨率图像生成

```bash
vllm serve Qwen/Qwen-Image --omni \
  --port 8091 \
  --vae-use-tiling \
  --max-generated-image-size 4096
```

### 场景 4：自定义采样参数

```bash
vllm serve Qwen/Qwen-Image --omni \
  --port 8091 \
  --default-sampling-params '{"0": {"num_inference_steps": 50, "guidance_scale": 7.5}}'
```

### 场景 5：分布式部署 - 指定 Stage

在分布式场景下，可以单独启动某个 Stage：

**Master 节点（Stage 0）：**
```bash
vllm serve Qwen/Qwen2.5-Omni-7B --omni \
  --port 8091 \
  --stage-id 0
```

**Worker 节点（Stage 1）：**
```bash
vllm serve Qwen/Qwen2.5-Omni-7B --omni \
  --headless \
  --worker-backend multi_process \
  --stage-id 1 \
  --omni-master-address 192.168.1.100 \
  --omni-master-port 8091
```

> ⚠️ **注意**：Headless 模式需要设置 `--worker-backend multi_process`，且需要同时指定 `--omni-master-address` 和 `--omni-master-port`。

### 场景 6：使用特定 GPU

通过环境变量控制 GPU 可见性：

```bash
CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen-Image --omni --port 8091
```

## Stage 配置文件

如果需要自定义 Stage 配置，可以创建 YAML 文件：

```yaml
# my_stage_config.yaml
stage_args:
  - stage_id: 0
    stage_type: diffusion
    runtime:
      devices: "0"
      max_batch_size: 8
    engine_args:
      gpu_memory_utilization: 0.9
      enforce_eager: false
    default_sampling_params:
      num_inference_steps: 28
      guidance_scale: 3.5
```

使用方式：
```bash
vllm serve Qwen/Qwen-Image --omni \
  --stage-configs-path ./my_stage_config.yaml \
  --port 8091
```

## 配置文件字段说明

### `stage_args` 列表项

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `stage_id` | int | ✅ | Stage 唯一标识符 |
| `stage_type` | str | ✅ | Stage 类型：`llm` 或 `diffusion` |
| `runtime.devices` | str | - | GPU 设备，如 `"0"` 或 `"0,1"` |
| `runtime.max_batch_size` | int | - | 最大批处理大小 |
| `engine_args` | dict | - | 传递给引擎的参数 |
| `default_sampling_params` | dict | - | 默认采样参数 |
| `final_output` | bool | - | 是否为最终输出 Stage |
| `final_output_type` | str | - | 输出类型：`text`/`audio`/`image` |

### `engine_args` 常用字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `model_stage` | str | 模型阶段：`thinker`/`talker`/`code2wav` |
| `model_arch` | str | 模型架构名称 |
| `worker_type` | str | Worker 类型：`ar`/`generation` |
| `gpu_memory_utilization` | float | GPU 显存利用率（0-1） |
| `enforce_eager` | bool | 是否强制 eager 模式 |
| `tensor_parallel_size` | int | 张量并行大小 |
| `distributed_executor_backend` | str | 分布式后端：`mp`/`ray` |

## 故障排查

### 1. 初始化超时

```
RuntimeError: Stage-0 initialization timeout
```

**解决方案**：增加超时时间
```bash
vllm serve Qwen/Qwen-Image --omni \
  --stage-init-timeout 600 \
  --init-timeout 1200
```

### 2. 显存不足 (OOM)

```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**解决方案**：
- 降低 `--gpu-memory-utilization`
- 启用 `--enable-cpu-offload`
- 启用 `--vae-use-slicing` 或 `--vae-use-tiling`

### 3. Headless 模式握手失败

```
RuntimeError: Handshake timeout for stage-1
```

**检查项**：
- Master 节点是否已启动
- IP 地址和端口是否正确
- 网络是否连通
- 防火墙是否放行端口

### 4. Stage ID 不存在

```
ValueError: No stage matches stage_id=3
```

**解决方案**：检查配置文件中是否定义了该 Stage ID

### 5. 服务启动后无法访问

**检查项**：
- 确认端口未被占用：`lsof -i :8091`
- 确认防火墙允许访问
- 使用 `curl http://localhost:8091/v1/models` 验证本地访问

## 相关文档

- [多 Stage 部署指南](./multi_stage.md)（待编写）
- [分布式部署指南](./distributed.md)（待编写）
- [Stage 配置文件参考](./stage_config_reference.md)（待编写）

## 反馈与贡献

如有问题或建议，请在 [GitHub Issues](https://github.com/vllm-project/vllm-omni/issues) 中反馈。
