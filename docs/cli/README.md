# vLLM-Omni CLI Guide

The CLI for vLLM-Omni inherits from vllm with some additional arguments.

## serve

Starts the vLLM-Omni OpenAI Compatible API server.

Start with a model:

```bash
vllm serve Qwen/Qwen2.5-Omni-7B --omni
```

Specify the port:

```bash
vllm serve Qwen/Qwen2.5-Omni-7B --omni --port 8091
```

If you have custom stage configs file, launch the server with command below
```bash
vllm serve Qwen/Qwen2.5-Omni-7B --omni --stage-configs-path /path/to/stage_configs_file
```


## bench

Run benchmark tests for online serving throughput.
Available Commands:

```bash
vllm bench serve --omni \
    --model Qwen/Qwen2.5-Omni-7B \
    --host server-host \
    --port server-port \
    --random-input-len 32 \
    --random-output-len 4  \
    --num-prompts  5
```

See [vllm bench serve](./bench/serve.md) for the full reference of all available arguments.

## Stage Based CLI

vLLM-Omni supports multi-stage inference architecture through Stage Based CLI. This allows complex multi-modal models to be split into independent stages that can run on different GPUs with independent configurations.

See [Stage Based CLI Guide](./stage_based_cli.md) for detailed documentation on:
- Single stage deployment
- CLI parameters
- Configuration file format
- Distributed deployment
