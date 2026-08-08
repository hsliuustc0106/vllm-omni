# Platform, Kernel, and Quantization Checks

Use as an overlay for accelerator platforms, kernels, attention backends,
quantization, precision, scales, or vendor-specific runners and connectors.

Official docs: [installation](https://docs.vllm.ai/projects/vllm-omni/en/latest/getting_started/installation/)
and [feature compatibility](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/feature_compatibility/).

- Keep vendor imports and capability probes behind platform guards so portable
  imports and unsupported-device errors remain actionable.
- Verify dtype, scale/granularity, zero point, layout, padding, device, and
  weight-name mapping through the real producer and kernel/model consumer.
- Check capability dispatch and fallback behavior for supported architectures;
  do not silently run a numerically different or much slower path.
- Preserve rank, stream, synchronization, workspace, graph-capture, and lifetime
  contracts across repeated requests and failures.
- Compare shared and vendor worker hooks when siblings do not inherit each
  other; simplified CPU test doubles must not hide MRO or initialization gaps.
- Require output correctness or accuracy evidence with an explicit tolerance.
  Quantitative speed or memory claims also use the performance reference.

Run portable import/static checks first, then the smallest matching-device test
when hardware is available. Name an unavailable GPU/NPU/XPU/ROCm path as a
validation gap; never simulate device evidence.
