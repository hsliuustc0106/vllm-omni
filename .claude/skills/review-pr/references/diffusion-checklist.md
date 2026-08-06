# Diffusion Review Checklist

Read this reference for diffusion pipeline, model, scheduler, latent, cache,
parallelism, offload, quantization, or serving changes.

Official docs: [adding a diffusion model](https://docs.vllm.ai/projects/vllm-omni/en/latest/contributing/model/adding_diffusion_model/),
[diffusion features](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/diffusion_features/),
and [feature compatibility](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/feature_compatibility/).

## Integration closure

- Align the pipeline registry, architecture/model identifier, exports, config,
  loader, processor, optional dependencies, examples, and tests.
- Exercise every advertised offline or online entrypoint through production
  dispatch; do not require an unsupported mode merely for checklist symmetry.
- Keep admission, batching, cancellation, and scheduling in the runtime rather
  than model-specific code.
- Confirm any borrowed diffusers component does not silently take ownership of
  lifecycle, device placement, or scheduling from vLLM-Omni.

## Runtime contracts

- Preserve scheduler ownership from request admission through terminal cleanup.
- Trace latent and conditioning shape, layout, dtype, device, batch/CFG
  expansion, timestep state, and output conversion through the final consumer.
- Verify cache keys include model, request, timestep, precision, device, and
  parallel layout inputs that affect correctness; preserve a correct disabled
  path and cleanup on success, failure, timeout, and cancellation.
- Check sender/receiver symmetry for distributed latents and all supported rank
  layouts. Keep vendor-specific assumptions behind platform guards.
- Make acceleration, offload, tiling, quantization, and parallel features
  opt-in or explicitly defaulted; verify feature-off behavior and supported
  combinations touched by the diff.

## Evidence

Require evidence proportionate to the contribution:

| Change | Minimum useful evidence |
| --- | --- |
| New pipeline/model | Runnable production-path inference, valid sample output with parameters, representative E2E test, and model/usage docs. |
| Runtime or lifecycle change | Focused scheduler/executor test plus failure, cancellation, and cleanup coverage for the changed state. |
| Parallel/cache/optimization feature | Feature off/on correctness, affected topology, output-quality comparison, and exact command/environment. |
| Performance or memory claim | Comparable base/head latency and peak memory plus quality evidence; use [perf-verification.md](perf-verification.md). |

Do not impose universal acceleration or memory-feature requirements on a model
addition. Ask for comparison with diffusers or another known-good implementation
when it exists and materially supports correctness or a claim.

## Documentation and tests

Check the supported-model and feature tables actually used by the reviewed
branch, a minimal runnable example, constraints, and known limitations. Ensure
tests assert output modality/shape/content rather than only “did not crash,” and
mark hardware/model tests for the correct CI lane.
