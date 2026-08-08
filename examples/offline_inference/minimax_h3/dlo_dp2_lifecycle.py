# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate MiniMax-H3 offload placement, repeatability, and lifecycle.

Examples:

    # One GPU: offload only the encoder and repeat the same request twice.
    CUDA_VISIBLE_DEVICES=0 python examples/offline_inference/minimax_h3/dlo_dp2_lifecycle.py \
        --model /path/to/MiniMax-H3/FL2VA --mode layerwise-single \
        --components text_encoder --runs 2

    # Two GPUs: host-sharded DP2 DLO with synchronized request waves.
    CUDA_VISIBLE_DEVICES=0,1 python examples/offline_inference/minimax_h3/dlo_dp2_lifecycle.py \
        --model /path/to/MiniMax-H3/FL2VA --mode dlo-dp2 --runs 2

    # Two GPUs: TP2 rank-local DLO, compatible with online FP8.
    CUDA_VISIBLE_DEVICES=0,1 python examples/offline_inference/minimax_h3/dlo_dp2_lifecycle.py \
        --model /path/to/MiniMax-H3/FL2VA --mode dlo-tp2 \
        --components dit,text_encoder,vae --quantization fp8 --runs 2

Set VLLM_WORKER_MULTIPROC_METHOD=spawn and
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=14400 for long production-shape runs. DP2
AllGather validation also uses VLLM_OMNI_DLO_DP_WAVE_TIMEOUT=600.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import multiprocessing
import time
from pathlib import Path
from typing import Any

import numpy as np

from vllm_omni.entrypoints.async_omni import AsyncOmni

DEFAULT_PROMPTS = (
    "At night, three cats march into a bedroom playing tiny brass instruments, "
    "then abruptly file out, with synchronized room ambience.",
    "A paper boat crosses a rain-filled street while distant traffic and water sounds remain synchronized.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to MiniMax-H3/FL2VA")
    parser.add_argument(
        "--mode",
        choices=("resident-single", "layerwise-single", "dlo-dp2", "dlo-tp2", "request"),
        required=True,
        help=("Single-GPU resident/layerwise, DP2 AllGather DLO, TP2 rank-local DLO, or resident TP2"),
    )
    parser.add_argument(
        "--components",
        help="Comma-separated dit,text_encoder,vae selection; omitted means all",
    )
    parser.add_argument("--quantization", choices=("fp8",))
    parser.add_argument("--resident-layers", type=int, default=0)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--batch-wait-ms", type=float, default=500.0)
    parser.add_argument("--init-timeout", type=float, default=1800.0)
    parser.add_argument(
        "--profiler-dir",
        type=Path,
        help="Profile only the final run with torch.profiler and write artifacts here",
    )
    parser.add_argument(
        "--profile-memory",
        action="store_true",
        help="Also capture the expensive torch profiler memory timeline and snapshot",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def engine_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    common: dict[str, Any] = {
        "model": args.model,
        "trust_remote_code": True,
        "usp": 1,
        "ring": 1,
        "vae_parallel_mode": "tile",
        "vae_use_tiling": True,
        "diffusion_attention_backend": "CUDNN_ATTN",
        "request_batch_max_wait_ms": args.batch_wait_ms,
        "enforce_eager": True,
        "stage_init_timeout": args.init_timeout,
        "init_timeout": args.init_timeout,
    }
    if args.mode == "resident-single":
        common.update(
            num_gpus=1,
            tensor_parallel_size=1,
            data_parallel_size=1,
            text_encoder_tp_size=1,
            vae_patch_parallel_size=1,
        )
    elif args.mode == "layerwise-single":
        common.update(
            num_gpus=1,
            tensor_parallel_size=1,
            data_parallel_size=1,
            text_encoder_tp_size=1,
            vae_patch_parallel_size=1,
            enable_layerwise_offload=True,
        )
    elif args.mode == "dlo-dp2":
        if args.resident_layers:
            raise ValueError("DLO+AllGather does not support --resident-layers")
        common.update(
            num_gpus=2,
            tensor_parallel_size=1,
            data_parallel_size=2,
            text_encoder_tp_size=1,
            vae_patch_parallel_size=1,
            enable_distributed_layerwise_offload=True,
            dlo_use_allgather=True,
            dlo_resident_layers=0,
        )
    elif args.mode == "dlo-tp2":
        common.update(
            num_gpus=2,
            tensor_parallel_size=2,
            data_parallel_size=1,
            text_encoder_tp_size=2,
            vae_patch_parallel_size=2,
            enable_distributed_layerwise_offload=True,
            dlo_use_allgather=False,
            dlo_resident_layers=args.resident_layers,
        )
    else:
        common.update(
            num_gpus=2,
            tensor_parallel_size=2,
            data_parallel_size=1,
            text_encoder_tp_size=2,
            vae_patch_parallel_size=2,
            enable_distributed_layerwise_offload=False,
        )

    if args.components is not None:
        if args.mode in ("resident-single", "request"):
            raise ValueError("--components requires a layerwise or DLO mode")
        common["layerwise_offload_components"] = args.components
    if args.quantization is not None:
        if args.mode == "dlo-dp2":
            raise ValueError("Online FP8 requires resident, ordinary layerwise, or DLO without AllGather")
        common["quantization"] = args.quantization
    if args.profiler_dir is not None:
        common["profiler_config"] = {
            "profiler": "torch",
            "torch_profiler_dir": str(args.profiler_dir),
            "torch_profiler_record_shapes": False,
            "torch_profiler_with_stack": False,
            "torch_profiler_with_memory": args.profile_memory,
            "torch_profiler_dump_cuda_time_total": True,
        }
    return common


def sampling_params(
    engine: AsyncOmni,
    args: argparse.Namespace,
    seed: int,
) -> list[Any]:
    params = copy.deepcopy(engine.default_sampling_params_list)
    diffusion = params[0]
    diffusion.width = args.width
    diffusion.height = args.height
    diffusion.fps = 24
    diffusion.num_inference_steps = args.steps
    diffusion.seed = seed
    diffusion.extra_args = {
        "task": "t2va",
        "duration": args.duration,
        "aspect_ratio": "16:9",
        "flow_shift": 12.0,
        "audio_flow_shift": 3.0,
    }
    return params


async def generate_one(
    engine: AsyncOmni,
    args: argparse.Namespace,
    *,
    request_id: str,
    prompt: str,
    seed: int,
) -> Any:
    final_output = None
    async for output in engine.generate(
        prompt=prompt,
        request_id=request_id,
        sampling_params_list=sampling_params(engine, args, seed),
    ):
        if output.finished:
            final_output = output
    if final_output is None:
        raise RuntimeError(f"{request_id} finished without an output")
    return final_output


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def output_summary(output: Any, args: argparse.Namespace) -> dict[str, Any]:
    if not output.images:
        raise RuntimeError(f"{output.request_id} returned no video")
    frames = np.asarray(output.images[0])
    multimodal = output.multimodal_output or {}
    audio = np.asarray(multimodal.get("audio"))
    if frames.ndim != 4 or tuple(frames.shape[1:]) != (
        args.height,
        args.width,
        3,
    ):
        raise RuntimeError(f"{output.request_id} returned invalid video shape {tuple(frames.shape)}")
    if audio.ndim != 3 or tuple(audio.shape[:2]) != (1, 2):
        raise RuntimeError(f"{output.request_id} returned invalid audio shape {tuple(audio.shape)}")
    if int(multimodal.get("fps", 0)) != 24 or int(multimodal.get("audio_sample_rate", 0)) != 32000:
        raise RuntimeError(
            f"{output.request_id} returned invalid media rates: "
            f"fps={multimodal.get('fps')}, audio_sample_rate={multimodal.get('audio_sample_rate')}"
        )
    if args.duration == 5.0 and args.height == 768 and args.width == 1344:
        if tuple(frames.shape) != (124, 768, 1344, 3) or tuple(audio.shape) != (1, 2, 165600):
            raise RuntimeError(
                f"{output.request_id} default shape mismatch: video={tuple(frames.shape)}, audio={tuple(audio.shape)}"
            )
    return {
        "request_id": output.request_id,
        "frames_shape": list(frames.shape),
        "audio_shape": list(audio.shape),
        "frames_sha256": _array_sha256(frames),
        "audio_sha256": _array_sha256(audio),
        "peak_memory_mb": output.peak_memory_mb,
        "stage_durations": output.stage_durations,
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    resolved_engine_kwargs = engine_kwargs(args)
    started = time.perf_counter()
    engine = AsyncOmni(**resolved_engine_kwargs)
    engine_init_s = time.perf_counter() - started
    selected_components = args.components
    if selected_components is None:
        selected_components = "none" if args.mode in ("resident-single", "request") else "dit,text_encoder,vae"
    summary: dict[str, Any] = {
        "mode": args.mode,
        "components": selected_components,
        "engine_kwargs": resolved_engine_kwargs,
        "engine_init_s": engine_init_s,
    }
    try:
        if args.mode == "dlo-dp2":
            started = time.perf_counter()
            asymmetric = await asyncio.gather(
                generate_one(
                    engine,
                    args,
                    request_id="invalid-empty",
                    prompt="",
                    seed=1001,
                ),
                generate_one(
                    engine,
                    args,
                    request_id="invalid-peer",
                    prompt=DEFAULT_PROMPTS[0],
                    seed=1002,
                ),
                return_exceptions=True,
            )
            summary["asymmetric_wave_s"] = time.perf_counter() - started
            summary["asymmetric_errors"] = [
                f"{type(result).__name__}: {result}" for result in asymmetric if isinstance(result, BaseException)
            ]
            if len(summary["asymmetric_errors"]) != 2:
                raise RuntimeError("Both requests in the asymmetric DP wave must fail before dispatch")

        request_count = 2 if args.mode == "dlo-dp2" else 1
        runs = []
        for run_index in range(args.runs):
            profile_this_run = args.profiler_dir is not None and run_index == args.runs - 1
            if profile_this_run:
                await engine.start_profile(profile_prefix="minimax_h3_encoder")
            started = time.perf_counter()
            try:
                outputs = await asyncio.gather(
                    *(
                        generate_one(
                            engine,
                            args,
                            request_id=f"run-{run_index + 1}-{index}",
                            prompt=DEFAULT_PROMPTS[index],
                            seed=2000 + index,
                        )
                        for index in range(request_count)
                    )
                )
            finally:
                if profile_this_run:
                    summary["profiler_results"] = await engine.stop_profile()
            runs.append(
                {
                    "run": run_index + 1,
                    "wall_time_s": time.perf_counter() - started,
                    "outputs": [output_summary(output, args) for output in outputs],
                }
            )
        summary["runs"] = runs
        summary["outputs"] = runs[-1]["outputs"]
        summary["valid_wave_s"] = runs[-1]["wall_time_s"]
        if len(runs) > 1:
            for summary_name, hash_name in (
                ("video_output_deterministic", "frames_sha256"),
                ("audio_output_deterministic", "audio_sha256"),
            ):
                summary[summary_name] = all(
                    len({run["outputs"][index][hash_name] for run in runs}) == 1 for index in range(request_count)
                )
            summary["steady_output_deterministic"] = (
                summary["video_output_deterministic"] and summary["audio_output_deterministic"]
            )
        else:
            summary["video_output_deterministic"] = None
            summary["audio_output_deterministic"] = None
            summary["steady_output_deterministic"] = None
    finally:
        started = time.perf_counter()
        engine.close()
        summary["shutdown_s"] = time.perf_counter() - started

    children = multiprocessing.active_children()
    summary["active_children"] = [{"name": child.name, "pid": child.pid} for child in children]
    if children:
        raise RuntimeError(f"Worker processes remain after shutdown: {children}")
    return summary


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.runs < 1:
        raise ValueError("--steps and --runs must be at least 1")
    if args.resident_layers < 0:
        raise ValueError("--resident-layers must be non-negative")
    summary = asyncio.run(run(args))
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(f"E2E_RESULT {rendered}", flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
