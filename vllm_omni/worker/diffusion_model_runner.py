# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional, Union, List

import torch

from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu_model_runner import (
    GPUModelRunner,
    EMPTY_MODEL_RUNNER_OUTPUT,
    IntermediateTensors,
    MultiModalKwargs,
    get_pp_group,
    has_kv_transfer_group,
    set_forward_context,
)

from vllm.v1.core.sched.output import SchedulerOutput

class DiffusionGPUModelRunner(GPUModelRunner):
    """Diffusion model runner for vLLM-omni (non-autoregressive).

    - Reuses GPUModelRunner preparation, multimodal handling, and TP/PP/DP glue.
    - Does not compute logits or perform token sampling.
    - Executes diffusion process and returns tensors via `pooler_output`.
    """

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[ModelRunnerOutput, IntermediateTensors]:
        self._update_states(scheduler_output)

        if not scheduler_output.total_num_scheduled_tokens:
            if not has_kv_transfer_group():
                return EMPTY_MODEL_RUNNER_OUTPUT
            return self.kv_connector_no_forward(scheduler_output,
                                                self.vllm_config)

        # Prepare decoder inputs and attention metadata (for batch/order mapping)
        (attn_metadata, attention_cuda_graphs, logits_indices,
         spec_decode_metadata, num_scheduled_tokens_np,
         spec_decode_common_attn_metadata) = self._prepare_inputs(
             scheduler_output)

        # Input token count for this iteration (not used by diffusion, but
        # retained to keep DP padding/ordering consistent)
        num_input_tokens = scheduler_output.total_num_scheduled_tokens
        num_pad, num_tokens_across_dp = self.get_dp_padding(num_input_tokens)
        num_input_tokens += num_pad

        # Multimodal conditioning (e.g., text/audio/video encoders)
        if self.is_multimodal_model:
            self._execute_mm_encoder(scheduler_output)
            mm_embeds = self._gather_mm_embeddings(scheduler_output)
        else:
            mm_embeds = []

        # Build inputs to mirror AR runner: input_ids/positions/embeds
        if self.is_multimodal_model and get_pp_group().is_first_rank:
            inputs_embeds_scheduled = self.model.get_input_embeddings(
                input_ids=self.input_ids[:scheduler_output.total_num_scheduled_tokens],
                multimodal_embeddings=mm_embeds or None,
            )
            self.inputs_embeds[:scheduler_output.total_num_scheduled_tokens].copy_(
                inputs_embeds_scheduled)
            input_ids = self.input_ids[:num_input_tokens]
            inputs_embeds = self.inputs_embeds[:num_input_tokens]
            model_mm_kwargs = self._extract_mm_kwargs(scheduler_output)
        else:
            input_ids = self.input_ids[:num_input_tokens]
            inputs_embeds = None
            model_mm_kwargs = {}

        # Positions (mrope or standard)
        if self.uses_mrope:
            positions = self.mrope_positions[:, :num_input_tokens]
        else:
            positions = self.positions[:num_input_tokens]

        # Intermediate tensors sync for PP (if any)
        if get_pp_group().is_first_rank:
            intermediate_tensors = None
        else:
            intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                num_input_tokens, intermediate_tensors, True)

        # Set forward context mainly for resource management and kv connector
        skip_cuda_graphs = True  # diffusion path does not rely on cuda graphs here
        with set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_input_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                skip_cuda_graphs=skip_cuda_graphs,
        ), self.maybe_get_kv_connector_output(
                scheduler_output) as kv_connector_output:

            if not get_pp_group().is_last_rank:
                # For non-last PP stages, pass through intermediate tensors.
                assert intermediate_tensors is not None
                intermediate_tensors.kv_connector_output = kv_connector_output
                return intermediate_tensors

            outputs = self._run_diffusion(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                multimodal_kwargs=model_mm_kwargs,
                logits_indices=logits_indices,
            )

        # Ensure one tensor per request, map to CPU for output struct
        pooler_output: List[Optional[torch.Tensor]] = []
        if isinstance(outputs, torch.Tensor):
            # If model returned a single stacked tensor, split by requests
            assert outputs.shape[0] == self.input_batch.num_reqs
            for i in range(self.input_batch.num_reqs):
                pooler_output.append(outputs[i].detach().cpu())
        elif isinstance(outputs, list):
            for out in outputs:
                pooler_output.append(out.detach().cpu() if out is not None else None)
        else:
            raise RuntimeError("Unsupported diffusion output type")

        self.eplb_step()

        return ModelRunnerOutput(
            req_ids=self.input_batch.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=[],
            spec_token_ids=None,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=pooler_output,
            kv_connector_output=kv_connector_output,
            num_nans_in_logits={},
            multimodal_outputs={},
        )

    def _run_diffusion(
        self,
        *,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor],
        multimodal_kwargs: dict,
        logits_indices: torch.Tensor,
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        """Runs the diffusion process and returns per-request tensors.

        Tries model interfaces in the following order for maximal compatibility:
        1) model.sample(condition=..., **kwargs)
        2) model.forward(condition=..., **kwargs)
        3) model.diffuse(condition=..., **kwargs)
        """
        # Keep inputs identical to AR runner
        kwargs = dict(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **MultiModalKwargs.as_kwargs(multimodal_kwargs, device=self.device),
            sampling_metadata=self.input_batch.sampling_metadata,
            logits_index=logits_indices,
            sampler=self.sampler,
        )

        # For Qwen 2.5 Omni's current implementation, we only support the forward method
        if hasattr(self.model, "forward"):
            return self.model.forward(**kwargs)
        
        # if hasattr(self.model, "sample"):
        #     return self.model.sample(**kwargs)
        # if hasattr(self.model, "forward"):
        #     return self.model.forward(**kwargs)
        # if hasattr(self.model, "diffuse"):
        #     return self.model.diffuse(**kwargs)

        raise RuntimeError(
            "The loaded model does not expose diffusion interfaces 'sample', "
            "'forward', or 'diffuse'. Please implement one of them or adapt the runner.")


