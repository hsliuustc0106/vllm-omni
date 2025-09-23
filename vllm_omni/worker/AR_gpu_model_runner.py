"""AR GPU Model Runner for vLLM-omni.

Exposes per-request hidden representations via ModelRunnerOutput.pooler_output
and also outputs sampled tokens.
"""

from __future__ import annotations

from typing import Optional, Union

import torch

from vllm import envs
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu_model_runner import (
    GPUModelRunner,
    EMPTY_MODEL_RUNNER_OUTPUT,
    IntermediateTensors,
    MultiModalKwargs,
    get_pp_group,
    get_tp_group,
    has_kv_transfer_group,
    set_forward_context,
)
from vllm.v1.core.sched.output import SchedulerOutput


class ARGPUModelRunner(GPUModelRunner):
    """Autoregressive GPU model runner that returns hidden states per request.

    This runner follows the same preparation and forward path as GPUModelRunner
    (inputs assembly, multi-modal handling, TP/PP/DP integration, CUDA graphs),
    and additionally performs lightweight sampling so that sampled tokens are
    available in outputs. Hidden representations are taken at the same indices
    that GPUModelRunner would use for sampling/logits (i.e. `logits_indices`).
    """

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[ModelRunnerOutput, IntermediateTensors]:
        # Update internal state with the new schedule
        self._update_states(scheduler_output)

        # If there's no work to do, either return empty output or kv-only path
        if not scheduler_output.total_num_scheduled_tokens:
            if not has_kv_transfer_group():
                return EMPTY_MODEL_RUNNER_OUTPUT
            return self.kv_connector_no_forward(scheduler_output,
                                                self.vllm_config)

        # Prepare decoder inputs and attention metadata
        (attn_metadata, attention_cuda_graphs, logits_indices,
         spec_decode_metadata, num_scheduled_tokens_np,
         spec_decode_common_attn_metadata) = self._prepare_inputs(
             scheduler_output)

        # Determine number of input tokens for this iteration
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        if (self.use_cuda_graph
                and num_scheduled_tokens <= self.cudagraph_batch_sizes[-1]):
            num_input_tokens = self.vllm_config.pad_for_cudagraph(
                num_scheduled_tokens)
        else:
            tp_size = self.vllm_config.parallel_config.tensor_parallel_size
            if (self.compilation_config.pass_config.enable_sequence_parallelism
                    and tp_size > 1):
                from vllm.compilation.utils import round_up  # lazy local import
                num_input_tokens = round_up(num_scheduled_tokens, tp_size)
            else:
                num_input_tokens = num_scheduled_tokens

        # DP padding
        num_pad, num_tokens_across_dp = self.get_dp_padding(num_input_tokens)
        num_input_tokens += num_pad

        # Multimodal handling (encode and gather embeddings if needed)
        if self.is_multimodal_model:
            self._execute_mm_encoder(scheduler_output)
            mm_embeds = self._gather_mm_embeddings(scheduler_output)
        else:
            mm_embeds = []

        # Build inputs: embeddings for multimodal-first PP rank; ids for text-only
        if self.is_multimodal_model and get_pp_group().is_first_rank:
            inputs_embeds_scheduled = self.model.get_input_embeddings(
                input_ids=self.input_ids[:num_scheduled_tokens],
                multimodal_embeddings=mm_embeds or None,
            )
            # Copy into persistent buffer to enable CUDA Graph capture
            self.inputs_embeds[:num_scheduled_tokens].copy_(
                inputs_embeds_scheduled)
            input_ids = self.input_ids[:num_input_tokens]  # preserved for APIs
            inputs_embeds = self.inputs_embeds[:num_input_tokens]
            model_mm_kwargs = self._extract_mm_kwargs(scheduler_output)
        else:
            input_ids = self.input_ids[:num_input_tokens]
            inputs_embeds = None
            model_mm_kwargs = {}

        # Positions/mRoPE
        if self.uses_mrope:
            positions = self.mrope_positions[:, :num_input_tokens]
        else:
            positions = self.positions[:num_input_tokens]

        # Handle pipeline-parallel intermediate tensors
        if get_pp_group().is_first_rank:
            intermediate_tensors = None
        else:
            intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                num_input_tokens, intermediate_tensors, True)

        # Some attention backends only support CUDA Graphs in pure decode.
        skip_cuda_graphs = self.full_cuda_graph and not attention_cuda_graphs

        # Forward pass
        with set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_input_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                skip_cuda_graphs=skip_cuda_graphs,
        ), self.maybe_get_kv_connector_output(
                scheduler_output) as kv_connector_output:

            model_output = self.model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **MultiModalKwargs.as_kwargs(
                    model_mm_kwargs,
                    device=self.device,
                ),
                sampling_metadata=self.input_batch.sampling_metadata,
                logits_index=logits_indices,
                sampler=self.sampler,
            )

        if self.use_aux_hidden_state_outputs:
            hidden_states, _aux_hidden_states = model_output
        else:
            hidden_states = model_output

        text_hidden_states, multimodal_outputs = (
            self.extract_multimodal_outputs(hidden_states))

        # Mid PP stages return intermediate tensors unmodified
        if not get_pp_group().is_last_rank:
            assert isinstance(text_hidden_states, IntermediateTensors)
            text_hidden_states.kv_connector_output = kv_connector_output
            return text_hidden_states

        # Broadcast PP output for external_launcher (torchrun)
        broadcast_pp_output = \
            self.parallel_config.distributed_executor_backend \
            == "external_launcher" and len(get_pp_group().ranks) > 0
        if not get_pp_group().is_last_rank:
            assert isinstance(text_hidden_states, IntermediateTensors)
            if not broadcast_pp_output:
                text_hidden_states.kv_connector_output = kv_connector_output
                return text_hidden_states
            get_pp_group().send_tensor_dict(text_hidden_states.tensors,
                                            all_gather_group=get_tp_group())
            logits = None
        else:
            if self.input_batch.pooling_params:
                return self._pool(text_hidden_states, num_scheduled_tokens,
                                  num_scheduled_tokens_np, kv_connector_output)

            sample_hidden_states = text_hidden_states[logits_indices]
            logits = self.model.compute_logits(sample_hidden_states, None)
        if broadcast_pp_output:
            model_output_broadcast_data = {
                "logits": logits.contiguous(),
            } if logits is not None else {}
            model_output_broadcast_data = get_pp_group().broadcast_tensor_dict(
                model_output_broadcast_data, src=len(get_pp_group().ranks) - 1)
            assert model_output_broadcast_data is not None
            logits = model_output_broadcast_data["logits"]

        # Apply structured output bitmasks if present
        if scheduler_output.grammar_bitmask is not None:
            self.apply_grammar_bitmask(scheduler_output, logits)

        # Sample the next token and get logprobs if needed (with spec decode)
        sampling_metadata = self.input_batch.sampling_metadata
        if spec_decode_metadata is None:
            sampler_output = self.sampler(
                logits=logits,
                sampling_metadata=sampling_metadata,
            )
        else:
            assert logits is not None
            bonus_logits = logits[spec_decode_metadata.bonus_logits_indices]
            sampler_output = self.sampler(
                logits=bonus_logits,
                sampling_metadata=sampling_metadata,
            )
            bonus_token_ids = sampler_output.sampled_token_ids
            target_logits = logits[spec_decode_metadata.target_logits_indices]
            output_token_ids = self.rejection_sampler(
                spec_decode_metadata,
                None,
                target_logits,
                bonus_token_ids,
                sampling_metadata,
            )
            sampler_output.sampled_token_ids = output_token_ids

        num_nans_in_logits = {}
        if envs.VLLM_COMPUTE_NANS_IN_LOGITS:
            num_nans_in_logits = self._get_nans_in_logits(logits)

        # Handle partial prefill: discard sampled tokens and rewind RNG
        discard_sampled_tokens_req_indices = []
        for i, req_id in enumerate(self.input_batch.req_ids):
            req_state = self.requests[req_id]
            seq_len = (req_state.num_computed_tokens +
                       scheduler_output.num_scheduled_tokens[req_id])
            if seq_len < req_state.num_tokens:
                generator = self.input_batch.generators.get(i)
                if generator is not None:
                    generator.set_offset(generator.get_offset() - 4)
                discard_sampled_tokens_req_indices.append(i)

        # Move CPU sync parts
        logprobs_tensors = sampler_output.logprobs_tensors
        logprobs_lists = logprobs_tensors.tolists() \
            if logprobs_tensors is not None else None

        # Prompt logprobs if needed
        prompt_logprobs_dict = self._get_prompt_logprobs_dict(
            text_hidden_states[:num_scheduled_tokens],
            scheduler_output,
        )

        # Parse valid sampled tokens
        sampled_token_ids = sampler_output.sampled_token_ids
        max_gen_len = sampled_token_ids.shape[-1]
        if max_gen_len == 1:
            valid_sampled_token_ids = sampled_token_ids.tolist()
        else:
            valid_sampled_token_ids = self.rejection_sampler.parse_output(
                sampled_token_ids,
                self.input_batch.vocab_size,
            )
        for i in discard_sampled_tokens_req_indices:
            valid_sampled_token_ids[i].clear()

        # Cache sampled tokens
        for req_idx, sampled_ids in enumerate(valid_sampled_token_ids):
            if not sampled_ids:
                continue
            start_idx = self.input_batch.num_tokens_no_spec[req_idx]
            end_idx = start_idx + len(sampled_ids)
            self.input_batch.token_ids_cpu[req_idx, start_idx:end_idx] = sampled_ids
            self.input_batch.num_tokens_no_spec[req_idx] = end_idx
            self.input_batch.num_tokens[req_idx] = end_idx
            req_id = self.input_batch.req_ids[req_idx]
            req_state = self.requests[req_id]
            req_state.output_token_ids.extend(sampled_ids)

        # Speculative decoding draft tokens if configured
        if not self.speculative_config:
            spec_token_ids = None
        else:
            assert spec_decode_common_attn_metadata is not None
            spec_token_ids = self.propose_draft_token_ids(
                scheduler_output,
                valid_sampled_token_ids,
                sampling_metadata,
                text_hidden_states,
                sample_hidden_states,
                _aux_hidden_states if '_aux_hidden_states' in locals() else None,
                spec_decode_metadata,
                spec_decode_common_attn_metadata,
            )

        # Convert to per-request tensors on CPU
        pooler_output: list[Optional[torch.Tensor]] = []
        for i in range(self.input_batch.num_reqs):
            pooler_output.append(sample_hidden_states[i].detach().cpu())


        self.eplb_step()

        return ModelRunnerOutput(
            req_ids=self.input_batch.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=valid_sampled_token_ids,
            spec_token_ids=spec_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict=prompt_logprobs_dict,
            pooler_output=pooler_output,
            kv_connector_output=kv_connector_output,
            num_nans_in_logits=num_nans_in_logits,
            multimodal_outputs=multimodal_outputs,
        )


