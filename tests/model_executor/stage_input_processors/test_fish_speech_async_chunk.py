# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.fish_speech import (
    slow_ar_to_dac_decoder_async_chunk,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _req(rid, *, finished, initial_codec_chunk_frames=None):
    ai = None
    if initial_codec_chunk_frames is not None:
        entry = SimpleNamespace(list_data=[initial_codec_chunk_frames])
        ai = SimpleNamespace(entries={"initial_codec_chunk_frames": entry})
    return SimpleNamespace(
        external_req_id=rid,
        is_finished=lambda: finished,
        additional_information=ai,
    )


def _tm(*, chunk_frames=25, left_context=25, initial_chunk_frames=0):
    return SimpleNamespace(
        code_prompt_token_ids=defaultdict(list),
        put_req_chunk=defaultdict(int),
        connector=SimpleNamespace(
            config={
                "extra": {
                    "codec_chunk_frames": chunk_frames,
                    "codec_left_context_frames": left_context,
                    "initial_codec_chunk_frames": initial_chunk_frames,
                }
            }
        ),
    )


def _call(tm, rid, *, n_frames, finished=False, req_ic=None):
    frame = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    payload = None
    for _ in range(n_frames):
        payload = slow_ar_to_dac_decoder_async_chunk(
            transfer_manager=tm,
            pooling_output={"audio_codes": frame},
            request=_req(rid, finished=finished, initial_codec_chunk_frames=req_ic),
            is_finished=finished,
        )
    return payload


@pytest.mark.parametrize(
    "n_frames,finished,expected",
    [
        (9, False, None),
        (10, False, (0, 10)),
        (25, False, None),
        (45, False, (20, 45)),
        (33, True, (20, 33)),
    ],
)
def test_initial_phase_transitions_match_qwen3_tts(n_frames, finished, expected):
    tm = _tm(chunk_frames=25, left_context=25, initial_chunk_frames=10)
    payload = _call(tm, "r", n_frames=n_frames, finished=finished)

    if expected is None:
        assert payload is None
    else:
        exp_ctx, exp_window = expected
        assert payload is not None
        assert payload["left_context_size"] == exp_ctx
        assert len(payload["code_predictor_codes"]) == 4 * exp_window


def test_finished_empty_emits_eof_sentinel():
    tm = _tm()
    payload = slow_ar_to_dac_decoder_async_chunk(
        transfer_manager=tm,
        pooling_output=None,
        request=_req("r", finished=True),
        is_finished=True,
    )

    assert payload == {"code_predictor_codes": [], "finished": True}
