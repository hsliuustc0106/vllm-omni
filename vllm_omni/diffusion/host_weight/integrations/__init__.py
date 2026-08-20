# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Built-in diffusion Host Weight Runtime model integrations."""

from __future__ import annotations

from ..model_integration import ModelIntegrationRegistry


def create_builtin_model_integration_registry() -> ModelIntegrationRegistry:
    """Compose built-ins without exposing concrete choices to generic callers."""

    from .minimax_h3_fp8 import MINIMAX_H3_FP8_INTEGRATION

    return ModelIntegrationRegistry((MINIMAX_H3_FP8_INTEGRATION,))


__all__ = ["create_builtin_model_integration_registry"]
