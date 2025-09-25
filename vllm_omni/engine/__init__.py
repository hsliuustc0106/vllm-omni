"""
Diffusion Engine: Diffusion Transformer (DiT) processing modules.

This module provides specialized processing for diffusion models including
step management, cache management, and model wrapping.
"""

from .step_manager import DiffusionStepManager
from .cache_manager import DiffusionCacheManager
from .models import DiffusionModel
from .base import BaseDiffusionEngine



from vllm.v1.engine import EngineCoreOutput as _BaseEngineCoreOutput  # type: ignore
from typing import Optional


class EngineCoreOutput(_BaseEngineCoreOutput):
    """Omni EngineCoreOutput.

    Currently identical to vLLM's EngineCoreOutput, which already includes
    fields such as `output_type`. This subclass exists to enable Omni-side
    overrides/extensions without affecting upstream vLLM.
    """

    output_type: Optional[str] = None

__all__ = [
    "DiffusionStepManager",
    "DiffusionCacheManager",
    "DiffusionModel",
    "BaseDiffusionEngine",
    "EngineCoreOutput",
]
