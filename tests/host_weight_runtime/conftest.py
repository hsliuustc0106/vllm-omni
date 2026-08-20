# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib

import pytest

from vllm_omni.host_weight_runtime import (
    ArtifactSpec,
    ArtifactTopologyDescriptor,
    ProducerDescriptor,
    TopologyCoordinate,
    WeightFormatDescriptor,
    derive_weight_format_plan_digest,
)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class ArtifactRegistrar:
    """Small exact-owner used by core contract tests."""

    def __init__(self) -> None:
        self.artifact = None

    def adopt_artifact(self, artifact) -> None:
        if self.artifact is not None:
            raise AssertionError("test artifact registrar already owns an artifact")
        self.artifact = artifact


def make_weight_format(
    *,
    format_id: str = "test-dense",
    adapter_abi: str = "test-dense/v1",
    semantic_fingerprint: str | None = None,
    format_recipe_schema_version: int = 1,
    target_module_type_id: str = "test.transformer",
    normalized_config=None,
    kernel_identity=None,
) -> WeightFormatDescriptor:
    semantic_fingerprint = semantic_fingerprint or digest("format-semantics")
    normalized_config = normalized_config or {"dtype": "float32"}
    kernel_identity = kernel_identity or {"kernel": "copy"}
    plan_digest = derive_weight_format_plan_digest(
        format_id=format_id,
        adapter_abi=adapter_abi,
        semantic_fingerprint=semantic_fingerprint,
        format_recipe_schema_version=format_recipe_schema_version,
        target_module_type_id=target_module_type_id,
        normalized_config=normalized_config,
        kernel_identity=kernel_identity,
    )
    return WeightFormatDescriptor(
        format_id=format_id,
        adapter_abi=adapter_abi,
        semantic_fingerprint=semantic_fingerprint,
        format_plan_digest=plan_digest,
        format_recipe_schema_version=format_recipe_schema_version,
        target_module_type_id=target_module_type_id,
        normalized_config=normalized_config,
        kernel_identity=kernel_identity,
    )


@pytest.fixture
def producer_descriptor() -> ProducerDescriptor:
    return ProducerDescriptor(
        producer_id="test-finalizer",
        producer_abi="test-finalizer/v1",
        semantic_fingerprint=digest("producer-semantics"),
    )


@pytest.fixture
def weight_format() -> WeightFormatDescriptor:
    return make_weight_format()


@pytest.fixture
def artifact_spec(producer_descriptor, weight_format) -> ArtifactSpec:
    return ArtifactSpec(
        source_fingerprint=digest("checkpoint"),
        producer=producer_descriptor,
        weight_format=weight_format,
        topology=ArtifactTopologyDescriptor(
            (
                TopologyCoordinate("pp", 1, 0),
                TopologyCoordinate("tp", 1, 0),
            )
        ),
        layout_abi="test-layout/v1",
    )
