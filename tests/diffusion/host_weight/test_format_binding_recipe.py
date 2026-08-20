# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import replace

import pytest

from vllm_omni.diffusion.host_weight.formats import (
    FORMAT_BINDING_RECIPE_SCHEMA_VERSION,
    BindingDestination,
    FormatBindingRecipe,
    FormatContractError,
    FormatTensorRole,
    LayerFormatSpec,
    ModuleStateKind,
    OptionalLayerTensorBinding,
    RequiredLayerTensorBinding,
    TargetModulePath,
)


def _required(
    role: FormatTensorRole,
    attribute: str,
    tensor_id: str,
) -> RequiredLayerTensorBinding:
    return RequiredLayerTensorBinding(
        role=role,
        destination=BindingDestination(
            module_path=TargetModulePath("proj"),
            attribute_name=attribute,
            state_kind=ModuleStateKind.PARAMETER,
        ),
        tensor_id=tensor_id,
    )


def _layer(*, bias_tensor_id: str | None = None) -> LayerFormatSpec:
    return LayerFormatSpec(
        module_path="proj",
        quant_method_id="method/v2",
        kernel_id="kernel/v2",
        tensor_bindings=(
            _required(FormatTensorRole.WEIGHT, "weight", "proj.weight"),
            _required(
                FormatTensorRole.WEIGHT_SCALE,
                "weight_scale",
                "proj.weight_scale",
            ),
            OptionalLayerTensorBinding(
                role=FormatTensorRole.BIAS,
                destination=BindingDestination(
                    module_path=TargetModulePath("proj"),
                    attribute_name="bias",
                    state_kind=ModuleStateKind.PARAMETER,
                ),
                tensor_id=bias_tensor_id,
            ),
        ),
        scalar_state={},
    )


def _recipe(layer: LayerFormatSpec | None = None) -> FormatBindingRecipe:
    return FormatBindingRecipe(
        schema_version=FORMAT_BINDING_RECIPE_SCHEMA_VERSION,
        format_plan_digest="0" * 64,
        target_module_type_id="target/v1",
        layers=(_layer() if layer is None else layer,),
        non_layer_bindings=(),
    )


def test_required_optional_binding_variants_round_trip_canonically() -> None:
    recipe = _recipe()

    metadata = recipe.to_dict()
    bindings = metadata["layers"][0]["tensor_bindings"]
    assert [binding["binding_kind"] for binding in bindings] == [
        "required",
        "required",
        "optional",
    ]
    assert bindings[-1]["tensor_id"] is None
    assert recipe.tensor_ids == ("proj.weight", "proj.weight_scale")
    assert FormatBindingRecipe.from_json(recipe.to_json()) == recipe


def test_optional_present_bias_participates_in_exact_tensor_coverage() -> None:
    recipe = _recipe(_layer(bias_tensor_id="proj.bias"))

    assert recipe.tensor_ids == (
        "proj.weight",
        "proj.weight_scale",
        "proj.bias",
    )


def test_layer_requires_closed_weight_scale_and_bias_schema() -> None:
    layer = _layer()

    with pytest.raises(FormatContractError, match="exactly one bias slot"):
        replace(
            layer,
            tensor_bindings=tuple(
                binding for binding in layer.tensor_bindings if binding.role is not FormatTensorRole.BIAS
            ),
        )

    bindings = list(layer.tensor_bindings)
    bindings[0] = OptionalLayerTensorBinding(
        role=FormatTensorRole.WEIGHT,
        destination=BindingDestination(
            module_path=TargetModulePath("proj"),
            attribute_name="weight",
            state_kind=ModuleStateKind.PARAMETER,
        ),
        tensor_id="proj.weight",
    )
    with pytest.raises(FormatContractError, match="weight.*as required"):
        replace(layer, tensor_bindings=tuple(bindings))


def test_binding_parser_rejects_ambiguous_or_nullable_required_variants() -> None:
    metadata = _recipe().to_dict()
    binding = metadata["layers"][0]["tensor_bindings"][0]
    binding["binding_kind"] = "maybe"
    with pytest.raises(FormatContractError, match="binding_kind"):
        FormatBindingRecipe.from_dict(metadata)

    metadata = _recipe().to_dict()
    binding = metadata["layers"][0]["tensor_bindings"][0]
    binding["tensor_id"] = None
    with pytest.raises(FormatContractError, match="tensor_id must be a string"):
        FormatBindingRecipe.from_dict(metadata)


def test_recipe_schema_bump_rejects_pre_union_metadata() -> None:
    metadata = _recipe().to_dict()
    metadata["schema_version"] = FORMAT_BINDING_RECIPE_SCHEMA_VERSION - 1

    with pytest.raises(FormatContractError, match="unsupported format recipe schema"):
        FormatBindingRecipe.from_dict(metadata)
