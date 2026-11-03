from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
from neuralop.layers.spectral_convolution import SpectralConv

from lean_no.adapters.fno import build_graph
from lean_no.bounds.activations import crown_tanh_rule, is_tanh
from lean_no.bounds.operators import (
    affine_bias,
    affine_weight,
    crown_relu_rule,
    interval_linear,
    is_identity,
    is_relu,
)
from lean_no.function_space.grid.ibp import ibp_layers, prepare_input, resolve_query
from lean_no.function_space.grid.ibp import validate as validate_grid_ibp
from lean_no.kernel import grid_kernel
from lean_no.spectral import real_inverse

@dataclass
class AffineBound:

    coefficient: torch.Tensor
    bias: torch.Tensor

@dataclass
class CrownBounds:

    lower: AffineBound
    upper: AffineBound

def backward_relu(
    bounds: CrownBounds,
    preactivation_lower: torch.Tensor,
    preactivation_upper: torch.Tensor,
) -> CrownBounds:
    lower_slope, upper_slope, upper_bias = crown_relu_rule(
        preactivation_lower,
        preactivation_upper,
    )
    lower_positive = bounds.lower.coefficient.clamp(min=0)
    lower_negative = bounds.lower.coefficient.clamp(max=0)
    upper_positive = bounds.upper.coefficient.clamp(min=0)
    upper_negative = bounds.upper.coefficient.clamp(max=0)
    dimensions = tuple(range(2, bounds.lower.coefficient.ndim))
    return CrownBounds(
        lower=AffineBound(
            lower_positive * lower_slope.unsqueeze(1)
            + lower_negative * upper_slope.unsqueeze(1),
            bounds.lower.bias
            + (lower_negative * upper_bias.unsqueeze(1)).sum(dim=dimensions),
        ),
        upper=AffineBound(
            upper_positive * upper_slope.unsqueeze(1)
            + upper_negative * lower_slope.unsqueeze(1),
            bounds.upper.bias
            + (upper_positive * upper_bias.unsqueeze(1)).sum(dim=dimensions),
        ),
    )

def backward_tanh(bounds, lower, upper):
    low_slope, low_bias, high_slope, high_bias = crown_tanh_rule(lower, upper)

    def apply(bound, upper_side):
        positive, negative = (
            bound.coefficient.clamp(min=0),
            bound.coefficient.clamp(max=0),
        )
        ps, pb, ns, nb = (
            (high_slope, high_bias, low_slope, low_bias)
            if upper_side
            else (low_slope, low_bias, high_slope, high_bias)
        )
        coefficient = positive * ps.unsqueeze(1) + negative * ns.unsqueeze(1)
        offset = positive * pb.unsqueeze(1) + negative * nb.unsqueeze(1)
        return AffineBound(coefficient, bound.bias + offset.flatten(2).sum(-1))

    return CrownBounds(apply(bounds.lower, False), apply(bounds.upper, True))

def backward_affine(
    bound: AffineBound,
    module: nn.Module,
) -> AffineBound:
    if isinstance(module, nn.Identity):
        return bound

    bias = bound.bias
    module_bias = affine_bias(module, bound.coefficient)
    if module_bias is not None:
        bias = bias + torch.einsum("bso...,o->bs", bound.coefficient, module_bias)
    weight = affine_weight(module).detach().to(bound.coefficient)
    coefficient = torch.einsum("bso...,oi->bsi...", bound.coefficient, weight)
    return AffineBound(coefficient, bias)

def backward_spectral(
    bound: AffineBound,
    conv: SpectralConv,
    kernel_frequency: torch.Tensor,
) -> AffineBound:
    bias = bound.bias
    conv_bias = getattr(conv, "bias", None)
    if conv_bias is not None:
        conv_bias = conv_bias.detach().to(bound.coefficient).reshape(-1)
        bias = bias + torch.einsum("bso...,o->bs", bound.coefficient, conv_bias)

    resolution = bound.coefficient.shape[3:]
    batch_size, specification_count, output_channels = bound.coefficient.shape[:3]
    values = bound.coefficient.reshape(
        batch_size * specification_count, output_channels, *resolution
    )
    dimensions = tuple(range(-len(resolution), 0))
    values_frequency = torch.fft.rfftn(values, dim=dimensions)
    output_frequency = torch.einsum(
        "oi...,si...->so...", kernel_frequency, values_frequency
    )
    coefficient = real_inverse(output_frequency, resolution) / math.prod(resolution)
    coefficient = coefficient.reshape(
        batch_size,
        specification_count,
        coefficient.shape[1],
        *resolution,
    )
    return AffineBound(coefficient, bias)

def backward_normalization(bounds, module, lower, upper):
    from lean_no.bounds.normalization import (
        normalization_adjoint,
        normalization_affine_bounds,
        normalization_parameters,
    )

    weight, bias, epsilon, groups = normalization_parameters(
        module, lower.shape[1], lower
    )

    def apply(bound, upper_side):
        coefficients, offsets = [], []
        for index, (lo, hi) in enumerate(zip(lower, upper)):
            scale, shift, error = normalization_affine_bounds(
                lo, hi, weight, bias, epsilon, groups
            )
            coefficient = bound.coefficient[index]
            bias_shape = (1, len(shift), *((1,) * (lo.ndim - 1)))
            spatial = tuple(range(1, coefficient.ndim))
            offset = (coefficient * shift.reshape(bias_shape)).sum(spatial)
            radius = (coefficient.abs() * error).sum(spatial)
            offsets.append(
                bound.bias[index] + offset + (radius if upper_side else -radius)
            )
            coefficients.append(normalization_adjoint(coefficient, scale, groups))
        return AffineBound(torch.stack(coefficients), torch.stack(offsets))

    return CrownBounds(apply(bounds.lower, False), apply(bounds.upper, True))

def backward_graph(bounds, graph, preactivations, kernels):
    pending = {len(graph.ops): bounds}

    def merge(left, right):
        return AffineBound(left.coefficient + right.coefficient, left.bias + right.bias)

    for index in range(len(graph.ops), 0, -1):
        if index not in pending:
            continue
        bounds = pending.pop(index)
        op = graph.ops[index - 1]
        if op.kind == "add":
            zero = CrownBounds(
                AffineBound(
                    bounds.lower.coefficient, torch.zeros_like(bounds.lower.bias)
                ),
                AffineBound(
                    bounds.upper.coefficient, torch.zeros_like(bounds.upper.bias)
                ),
            )
            contributions = (bounds, zero)
        else:
            if op.kind == "activation" and is_relu(op.data["activation"]):
                bounds = backward_relu(bounds, *preactivations[op.name])
            elif op.kind == "activation" and is_tanh(op.data["activation"]):
                bounds = backward_tanh(bounds, *preactivations[op.name])
            elif op.kind == "normalize":
                bounds = backward_normalization(bounds, op, *preactivations[op.name])
            elif op.kind == "affine":
                bounds = CrownBounds(
                    backward_affine(bounds.lower, op), backward_affine(bounds.upper, op)
                )
            elif op.kind == "spectral":
                bounds = CrownBounds(
                    backward_spectral(bounds.lower, op, kernels[op.name]),
                    backward_spectral(bounds.upper, op, kernels[op.name]),
                )
            contributions = (bounds,)
        for parent, value in zip(op.parents, contributions):
            if parent in pending:
                previous = pending[parent]
                value = CrownBounds(
                    merge(previous.lower, value.lower),
                    merge(previous.upper, value.upper),
                )
            pending[parent] = value
    return pending[0]

def concretize(
    bounds: CrownBounds,
    input_lower: torch.Tensor,
    input_upper: torch.Tensor,
    shared_input_axes: tuple[int, ...] = (),
) -> tuple[torch.Tensor, torch.Tensor]:

    if shared_input_axes:
        center = (input_lower + input_upper) / 2
        radius = (input_upper - input_lower) / 2
        axes = tuple(axis + 3 for axis in shared_input_axes)
        if len(set(axes)) != len(axes) or any(
            axis < 3 or axis >= radius.ndim + 1 for axis in axes
        ):
            raise ValueError("shared input axes must be distinct spatial axes")
        reduced_radius = radius.unsqueeze(1)
        for axis in sorted(axes, reverse=True):
            first = reduced_radius.select(axis, 0).unsqueeze(axis)
            if not torch.equal(reduced_radius, first.expand_as(reduced_radius)):
                raise ValueError(
                    "shared perturbations require constant radii along shared axes"
                )
            reduced_radius = first

        def evaluate(bound, sign):
            nominal = (bound.coefficient * center.unsqueeze(1)).flatten(2).sum(-1)
            coefficient = bound.coefficient.sum(dim=axes, keepdim=True)
            uncertainty = (coefficient.abs() * reduced_radius).flatten(2).sum(-1)
            return bound.bias + nominal + sign * uncertainty

        return evaluate(bounds.lower, -1), evaluate(bounds.upper, 1)

    def multiply(coefficient: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        dimensions = tuple(range(2, coefficient.ndim))
        return (coefficient * values.unsqueeze(1)).sum(dim=dimensions)

    lower, _ = interval_linear(
        bounds.lower.coefficient,
        input_lower,
        input_upper,
        multiply,
    )
    _, upper = interval_linear(
        bounds.upper.coefficient,
        input_lower,
        input_upper,
        multiply,
    )
    return bounds.lower.bias + lower, bounds.upper.bias + upper

def initial(
    output: torch.Tensor,
    flat_indices: torch.Tensor,
) -> AffineBound:
    batch_size, output_channels = output.shape[:2]
    resolution = output.shape[2:]
    point_count = math.prod(resolution)
    specification_count = flat_indices.numel()
    coefficient = output.new_zeros(
        batch_size,
        specification_count,
        output_channels,
        *resolution,
    )
    channels = torch.div(flat_indices, point_count, rounding_mode="floor")
    points = flat_indices % point_count
    flat = coefficient.flatten(3)
    flat[
        :,
        torch.arange(specification_count, device=output.device),
        channels,
        points,
    ] = 1
    return AffineBound(
        coefficient,
        output.new_zeros(batch_size, specification_count),
    )

def validate(model: nn.Module, resolution: int | tuple[int, ...]) -> None:
    validate_grid_ibp(model, resolution)
    graph = build_graph(model, normalization=True)
    if any(
        not is_relu(activation)
        and not is_identity(activation)
        and not is_tanh(activation)
        for activation in graph.activations
    ):
        raise NotImplementedError("discrete CROWN supports ReLU, tanh and identity")

@torch.no_grad()
def grid_crown(
    model: nn.Module,
    lower: torch.Tensor,
    upper: torch.Tensor,
    output_chunk_size: int = 64,
    query: int | tuple[int, ...] | torch.Tensor | None = None,
    progress: Callable[[int, int], None] | None = None,
    shared_input_axes: tuple[int, ...] = (),
) -> tuple[torch.Tensor, torch.Tensor]:
    if output_chunk_size <= 0:
        raise ValueError("output_chunk_size must be positive")
    resolution = tuple(lower.shape[2:])
    validate(model, resolution)
    input_lower, input_upper = prepare_input(model, lower, upper)
    graph = build_graph(model, normalization=True)
    preactivations = {}
    output_lower, _ = ibp_layers(graph, input_lower, input_upper, preactivations)

    # The sampled layer kernels are fixed throughout this certificate. Preserve
    # their synthesis, adjoint flips and FFT, and reuse them for both bounds.
    kernel_frequencies = {}
    dimensions = tuple(range(-len(resolution), 0))
    for op in graph.ops:
        if op.kind != "spectral":
            continue
        kernel = grid_kernel(op, resolution).to(input_lower)
        for axis, count in enumerate(resolution):
            index = -torch.arange(count, device=input_lower.device) % count
            kernel = kernel.index_select(axis + 2, index)
        kernel_frequencies[op.name] = torch.fft.rfftn(
            kernel.transpose(0, 1), dim=dimensions
        )
    del kernel

    batch_size, channels = output_lower.shape[:2]
    point_count = math.prod(resolution)
    if query is None:
        flat_indices = torch.arange(
            channels * point_count,
            device=output_lower.device,
        )
        output_shape = output_lower.shape
    else:
        points = resolve_query(query, resolution, output_lower.device)
        if isinstance(points, tuple):
            point_index = torch.zeros_like(points[0])
            for index, size in zip(points, resolution):
                point_index = point_index * size + index
        else:
            point_index = points
        channel = torch.arange(channels, device=output_lower.device)
        channel = channel.reshape(channels, *((1,) * point_index.ndim))
        flat_indices = (channel * point_count + point_index.unsqueeze(0)).reshape(-1)
        output_shape = (batch_size, channels, *point_index.shape)

    lower_chunks = []
    upper_chunks = []
    for start in range(0, flat_indices.numel(), output_chunk_size):
        selected = flat_indices[start : start + output_chunk_size]
        identity = initial(output_lower, selected)
        bounds = CrownBounds(identity, identity)
        bounds = backward_graph(bounds, graph, preactivations, kernel_frequencies)
        chunk_lower, chunk_upper = concretize(
            bounds,
            input_lower,
            input_upper,
            shared_input_axes,
        )
        lower_chunks.append(chunk_lower)
        upper_chunks.append(chunk_upper)
        if progress is not None:
            progress(
                min(start + output_chunk_size, flat_indices.numel()),
                flat_indices.numel(),
            )

    return (
        torch.cat(lower_chunks, dim=1).reshape(output_shape),
        torch.cat(upper_chunks, dim=1).reshape(output_shape),
    )
