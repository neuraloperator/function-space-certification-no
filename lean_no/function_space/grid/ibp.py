from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from lean_no.adapters.fno import build_graph, model_real_dtype
from lean_no.bounds.operators import (
    affine_bias,
    affine_weight,
    convolution,
    gelu_bounds,
    interval_linear,
    is_gelu,
    is_identity,
    is_relu,
    is_tanh,
)
from lean_no.graph import execute
from lean_no.kernel import grid_kernel

def resolve_query(
    query: int | tuple[int, ...] | torch.Tensor,
    resolution: int | tuple[int, ...],
    device: torch.device,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    if isinstance(resolution, int):
        resolution = (resolution,)
    points = torch.as_tensor(query, device=device)
    if points.dtype == torch.bool or points.dtype.is_floating_point:
        raise TypeError("grid query must contain integer indices")
    if points.numel() == 0:
        raise ValueError("grid query must not be empty")
    if len(resolution) == 1:
        if torch.any(points < 0) or torch.any(points >= resolution[0]):
            raise ValueError("grid query indices are out of range")
        return points.long()
    if points.ndim == 0 or points.shape[0] != len(resolution):
        raise ValueError("grid query must have one index per spatial axis")
    indices = tuple(points[axis] for axis in range(len(resolution)))
    if any(
        torch.any(index < 0) or torch.any(index >= size)
        for index, size in zip(indices, resolution)
    ):
        raise ValueError("grid query indices are out of range")
    return tuple(index.long() for index in indices)

@torch.no_grad()
def grid_ibp_conv(
    conv: nn.Module,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = conv.weight
    real_dtype = weight.real.dtype if weight.is_complex() else weight.dtype
    device = weight.device
    lower = lower.detach().to(device=device, dtype=real_dtype)
    upper = upper.detach().to(device=device, dtype=real_dtype)
    dimension = lower.ndim - 2
    kernel_values = grid_kernel(conv, tuple(lower.shape[2:])).to(lower)
    output_lower, output_upper = interval_linear(
        kernel_values,
        lower,
        upper,
        convolution,
    )

    bias = getattr(conv, "bias", None)
    if bias is not None:
        bias = bias.detach().to(device=device, dtype=real_dtype)
        bias = bias.reshape(1, -1, *((1,) * dimension))
        output_lower = output_lower + bias
        output_upper = output_upper + bias

    return output_lower, output_upper

def ibp_affine(
    module: nn.Module,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(module, nn.Identity):
        return lower, upper

    weight = affine_weight(module).detach().to(lower)
    output_lower, output_upper = interval_linear(
        weight,
        lower,
        upper,
        lambda matrix, values: torch.einsum("oi,bi...->bo...", matrix, values),
    )

    bias = affine_bias(module, lower)
    if bias is not None:
        bias = bias.reshape(1, -1, *((1,) * (lower.ndim - 2)))
        output_lower = output_lower + bias
        output_upper = output_upper + bias
    return output_lower, output_upper

def ibp_activation(
    activation: Callable,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if is_identity(activation):
        return lower, upper
    if is_relu(activation):
        return torch.relu(lower), torch.relu(upper)
    if is_tanh(activation):
        return torch.tanh(lower), torch.tanh(upper)
    if is_gelu(activation):
        if isinstance(activation, nn.GELU) and activation.approximate != "none":
            raise NotImplementedError("only exact GELU interval bounds are supported")
        return gelu_bounds(lower, upper)
    raise NotImplementedError(
        f"unsupported activation for FNO IBP: {type(activation).__name__}"
    )

def ibp_normalization(module, lower, upper, preactivations=None):
    from lean_no.bounds.normalization import (
        normalization_affine_bounds,
        normalization_parameters,
    )

    weight, bias, epsilon, groups = normalization_parameters(
        module, lower.shape[1], lower
    )
    lows, highs = [], []
    for lo, hi in zip(lower, upper):
        scale, offset, error = normalization_affine_bounds(
            lo, hi, weight, bias, epsilon, groups
        )
        count = lo.numel() // groups
        # Sign-split the exact group-centering matrix without allocating it.
        grouped_lo, grouped_hi = lo.reshape(groups, -1), hi.reshape(groups, -1)
        centered_lo = (
            grouped_lo
            - (grouped_hi.sum(-1, keepdim=True) - grouped_hi + grouped_lo) / count
        )
        centered_hi = (
            grouped_hi
            - (grouped_lo.sum(-1, keepdim=True) - grouped_lo + grouped_hi) / count
        )
        shape = (len(scale), *((1,) * (lo.ndim - 1)))
        positive, negative = (
            scale.clamp_min(0).reshape(shape),
            scale.clamp_max(0).reshape(shape),
        )
        shift = offset.reshape(shape)
        lows.append(
            positive * centered_lo.reshape_as(lo)
            + negative * centered_hi.reshape_as(hi)
            + shift
            - error
        )
        highs.append(
            positive * centered_hi.reshape_as(hi)
            + negative * centered_lo.reshape_as(lo)
            + shift
            + error
        )
    return torch.stack(lows), torch.stack(highs)

def validate(model, resolution):
    graph = build_graph(model, normalization=True)
    resolution = (resolution,) if isinstance(resolution, int) else tuple(resolution)
    if graph.dimension not in (1, 2) or len(resolution) != graph.dimension:
        raise ValueError(
            "grid bounds require matching one- or two-dimensional geometry"
        )
    for op in graph.ops:
        if op.kind == "spectral":
            for axis, (modes, count) in enumerate(zip(op.data["modes"], resolution)):
                # Full FFT axes store the negative Nyquist bin once. The
                # last (real FFT) axis needs special endpoint weights, which
                # the sampled continuum kernel does not currently implement.
                invalid = (modes < -count / 2).any() or (modes >= count / 2).any()
                if axis == len(resolution) - 1:
                    invalid = (modes.abs() >= count / 2).any()
                if invalid:
                    raise ValueError("active modes exceed supported discrete FFT bins")

@torch.no_grad()
def prepare_input(
    model: nn.Module,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    model = build_graph(model, normalization=True)
    if lower.shape != upper.shape:
        raise ValueError("grid lower and upper bounds must have the same shape")
    if lower.ndim != model.dimension + 2 or lower.shape[1] != model.in_channels:
        raise ValueError("grid input shape does not match the FNO")
    if not torch.isfinite(lower).all() or not torch.isfinite(upper).all():
        raise ValueError("grid input bounds must be finite")
    if torch.any(lower > upper):
        raise ValueError("grid lower bounds exceed upper bounds")
    device = model.device
    real_dtype = model_real_dtype(model)
    lower = lower.detach().to(device=device, dtype=real_dtype)
    upper = upper.detach().to(device=device, dtype=real_dtype)
    lower = model.append_coordinates(lower)
    upper = model.append_coordinates(upper)
    return lower, upper

def ibp_layers(graph, lower, upper, preactivations=None):

    def apply(op, parents):
        low, high = parents[0]
        if preactivations is not None and op.kind in ("activation", "normalize"):
            preactivations[op.name] = (low, high)
        if op.kind == "affine":
            return ibp_affine(op, low, high)
        if op.kind == "spectral":
            return grid_ibp_conv(op, low, high)
        if op.kind == "normalize":
            return ibp_normalization(op, low, high)
        if op.kind == "add":
            return tuple(a + b for a, b in zip(*parents))
        if op.kind == "activation":
            return ibp_activation(op.data["activation"], low, high)
        raise NotImplementedError(op.kind)

    return execute(graph, (lower, upper), apply)

@torch.no_grad()
def grid_ibp(
    model: nn.Module,
    lower: torch.Tensor,
    upper: torch.Tensor,
    query: int | tuple[int, ...] | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    resolution = tuple(lower.shape[2:])
    validate(model, resolution)
    lower, upper = prepare_input(model, lower, upper)
    lower, upper = ibp_layers(build_graph(model, normalization=True), lower, upper)
    if query is None:
        return lower, upper
    points = resolve_query(query, resolution, lower.device)
    if isinstance(points, tuple):
        return lower[(..., *points)], upper[(..., *points)]
    return lower[..., points], upper[..., points]
