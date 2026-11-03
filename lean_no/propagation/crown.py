from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace

import torch
from neuralop.layers.spectral_convolution import SpectralConv
from torch import nn

from lean_no.bounds.operators import (
    affine_bias,
    affine_weight,
    crown_relu_rule,
    crown_tanh_rule,
    is_identity,
    is_relu,
    is_tanh,
)
from lean_no.function_space.input import InputBounds
from lean_no.graph import Graph
from lean_no.integration import Integration
from lean_no.kernel import kernel_gain, kernel_with_partials
from lean_no.propagation.linear import (
    Bounds,
    PointBounds,
    propagate,
    sample_input,
)
from lean_no.propagation.linear import validate as validate_ibp
from lean_no.spectral import real_inverse

@dataclass(frozen=True)
class KernelField:

    axes: tuple[torch.Tensor, ...]
    lengths: tuple[float, ...]
    node_lower: torch.Tensor
    node_upper: torch.Tensor
    partial_lower: torch.Tensor
    partial_upper: torch.Tensor

    rule: Integration = Integration()

    @property
    def dimension(self) -> int:
        return len(self.axes)

    @property
    def resolution(self) -> tuple[int, ...]:
        return tuple(axis.numel() for axis in self.axes)

    @property
    def steps(self) -> tuple[float, ...]:
        return tuple(
            length / count for length, count in zip(self.lengths, self.resolution)
        )

    @property
    def partial_magnitude(self) -> torch.Tensor:
        return torch.maximum(
            self.partial_lower.abs(),
            self.partial_upper.abs(),
        )

    @property
    def cell_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (
            (1,) * (self.node_lower.ndim - self.dimension)
            + (self.dimension,)
            + (1,) * self.dimension
        )
        spacing = self.node_lower.new_tensor(self.steps).reshape(shape)
        radius = (spacing * self.partial_magnitude).sum(
            dim=self.node_lower.ndim - self.dimension
        )
        return self.node_lower - radius, self.node_upper + radius

    @property
    def cell_supremum(self) -> torch.Tensor:
        lower, upper = self.cell_bounds
        return torch.maximum(lower.abs(), upper.abs())

@dataclass(frozen=True)
class AffineBound:

    local: torch.Tensor
    kernel: KernelField | None
    bias: torch.Tensor

@dataclass(frozen=True)
class CrownState:

    lower: AffineBound
    upper: AffineBound
    axes: tuple[torch.Tensor, ...]
    lengths: tuple[float, ...]
    output_indices: torch.Tensor

    rule: Integration = Integration()

@dataclass(frozen=True)
class OutputCover:

    starts: torch.Tensor
    stops: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor
    length: float
    resolution: int
    evaluated: int

    @property
    def channels(self) -> int:
        return self.lower.shape[0]

    @property
    def cell_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.lower, self.upper

    @property
    def width(self) -> float:
        return (self.upper - self.lower).amax().item()

    def query(self, points):
        lower, upper = self.bounds(points[:, 0])
        return lower.T, upper.T

    def bounds(self, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not torch.isfinite(query).all():
            raise ValueError("output queries must be finite")
        if torch.any(query < 0) or torch.any(query > self.length):
            raise ValueError("output queries must lie in the domain")
        flat = query.detach().to(self.stops.device).reshape(-1)
        index = torch.floor(flat * (self.resolution / self.length)).long()
        index = index.clamp(max=self.resolution - 1)
        leaf = torch.searchsorted(self.stops, index, right=True)
        shape = (self.channels, *query.shape)
        return (
            self.lower[:, leaf].reshape(shape).to(query.device),
            self.upper[:, leaf].reshape(shape).to(query.device),
        )

def field_add(
    left: KernelField | None,
    right: KernelField | None,
) -> KernelField | None:
    if left is None:
        return right
    if right is None:
        return left
    return KernelField(
        axes=left.axes,
        lengths=left.lengths,
        node_lower=left.node_lower + right.node_lower,
        node_upper=left.node_upper + right.node_upper,
        partial_lower=left.partial_lower + right.partial_lower,
        partial_upper=left.partial_upper + right.partial_upper,
        rule=left.rule,
    )

def field_affine(
    field: KernelField,
    weight: torch.Tensor,
) -> KernelField:

    weight = weight.detach().to(field.node_lower)
    positive = weight.clamp(min=0)
    negative = weight.clamp(max=0)

    def apply(
        lower: torch.Tensor,
        upper: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # C̲_new = C̲ W⁺ + C̄ W⁻,  C̄_new = C̄ W⁺ + C̲ W⁻.
        return (
            torch.einsum("so...,oi->si...", lower, positive)
            + torch.einsum("so...,oi->si...", upper, negative),
            torch.einsum("so...,oi->si...", upper, positive)
            + torch.einsum("so...,oi->si...", lower, negative),
        )

    node_lower, node_upper = apply(field.node_lower, field.node_upper)
    partial_lower, partial_upper = apply(
        field.partial_lower,
        field.partial_upper,
    )
    return KernelField(
        field.axes,
        field.lengths,
        node_lower,
        node_upper,
        partial_lower,
        partial_upper,
        rule=field.rule,
    )

def field_relu_coefficient(
    field: KernelField,
    positive_slope: torch.Tensor,
    negative_slope: torch.Tensor,
) -> KernelField:

    positive_slope = positive_slope.to(field.node_lower)
    negative_slope = negative_slope.to(field.node_lower)
    while positive_slope.ndim < field.node_lower.ndim:
        positive_slope = positive_slope.unsqueeze(0)
        negative_slope = negative_slope.unsqueeze(0)
    positive_partial = positive_slope.unsqueeze(-field.dimension - 1)
    negative_partial = negative_slope.unsqueeze(-field.dimension - 1)

    def transform(
        lower: torch.Tensor,
        upper: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mapped_lower = torch.where(
            lower >= 0,
            positive_slope * lower,
            negative_slope * lower,
        )
        mapped_upper = torch.where(
            upper >= 0,
            positive_slope * upper,
            negative_slope * upper,
        )
        return mapped_lower, mapped_upper

    node_lower, node_upper = transform(field.node_lower, field.node_upper)
    cell_lower, cell_upper = field.cell_bounds
    # Active C̲_j ≥ 0: T' = α_pos; inactive C̄_j ≤ 0: T' = α_neg;
    # uncertain: |T'| ≤ max(α_pos, α_neg) M_C.
    positive_cell = (cell_lower >= 0).unsqueeze(-field.dimension - 1)
    negative_cell = (cell_upper <= 0).unsqueeze(-field.dimension - 1)
    positive_slope_positive = positive_partial.clamp(min=0)
    positive_slope_negative = positive_partial.clamp(max=0)
    partial_lower_positive = (
        positive_slope_positive * field.partial_lower
        + positive_slope_negative * field.partial_upper
    )
    partial_upper_positive = (
        positive_slope_positive * field.partial_upper
        + positive_slope_negative * field.partial_lower
    )
    negative_slope_positive = negative_partial.clamp(min=0)
    negative_slope_negative = negative_partial.clamp(max=0)
    partial_lower_negative = (
        negative_slope_positive * field.partial_lower
        + negative_slope_negative * field.partial_upper
    )
    partial_upper_negative = (
        negative_slope_positive * field.partial_upper
        + negative_slope_negative * field.partial_lower
    )
    uncertain_magnitude = field.partial_magnitude * torch.maximum(
        positive_partial,
        negative_partial,
    )
    partial_lower = torch.where(
        positive_cell,
        partial_lower_positive,
        torch.where(
            negative_cell,
            partial_lower_negative,
            -uncertain_magnitude,
        ),
    )
    partial_upper = torch.where(
        positive_cell,
        partial_upper_positive,
        torch.where(
            negative_cell,
            partial_upper_negative,
            uncertain_magnitude,
        ),
    )
    return KernelField(
        field.axes,
        field.lengths,
        node_lower,
        node_upper,
        partial_lower,
        partial_upper,
        rule=field.rule,
    )

def field_part(field: KernelField, positive: bool) -> KernelField:
    zero_derivative = torch.zeros_like(field.partial_lower)
    cell_lower, cell_upper = field.cell_bounds
    if positive:
        # v^+ = max(v, 0): active v̲_j ≥ 0, inactive v̄_j ≤ 0.
        node_lower = field.node_lower.clamp(min=0)
        node_upper = field.node_upper.clamp(min=0)
        active = (cell_lower >= 0).unsqueeze(-field.dimension - 1)
        inactive = (cell_upper <= 0).unsqueeze(-field.dimension - 1)
    else:
        # v^- = min(v, 0): active v̄_j ≤ 0, inactive v̲_j ≥ 0.
        node_lower = field.node_lower.clamp(max=0)
        node_upper = field.node_upper.clamp(max=0)
        active = (cell_upper <= 0).unsqueeze(-field.dimension - 1)
        inactive = (cell_lower >= 0).unsqueeze(-field.dimension - 1)
    partial_lower = torch.where(
        active,
        field.partial_lower,
        torch.where(
            inactive,
            zero_derivative,
            torch.minimum(field.partial_lower, zero_derivative),
        ),
    )
    partial_upper = torch.where(
        active,
        field.partial_upper,
        torch.where(
            inactive,
            zero_derivative,
            torch.maximum(field.partial_upper, zero_derivative),
        ),
    )
    return KernelField(
        field.axes,
        field.lengths,
        node_lower,
        node_upper,
        partial_lower,
        partial_upper,
        rule=field.rule,
    )

def field_scale(
    field: KernelField,
    scale: torch.Tensor,
) -> KernelField:
    scale = scale.to(field.node_lower)
    while scale.ndim < field.node_lower.ndim:
        scale = scale.unsqueeze(0)

    positive = scale.clamp(min=0)
    negative = scale.clamp(max=0)
    # w̲ = β̄⁺ v̲ + β̄⁻ v̄,  w̄ = β̄⁺ v̄ + β̄⁻ v̲.
    node_lower = positive * field.node_lower + negative * field.node_upper
    node_upper = positive * field.node_upper + negative * field.node_lower
    positive = positive.unsqueeze(-field.dimension - 1)
    negative = negative.unsqueeze(-field.dimension - 1)
    partial_lower = positive * field.partial_lower + negative * field.partial_upper
    partial_upper = positive * field.partial_upper + negative * field.partial_lower
    return KernelField(
        field.axes,
        field.lengths,
        node_lower,
        node_upper,
        partial_lower,
        partial_upper,
        rule=field.rule,
    )

def product_bounds(
    left_lower: torch.Tensor,
    left_upper: torch.Tensor,
    right_lower: torch.Tensor,
    right_upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    value = left_lower * right_lower
    lower = value
    upper = value
    for left, right in (
        (left_lower, right_upper),
        (left_upper, right_lower),
        (left_upper, right_upper),
    ):
        value = left * right
        lower = torch.minimum(lower, value)
        upper = torch.maximum(upper, value)
    return lower, upper

def field_product(
    field: KernelField,
    bounds: Bounds,
) -> KernelField:
    # f̲, f̄ from the four endpoint products u_j v_j.
    node_lower, node_upper = product_bounds(
        field.node_lower,
        field.node_upper,
        bounds.node_lower.unsqueeze(0),
        bounds.node_upper.unsqueeze(0),
    )
    cell_lower, cell_upper = field.cell_bounds
    bound_cell_lower, bound_cell_upper = bounds.cell_bounds
    # (uv)' = u' v + u v'.
    first_lower, first_upper = product_bounds(
        field.partial_lower,
        field.partial_upper,
        bound_cell_lower[None, :, None],
        bound_cell_upper[None, :, None],
    )
    second_lower, second_upper = product_bounds(
        cell_lower[:, :, None],
        cell_upper[:, :, None],
        bounds.partial_lower.unsqueeze(0),
        bounds.partial_upper.unsqueeze(0),
    )
    partial_lower = first_lower + second_lower
    partial_upper = first_upper + second_upper
    return KernelField(
        field.axes,
        field.lengths,
        node_lower,
        node_upper,
        partial_lower,
        partial_upper,
        rule=field.rule,
    )

def reverse_grid(values: torch.Tensor, dimension: int) -> torch.Tensor:
    for axis, count in enumerate(values.shape[-dimension:]):
        index = -torch.arange(count, device=values.device) % count
        values = values.index_select(values.ndim - dimension + axis, index)
    return values

@dataclass(frozen=True)
class KernelPlan:

    reversed_values: torch.Tensor
    reversed_partials: tuple[torch.Tensor, ...]
    gradient_gain: torch.Tensor
    hessian_gain: torch.Tensor

@dataclass(frozen=True)
class AdjointPlan:

    values: torch.Tensor
    derivatives: tuple[torch.Tensor, ...]
    value_radius: torch.Tensor
    derivative_radii: tuple[torch.Tensor, ...]
    hessian_gain: torch.Tensor

@dataclass(frozen=True)
class FrequencyPlan:

    values: torch.Tensor
    derivatives: tuple[torch.Tensor, ...]
    absolute_values: torch.Tensor
    absolute_derivatives: tuple[torch.Tensor, ...]

def kernel_cache_key(
    kind: str,
    conv: SpectralConv,
    axes: tuple[torch.Tensor, ...],
    lengths: tuple[float, ...],
    reference: torch.Tensor,
) -> tuple:
    return (
        kind,
        id(conv),
        tuple(axis.numel() for axis in axes),
        lengths,
        reference.device,
        reference.dtype,
    )

def kernel_parts(
    conv: SpectralConv,
    grid: torch.Tensor | tuple[torch.Tensor, ...],
    L: float | tuple[float, ...],
    reference: torch.Tensor,
    cache: dict | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    axes = grid if isinstance(grid, tuple) else (grid,)
    lengths = L if isinstance(L, tuple) else (float(L),)
    key = kernel_cache_key("parts", conv, axes, lengths, reference)
    if cache is not None and key in cache:
        return cache[key]
    aligned = tuple(
        axis.to(device=reference.device, dtype=reference.dtype) for axis in axes
    )
    values, partials = kernel_with_partials(conv, aligned, lengths)
    result = values, partials
    if cache is not None:
        cache[key] = result
    return result

def kernel_plan(
    conv: SpectralConv,
    axes: tuple[torch.Tensor, ...],
    lengths: tuple[float, ...],
    reference: torch.Tensor,
    cache: dict | None = None,
) -> KernelPlan:
    key = kernel_cache_key("plan", conv, axes, lengths, reference)
    if cache is not None and key in cache:
        return cache[key]
    values, partials = kernel_parts(conv, axes, lengths, reference, cache)
    dimension = len(axes)
    gradient_gain = torch.stack(
        tuple(
            kernel_gain(
                conv,
                lengths,
                tuple(int(index == axis) for index in range(dimension)),
                reference.dtype,
            )
            for axis in range(dimension)
        ),
        dim=2,
    ).to(reference)
    rows = []
    for row in range(dimension):
        entries = []
        for column in range(dimension):
            orders = [0] * dimension
            orders[row] += 1
            orders[column] += 1
            entries.append(kernel_gain(conv, lengths, tuple(orders), reference.dtype))
        rows.append(torch.stack(entries, dim=2))
    hessian_gain = torch.stack(rows, dim=2).to(reference)
    result = KernelPlan(
        reverse_grid(values, dimension),
        tuple(-reverse_grid(partial, dimension) for partial in partials),
        gradient_gain,
        hessian_gain,
    )
    if cache is not None:
        cache[key] = result
    return result

def adjoint_plan(
    conv: SpectralConv,
    axes: tuple[torch.Tensor, ...],
    lengths: tuple[float, ...],
    reference: torch.Tensor,
    cache: dict | None = None,
) -> AdjointPlan:
    key = kernel_cache_key("adjoint", conv, axes, lengths, reference)
    if cache is not None and key in cache:
        return cache[key]
    plan = kernel_plan(conv, axes, lengths, reference, cache)
    steps = reference.new_tensor(
        tuple(length / axis.numel() for axis, length in zip(axes, lengths))
    )
    values = plan.reversed_values.transpose(0, 1)
    derivatives = tuple(partial.transpose(0, 1) for partial in plan.reversed_partials)
    gradient_gain = plan.gradient_gain.permute(1, 0, 2)
    hessian_gain = plan.hessian_gain.permute(1, 0, 2, 3)
    result = AdjointPlan(
        values,
        derivatives,
        (gradient_gain * steps[None, None]).sum(dim=2),
        tuple(
            (hessian_gain[:, :, row] * steps[None, None]).sum(dim=2)
            for row in range(len(axes))
        ),
        hessian_gain,
    )
    if cache is not None:
        cache[key] = result
    return result

def frequency_plan(
    conv: SpectralConv,
    axes: tuple[torch.Tensor, ...],
    lengths: tuple[float, ...],
    reference: torch.Tensor,
    cache: dict | None = None,
) -> FrequencyPlan:
    key = kernel_cache_key("frequency", conv, axes, lengths, reference)
    if cache is not None and key in cache:
        return cache[key]
    plan = adjoint_plan(conv, axes, lengths, reference, cache)
    dimensions = tuple(range(-len(axes), 0))
    result = FrequencyPlan(
        torch.fft.rfftn(plan.values, dim=dimensions),
        tuple(torch.fft.rfftn(value, dim=dimensions) for value in plan.derivatives),
        torch.fft.rfftn(plan.values.abs(), dim=dimensions),
        tuple(
            torch.fft.rfftn(value.abs(), dim=dimensions) for value in plan.derivatives
        ),
    )
    if cache is not None:
        cache[key] = result
    return result

def frequency_conv(
    kernel_frequency: torch.Tensor,
    values_frequency: torch.Tensor,
    resolution: tuple[int, ...],
) -> torch.Tensor:
    output_frequency = torch.einsum(
        "oi...,si...->so...",
        kernel_frequency,
        values_frequency,
    )
    return real_inverse(output_frequency, resolution) / math.prod(resolution)

def add_constant(
    values: torch.Tensor,
    coefficient: torch.Tensor,
    source_mean: torch.Tensor,
    dimension: int,
) -> torch.Tensor:
    constant = torch.einsum("io,so->si", coefficient, source_mean)
    return values + constant.reshape(
        constant.shape[0],
        constant.shape[1],
        *((1,) * dimension),
    )

def adjoint_field(conv, field, cache=None):
    if field is None:
        return None
    from lean_no.integration import AdjointKernel

    return field.rule(field, AdjointKernel(conv, cache), field.lengths)

def local_kernel(
    conv: SpectralConv,
    local: torch.Tensor,
    axes: tuple[torch.Tensor, ...],
    lengths: tuple[float, ...],
    output_indices: torch.Tensor,
    cache: dict,
    rule=Integration(),
) -> KernelField:
    dimension = len(axes)
    resolution = tuple(axis.numel() for axis in axes)
    steps = tuple(length / count for length, count in zip(lengths, resolution))
    plan = kernel_plan(conv, axes, lengths, local, cache)
    current = output_indices
    indices = []
    for size in reversed(resolution):
        indices.append(current % size)
        current = torch.div(current, size, rounding_mode="floor")
    indices = tuple(reversed(indices))
    spatial = tuple(range(2, 2 + dimension))
    nodes = []
    partial_nodes = [[] for _ in range(dimension)]
    for specification in range(local.shape[0]):
        # C_loc(x_j, s) = P(x_j) K(x_j - s) by rolling K̃.
        shifts = tuple(int(index[specification]) for index in indices)
        shifted = torch.roll(plan.reversed_values, shifts, spatial)
        nodes.append(torch.einsum("o,oi...->i...", local[specification], shifted))
        for axis, partial in enumerate(plan.reversed_partials):
            shifted = torch.roll(partial, shifts, spatial)
            partial_nodes[axis].append(
                torch.einsum("o,oi...->i...", local[specification], shifted)
            )
    node = torch.stack(nodes)
    partial_nodes = [torch.stack(values) for values in partial_nodes]

    partial_lower = []
    partial_upper = []
    for row in range(dimension):
        derivative = partial_nodes[row]
        radius = local.new_zeros(local.shape[0], conv.in_channels)
        for column, spacing in enumerate(steps):
            # ∑_q h_q |P| M_{K,pq}.
            radius = radius + spacing * torch.einsum(
                "so,oi->si",
                local.abs(),
                plan.hessian_gain[:, :, row, column],
            )
        radius = radius.reshape(
            local.shape[0],
            conv.in_channels,
            *((1,) * dimension),
        )
        # ∂_p C_loc ∈ [∂_p (P K) ± ∑_q h_q |P| M_{K,pq}].
        partial_lower.append(derivative - radius)
        partial_upper.append(derivative + radius)
    partial_lower = torch.stack(partial_lower, dim=2)
    partial_upper = torch.stack(partial_upper, dim=2)
    return KernelField(
        axes=axes,
        lengths=lengths,
        node_lower=node,
        node_upper=node,
        partial_lower=partial_lower,
        partial_upper=partial_upper,
        rule=rule,
    )

def initial(
    output_bounds: Bounds,
    flat: torch.Tensor,
) -> CrownState:
    point_count = math.prod(output_bounds.resolution)
    specification_count = flat.numel()
    channels = torch.div(flat, point_count, rounding_mode="floor")
    points = flat % point_count
    local = output_bounds.node_lower.new_zeros(
        specification_count,
        output_bounds.channels,
    )
    local[torch.arange(specification_count, device=flat.device), channels] = 1
    bias = output_bounds.node_lower.new_zeros(specification_count)
    affine = AffineBound(local, None, bias)
    return CrownState(
        lower=affine,
        upper=affine,
        axes=output_bounds.axes,
        lengths=output_bounds.lengths,
        output_indices=points,
        rule=output_bounds.rule,
    )

def relu_bound(
    bound: AffineBound,
    preactivation: Bounds,
    output_indices: torch.Tensor,
    upper: bool,
) -> AffineBound:
    local_lower = preactivation.node_lower.reshape(
        preactivation.channels,
        -1,
    )[:, output_indices].T
    local_upper = preactivation.node_upper.reshape(
        preactivation.channels,
        -1,
    )[:, output_indices].T
    local_lower_slope, local_upper_slope, local_upper_bias = crown_relu_rule(
        local_lower,
        local_upper,
    )
    # T^U(P;x) = P⁺ ᾱ(x) + P⁻ α̲(x); T^L swaps ᾱ and α̲.
    positive_local_slope = local_upper_slope if upper else local_lower_slope
    negative_local_slope = local_lower_slope if upper else local_upper_slope
    local_positive = bound.local.clamp(min=0)
    local_negative = bound.local.clamp(max=0)
    local = (
        local_positive * positive_local_slope + local_negative * negative_local_slope
    )
    # R^U(P;x) = P⁺ β̄(x); R^L(P;x) = P⁻ β̄(x).
    local_beta = (local_positive if upper else local_negative) * local_upper_bias
    bias_increment = local_beta.sum(dim=-1)
    bias = bound.bias + bias_increment

    kernel_field = bound.kernel
    if kernel_field is not None:
        cell_lower, cell_upper = preactivation.cell_bounds
        lower_slope, upper_slope, upper_bias = crown_relu_rule(
            cell_lower,
            cell_upper,
        )
        positive_slope = upper_slope if upper else lower_slope
        negative_slope = lower_slope if upper else upper_slope
        # ⟨R(C;·), 1⟩_L = (1/L) ∫ C⁺ β̄ (upper) or C⁻ β̄ (lower).
        beta_part = field_part(
            kernel_field,
            positive=upper,
        )
        beta_field = field_scale(beta_part, upper_bias)
        beta_lower, beta_upper = beta_field.rule(beta_field)
        beta = beta_upper if upper else beta_lower
        bias = bias + beta.sum(dim=-1)
        # C ← T(C;s).
        kernel_field = field_relu_coefficient(
            kernel_field,
            positive_slope,
            negative_slope,
        )

    return AffineBound(
        local,
        kernel_field,
        bias,
    )

def backward_relu(
    state: CrownState,
    preactivation: Bounds,
) -> CrownState:
    return replace(
        state,
        lower=relu_bound(
            state.lower,
            preactivation,
            state.output_indices,
            upper=False,
        ),
        upper=relu_bound(
            state.upper,
            preactivation,
            state.output_indices,
            upper=True,
        ),
    )

def tanh_bound(
    bound: AffineBound,
    preactivation: Bounds,
    upper: bool,
) -> AffineBound:
    cell_lower, cell_upper = preactivation.cell_bounds
    spatial = tuple(range(1, cell_lower.ndim))
    lower = cell_lower.amin(dim=spatial)
    high = cell_upper.amax(dim=spatial)
    return affine_rule_bound(bound, preactivation, crown_tanh_rule(lower, high), upper)

def affine_rule_bound(bound, preactivation, rule, upper):
    low_slope, low_bias, high_slope, high_bias = rule
    shape = (preactivation.channels,) + (1,) * preactivation.dimension
    low_slope = low_slope.reshape(shape)
    low_bias = low_bias.reshape(shape)
    high_slope = high_slope.reshape(shape)
    high_bias = high_bias.reshape(shape)
    pos_slope = high_slope if upper else low_slope
    neg_slope = low_slope if upper else high_slope
    pos_bias = high_bias if upper else low_bias
    neg_bias = low_bias if upper else high_bias

    positive = bound.local.clamp(min=0)
    negative = bound.local.clamp(max=0)
    local_pos_slope = pos_slope.reshape(-1)
    local_neg_slope = neg_slope.reshape(-1)
    local_pos_bias = pos_bias.reshape(-1)
    local_neg_bias = neg_bias.reshape(-1)
    local = positive * local_pos_slope + negative * local_neg_slope
    bias = bound.bias + (positive * local_pos_bias + negative * local_neg_bias).sum(
        dim=-1
    )

    kernel = bound.kernel
    if kernel is not None:
        positive_field = field_part(kernel, True)
        negative_field = field_part(kernel, False)
        bias_field = field_add(
            field_scale(positive_field, pos_bias),
            field_scale(negative_field, neg_bias),
        )
        integral_lower, integral_upper = bias_field.rule(bias_field)
        integral = integral_upper if upper else integral_lower
        bias = bias + integral.sum(dim=-1)
        kernel = field_add(
            field_scale(positive_field, pos_slope),
            field_scale(negative_field, neg_slope),
        )
    return AffineBound(local, kernel, bias)

def backward_tanh(
    state: CrownState,
    preactivation: Bounds,
) -> CrownState:
    return replace(
        state,
        lower=tanh_bound(state.lower, preactivation, False),
        upper=tanh_bound(state.upper, preactivation, True),
    )

def backward_normalization(state, source, module):
    from lean_no.bounds.normalization import normalization_rule

    scale, low, high, _ = normalization_rule(source, module)
    rule = (scale, low, scale, high)
    return replace(
        state,
        lower=affine_rule_bound(state.lower, source, rule, False),
        upper=affine_rule_bound(state.upper, source, rule, True),
    )

def backward_affine(
    bound: AffineBound,
    module: nn.Module,
    upper: bool,
) -> AffineBound:
    if isinstance(module, nn.Identity):
        return bound

    weight = affine_weight(module).detach().to(bound.local)
    module_bias = affine_bias(module, bound.local)
    local_bias = bound.local.new_zeros(bound.local.shape[0])
    kernel_bias = local_bias
    if module_bias is not None:
        # b · P.
        local_bias = torch.einsum("so,o->s", bound.local, module_bias)
        if bound.kernel is not None:
            bias_field = field_affine(
                bound.kernel,
                module_bias.detach().to(bound.kernel.node_lower)[:, None],
            )
            # ⟨C, b⟩_L = (1/L) ∫ C b.
            integral_lower, integral_upper = bias_field.rule(bias_field)
            kernel_bias = (integral_upper if upper else integral_lower).squeeze(-1)
    # P ← Wᵀ P,  C ← Wᵀ C.
    kernel_field = None if bound.kernel is None else field_affine(bound.kernel, weight)
    return AffineBound(
        local=torch.einsum("so,oi->si", bound.local, weight),
        kernel=kernel_field,
        bias=bound.bias + local_bias + kernel_bias,
    )

def backward_spectral(
    bound: AffineBound,
    conv: SpectralConv,
    axes: tuple[torch.Tensor, ...],
    lengths: tuple[float, ...],
    output_indices: torch.Tensor,
    cache: dict,
    upper: bool,
    rule=Integration(),
) -> AffineBound:
    conv_bias = getattr(conv, "bias", None)
    bias = bound.bias
    if conv_bias is not None:
        conv_bias = conv_bias.detach().to(bound.local).reshape(-1)
        # b · P.
        local_bias = torch.einsum("so,o->s", bound.local, conv_bias)
        bias = bias + local_bias
        if bound.kernel is not None:
            bias_field = field_affine(
                bound.kernel,
                conv_bias[:, None],
            )
            # ⟨C, b⟩_L.
            integral_lower, integral_upper = bias_field.rule(bias_field)
            integral = integral_upper if upper else integral_lower
            bias = bias + integral.squeeze(-1)

    # C_new = (K ⋆ C) + P K(x-s),  P_new = 0.
    adjoint = adjoint_field(conv, bound.kernel, cache)
    localized_kernel = local_kernel(
        conv,
        bound.local,
        axes,
        lengths,
        output_indices,
        cache,
        rule,
    )
    return AffineBound(
        local=bound.local.new_zeros(
            bound.local.shape[0],
            conv.in_channels,
        ),
        kernel=field_add(adjoint, localized_kernel),
        bias=bias,
    )

def backward_graph(state, graph, preactivations, cache=None):
    cache = {} if cache is None else cache
    pending = {len(graph.ops): state}

    def merge(left, right):
        return AffineBound(
            left.local + right.local,
            field_add(left.kernel, right.kernel),
            left.bias + right.bias,
        )

    for index in range(len(graph.ops), 0, -1):
        if index not in pending:
            continue
        state = pending.pop(index)
        op = graph.ops[index - 1]
        if op.kind == "add":
            zero = replace(
                state,
                lower=replace(state.lower, bias=torch.zeros_like(state.lower.bias)),
                upper=replace(state.upper, bias=torch.zeros_like(state.upper.bias)),
            )
            contributions = (state, zero)
        else:
            if op.kind == "activation":
                name = op.data["activation"]
                if is_relu(name):
                    state = backward_relu(state, preactivations[op.name])
                elif is_tanh(name):
                    state = backward_tanh(state, preactivations[op.name])
            elif op.kind == "normalize":
                state = backward_normalization(state, preactivations[op.name], op)
            elif op.kind == "affine":
                state = replace(
                    state,
                    lower=backward_affine(state.lower, op, upper=False),
                    upper=backward_affine(state.upper, op, upper=True),
                )
            elif op.kind == "spectral":
                state = replace(
                    state,
                    lower=backward_spectral(
                        state.lower,
                        op,
                        state.axes,
                        state.lengths,
                        state.output_indices,
                        cache,
                        upper=False,
                        rule=state.rule,
                    ),
                    upper=backward_spectral(
                        state.upper,
                        op,
                        state.axes,
                        state.lengths,
                        state.output_indices,
                        cache,
                        upper=True,
                        rule=state.rule,
                    ),
                )
            else:
                raise NotImplementedError(op.kind)
            contributions = (state,)
        for parent, value in zip(op.parents, contributions):
            if parent in pending:
                previous = pending[parent]
                value = replace(
                    value,
                    lower=merge(previous.lower, value.lower),
                    upper=merge(previous.upper, value.upper),
                )
            pending[parent] = value
    return pending[0]

def concretize(
    state: CrownState,
    input_bounds: Bounds,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_lower = input_bounds.node_lower.reshape(
        input_bounds.channels,
        -1,
    )[:, state.output_indices].T
    input_upper = input_bounds.node_upper.reshape(
        input_bounds.channels,
        -1,
    )[:, state.output_indices].T
    # P⁺ a̲ + P⁻ ā  (lower);  P⁺ ā + P⁻ a̲  (upper).
    lower_local = (
        state.lower.local.clamp(min=0) * input_lower
        + state.lower.local.clamp(max=0) * input_upper
    ).sum(dim=-1)
    upper_local = (
        state.upper.local.clamp(min=0) * input_upper
        + state.upper.local.clamp(max=0) * input_lower
    ).sum(dim=-1)
    lower = state.lower.bias + lower_local
    upper = state.upper.bias + upper_local
    if state.lower.kernel is not None:
        # ⟨C, a⟩_L sign-split, with quadrature remainder on C a.
        product = field_product(state.lower.kernel, input_bounds)
        integral_lower, _ = product.rule(product)
        lower = lower + integral_lower.sum(dim=-1)
    if state.upper.kernel is not None:
        product = field_product(state.upper.kernel, input_bounds)
        _, integral_upper = product.rule(product)
        upper = upper + integral_upper.sum(dim=-1)
    return lower, upper

def bound_nodes(
    graph: Graph,
    input_bounds: Bounds,
    forward_output: Bounds,
    recorded: dict,
    flat: torch.Tensor,
    chunk_size: int,
    cache: dict,
    progress=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    lower_chunks: list[torch.Tensor] = []
    upper_chunks: list[torch.Tensor] = []
    for start in range(0, flat.numel(), chunk_size):
        selected = flat[start : start + chunk_size]
        state = initial(forward_output, selected)
        state = backward_graph(state, graph, recorded, cache)
        lower, upper = concretize(state, input_bounds)
        lower_chunks.append(lower)
        upper_chunks.append(upper)
        if progress is not None:
            progress(
                dict(
                    phase="CROWN",
                    completed=min(start + chunk_size, flat.numel()),
                    total=flat.numel(),
                )
            )
    return torch.cat(lower_chunks), torch.cat(upper_chunks)

@torch.no_grad()
def graph(
    graph: Graph,
    n: int | tuple[int, ...],
    L: float | tuple[float, ...],
    function: InputBounds,
    output_chunk_size: int = 64,
    query: torch.Tensor | None = None,
    progress=None,
    rule=Integration(),
) -> Bounds | PointBounds:
    if output_chunk_size <= 0:
        raise ValueError("output_chunk_size must be positive")
    start_time = time.perf_counter()
    input_bounds = replace(sample_input(graph, n, L, function), rule=rule)
    recorded: dict = {}
    forward_output = propagate(graph, input_bounds, recorded)
    cache: dict = {}
    if graph.device.type == "cuda":
        torch.cuda.synchronize(graph.device)
    forward_seconds = time.perf_counter() - start_time
    channels = forward_output.channels
    point_count = math.prod(forward_output.resolution)
    reference = forward_output.node_lower
    if query is None:
        flat = torch.arange(channels * point_count, device=reference.device)
        shape = forward_output.node_lower.shape
        query_index = None
        distance = None
    else:
        points = query.detach().to(reference)
        query_index, distance = forward_output.cell(points)
        channel = torch.arange(channels, device=reference.device)
        flat = (channel[:, None] * point_count + query_index.reshape(1, -1)).reshape(-1)
        query_shape = query.shape if forward_output.dimension == 1 else query.shape[1:]
        shape = (channels, *query_shape)

    value_start = time.perf_counter()
    crown_lower, crown_upper = bound_nodes(
        graph,
        input_bounds,
        forward_output,
        recorded,
        flat,
        output_chunk_size,
        cache,
        progress=progress,
    )
    if reference.is_cuda:
        torch.cuda.synchronize(reference.device)
    value_seconds = time.perf_counter() - value_start
    node_part_lower, node_part_upper = forward_output.partial_nodes
    stats: dict[str, float | str] = {
        "derivative_method": "ibp",
        "forward_seconds": forward_seconds,
        "value_crown_seconds": value_seconds,
        "derivative_seconds": 0.0,
        "total_seconds": time.perf_counter() - start_time,
        "value_node_width": (crown_upper - crown_lower).amax().item(),
        "derivative_node_width": (node_part_upper - node_part_lower).amax().item(),
        "derivative_cell_width": (
            forward_output.partial_upper - forward_output.partial_lower
        )
        .amax()
        .item(),
    }
    if query_index is not None and distance is not None:
        crown_lower = crown_lower.reshape(channels, -1)
        crown_upper = crown_upper.reshape(channels, -1)
        if forward_output.dimension == 1:
            distance = distance.reshape(1, -1)
        else:
            distance = distance.reshape(forward_output.dimension, -1)
        flat_index = query_index.reshape(-1)
        magnitude = forward_output.partial_magnitude.reshape(
            channels,
            forward_output.dimension,
            -1,
        )[:, :, flat_index]
        node_lower, node_upper = forward_output.partial_nodes
        node_lower = node_lower.reshape(channels, forward_output.dimension, -1)
        node_upper = node_upper.reshape(channels, forward_output.dimension, -1)
        node_lower = node_lower[:, :, flat_index]
        node_upper = node_upper[:, :, flat_index]
        cell_lower = forward_output.partial_lower.reshape(
            channels, forward_output.dimension, -1
        )[:, :, flat_index]
        cell_upper = forward_output.partial_upper.reshape(
            channels, forward_output.dimension, -1
        )[:, :, flat_index]
        stats["derivative_node_width"] = (node_upper - node_lower).amax().item()
        stats["derivative_cell_width"] = (cell_upper - cell_lower).amax().item()
        radius = (magnitude * distance[None]).sum(dim=1)
        low = (crown_lower - radius).reshape(shape)
        high = (crown_upper + radius).reshape(shape)
        stats["value_cell_width"] = (high - low).amax().item()
        derivative_lower = cell_lower.reshape(
            channels, forward_output.dimension, *shape[1:]
        )
        derivative_upper = cell_upper.reshape(
            channels, forward_output.dimension, *shape[1:]
        )
        return PointBounds(
            query.detach().cpu().clone(),
            low,
            high,
            stats,
            derivative_lower,
            derivative_upper,
        )
    crown_lower = crown_lower.reshape(shape)
    crown_upper = crown_upper.reshape(shape)
    result = Bounds(
        axes=forward_output.axes,
        lengths=forward_output.lengths,
        node_lower=crown_lower,
        node_upper=crown_upper,
        partial_lower=forward_output.partial_lower,
        partial_upper=forward_output.partial_upper,
        partial_node_lower=forward_output.partial_node_lower,
        partial_node_upper=forward_output.partial_node_upper,
        stats=stats,
    )
    stats["value_cell_width"] = (
        (result.cell_bounds[1] - result.cell_bounds[0]).amax().item()
    )
    return result

def output_partition(forward, output_cells):
    count = forward.resolution[0]
    mass = forward.partial_magnitude[:, 0].double() * (forward.lengths[0] / count)
    if not torch.isfinite(mass).all():
        raise ValueError("fixed output cover requires finite derivative bounds")
    placement = torch.cat((mass.new_zeros(1), mass.amax(0).cumsum(0)))
    levels = torch.arange(output_cells + 1, device=mass.device, dtype=mass.dtype)
    edges = torch.searchsorted(
        placement, levels * (placement[-1] / output_cells)
    ).clamp_max(count)
    edges[0], edges[-1] = 0, count
    return torch.unique_consecutive(edges)

@torch.no_grad()
def fixed_graph(
    graph: Graph,
    n: int | tuple[int, ...],
    L: float | tuple[float, ...],
    function: InputBounds,
    output_cells: int,
    output_chunk_size: int = 64,
    progress=None,
    rule=Integration(),
) -> OutputCover:
    resolution = (n,) if isinstance(n, int) else n
    lengths = L if isinstance(L, tuple) else (float(L),)
    if len(resolution) != 1 or len(lengths) != 1:
        raise ValueError("fixed CROWN output covers require a one-dimensional grid")
    if isinstance(output_cells, torch.Tensor):
        if (
            output_cells.ndim != 1
            or output_cells.numel() < 2
            or output_cells.dtype not in (torch.int32, torch.int64)
            or int(output_cells[0]) != 0
            or int(output_cells[-1]) != resolution[0]
            or not bool((output_cells[1:] > output_cells[:-1]).all())
        ):
            raise ValueError(
                "output partition must contain increasing integer edges from zero to resolution"
            )
    elif type(output_cells) is not int or not 1 <= output_cells <= resolution[0]:
        raise ValueError(
            "output_cells must be an integer between one and the resolution"
        )
    if type(output_chunk_size) is not int or output_chunk_size < 1:
        raise ValueError("output_chunk_size must be positive")
    input_bounds = replace(
        sample_input(graph, resolution, lengths, function), rule=rule
    )
    recorded: dict = {}
    forward = propagate(graph, input_bounds, recorded)
    channels, count = forward.channels, resolution[0]
    mass = forward.partial_magnitude[:, 0].double() * (lengths[0] / count)
    if not torch.isfinite(mass).all():
        raise ValueError("fixed output cover requires finite derivative bounds")
    prefix = torch.cat((mass.new_zeros(channels, 1), mass.cumsum(-1)), -1)
    edges = (
        output_cells.to(device=mass.device, dtype=torch.long)
        if isinstance(output_cells, torch.Tensor)
        else output_partition(forward, output_cells)
    )
    nodes = torch.cat((edges[:-1], edges.new_tensor([count - 1])))
    channel = torch.arange(channels, device=mass.device)
    flat = (channel[:, None] * count + nodes[None]).reshape(-1)
    lower, upper = bound_nodes(
        graph,
        input_bounds,
        forward,
        recorded,
        flat,
        output_chunk_size,
        {},
        progress=progress,
    )
    lower, upper = lower.reshape(channels, -1), upper.reshape(channels, -1)
    right_lower = torch.cat((lower[:, 1:-1].double(), lower[:, -1:] - mass[:, -1:]), -1)
    right_upper = torch.cat((upper[:, 1:-1].double(), upper[:, -1:] + mass[:, -1:]), -1)
    radius = prefix[:, edges[1:]] - prefix[:, edges[:-1]]
    return OutputCover(
        edges[:-1],
        edges[1:],
        (lower[:, :-1] + right_lower - radius) / 2,
        (upper[:, :-1] + right_upper + radius) / 2,
        lengths[0],
        count,
        nodes.numel(),
    )

def validate(
    model: nn.Module,
    resolution: tuple[int, ...] | None,
    L: tuple[float, ...],
) -> None:
    validate_ibp(model, resolution, L)
    from lean_no.adapters.fno import build_graph

    graph = build_graph(model, normalization=True)
    if any(
        not is_relu(activation)
        and not is_tanh(activation)
        and not is_identity(activation)
        for activation in graph.activations
    ):
        raise NotImplementedError("continuum CROWN supports ReLU, tanh and identity")
