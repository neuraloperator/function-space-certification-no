from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace

import torch
import torch.nn as nn

from lean_no.bounds.activations import tanh_hessian, tanh_prime
from lean_no.bounds.interval import product_bounds
from lean_no.bounds.operators import (
    affine_bias,
    affine_weight,
    gelu_bounds,
    gelu_lipschitz,
    is_gelu,
    is_identity,
    is_relu,
    is_tanh,
)
from lean_no.function_space.input import InputBounds
from lean_no.graph import Graph
from lean_no.integration import Integration

def make_grid(
    resolution: tuple[int, ...],
    L: tuple[float, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    axes = tuple(
        torch.arange(count, device=device, dtype=dtype) * (length / count)
        for count, length in zip(resolution, L)
    )
    return axes, torch.stack(torch.meshgrid(*axes, indexing="ij"))

@dataclass(frozen=True)
class Bounds:

    axes: tuple[torch.Tensor, ...]
    lengths: tuple[float, ...]
    node_lower: torch.Tensor
    node_upper: torch.Tensor
    partial_lower: torch.Tensor
    partial_upper: torch.Tensor
    partial_node_lower: torch.Tensor | None = None
    partial_node_upper: torch.Tensor | None = None
    second_lower: torch.Tensor | None = None
    second_upper: torch.Tensor | None = None
    stats: dict[str, float | str] | None = None
    rule: Integration = Integration()

    @property
    def channels(self) -> int:
        return self.node_lower.shape[0]

    @property
    def dimension(self) -> int:
        return self.node_lower.ndim - 1

    @property
    def resolution(self) -> tuple[int, ...]:
        return tuple(self.node_lower.shape[1:])

    @property
    def steps(self) -> tuple[float, ...]:
        return tuple(
            length / count for length, count in zip(self.lengths, self.resolution)
        )

    @property
    def partial_magnitude(self) -> torch.Tensor:
        return torch.maximum(self.partial_lower.abs(), self.partial_upper.abs())

    @property
    def partial_nodes(self) -> tuple[torch.Tensor, torch.Tensor]:
        lower = (
            self.partial_lower
            if self.partial_node_lower is None
            else self.partial_node_lower
        )
        upper = (
            self.partial_upper
            if self.partial_node_upper is None
            else self.partial_node_upper
        )
        return lower, upper

    @property
    def second_magnitude(self) -> torch.Tensor:
        if self.second_lower is None or self.second_upper is None:
            raise ValueError("second derivative bounds are unavailable")
        return torch.maximum(self.second_lower.abs(), self.second_upper.abs())

    @property
    def cell_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (1, self.dimension) + (1,) * self.dimension
        spacing = self.node_lower.new_tensor(self.steps).reshape(shape)
        radius = (spacing * self.partial_magnitude).sum(dim=1)
        return self.node_lower - radius, self.node_upper + radius

    @property
    def supremum(self) -> torch.Tensor:
        cell_lower, cell_upper = self.cell_bounds
        spatial = tuple(range(1, self.node_lower.ndim))
        return torch.maximum(cell_lower.abs(), cell_upper.abs()).amax(dim=spatial)

    @property
    def lipschitz(self) -> torch.Tensor:
        spatial = tuple(range(2, self.partial_magnitude.ndim))
        value = self.partial_magnitude.amax(dim=spatial)
        return value[:, 0] if self.dimension == 1 else value

    def query(self, points):
        points = points.to(self.node_lower)
        coordinates = points[:, 0] if self.dimension == 1 else points.T.contiguous()
        lower, upper = self.bounds(coordinates)
        return lower.T, upper.T

    @property
    def channel_width(self):
        lower, upper = self.cell_bounds
        return (upper - lower).flatten(1).amax(1)

    def cell(
        self,
        query: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not torch.isfinite(query).all():
            raise ValueError("quadrature queries must be finite")
        if self.dimension == 1:
            flat = query.reshape(1, -1)
        else:
            if query.ndim == 0 or query.shape[0] != self.dimension:
                raise ValueError("queries must have shape [dimension, ...]")
            flat = query.reshape(self.dimension, -1)
        lengths = query.new_tensor(self.lengths)[:, None]
        if torch.any(flat < 0) or torch.any(flat > lengths):
            raise ValueError("quadrature queries must lie in the domain")

        linear = torch.zeros(flat.shape[1], device=query.device, dtype=torch.long)
        distances = []
        for axis, count, values in zip(self.axes, self.resolution, flat):
            current = axis.to(device=query.device).to(dtype=query.dtype)
            index = torch.searchsorted(current, values, right=True) - 1
            index = index.clamp(0, current.numel() - 1)
            linear = linear * count + index
            distances.append(values - current[index])
        distance = torch.stack(distances)
        return linear, distance[0] if self.dimension == 1 else distance

    def lower(self, query: torch.Tensor) -> torch.Tensor:
        return self.bounds(query)[0]

    def upper(self, query: torch.Tensor) -> torch.Tensor:
        return self.bounds(query)[1]

    def bounds(self, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        index, distance = self.cell(query)
        if self.dimension == 1:
            distance = distance[None]
        magnitude = (
            self.partial_magnitude.to(device=query.device)
            .to(dtype=query.dtype)
            .reshape(
                self.channels,
                self.dimension,
                -1,
            )[:, :, index]
        )
        radius = (magnitude * distance[None]).sum(dim=1)
        shape = query.shape if self.dimension == 1 else query.shape[1:]
        lower = (
            self.node_lower.to(device=query.device)
            .to(dtype=query.dtype)
            .reshape(self.channels, -1)[:, index]
        )
        upper = (
            self.node_upper.to(device=query.device)
            .to(dtype=query.dtype)
            .reshape(self.channels, -1)[:, index]
        )
        return (
            (lower - radius).reshape(self.channels, *shape),
            (upper + radius).reshape(self.channels, *shape),
        )

    @classmethod
    def from_function(
        cls,
        function: InputBounds,
        axes: tuple[torch.Tensor, ...],
        coordinates: torch.Tensor,
        lengths: tuple[float, ...],
        second_order: bool = True,
    ) -> Bounds:
        dimension = len(axes)
        sample = coordinates[0] if dimension == 1 else coordinates
        shape = (function.channels, *(axis.numel() for axis in axes))
        node_lower = function.lower(sample).to(sample)
        node_upper = function.upper(sample).to(sample)
        if node_lower.shape != shape or node_upper.shape != shape:
            raise ValueError("input callbacks returned the wrong shape")
        if torch.any(node_lower > node_upper):
            raise ValueError("input lower bounds exceed upper bounds")
        lipschitz = function.lipschitz.to(sample)
        expected = (
            (function.channels,)
            if dimension == 1
            else (
                function.channels,
                dimension,
            )
        )
        if lipschitz.shape != expected:
            if dimension == 1:
                message = "[channels]"
            else:
                message = f"[channels, {dimension}]"
            raise ValueError(f"input lipschitz bounds must have shape {message}")
        # ∂_k a ∈ [-M_{a,k}, M_{a,k}] on every cell.
        magnitude = lipschitz.reshape(
            function.channels,
            dimension,
            *((1,) * dimension),
        ).expand(function.channels, dimension, *shape[1:])
        curvature = function.curvature if second_order else None
        if curvature is not None:
            if dimension == 1:
                curvature = (
                    curvature.to(sample).reshape(function.channels, 1).expand(shape)
                )
            else:
                if curvature.shape != (function.channels, dimension, dimension):
                    raise ValueError("2D curvature must have shape [channels, 2, 2]")
                curvature = curvature.to(sample).reshape(
                    function.channels, dimension, dimension, 1, 1
                )
                curvature = curvature.expand(
                    function.channels, dimension, dimension, *shape[1:]
                )
        result = cls(
            axes=axes,
            lengths=lengths,
            node_lower=node_lower,
            node_upper=node_upper,
            partial_lower=-magnitude,
            partial_upper=magnitude,
            partial_node_lower=None if function.curvature is None else -magnitude,
            partial_node_upper=None if function.curvature is None else magnitude,
            second_lower=None if curvature is None else -curvature,
            second_upper=curvature,
        )
        if dimension == 1:
            first = function.derivative_bounds(axes[0], lengths[0], 1)
            if first is not None:
                result = replace(
                    result,
                    partial_node_lower=first[0].unsqueeze(1),
                    partial_node_upper=first[1].unsqueeze(1),
                    partial_lower=first[2].unsqueeze(1),
                    partial_upper=first[3].unsqueeze(1),
                )
            if second_order and function.curvature is not None:
                second = function.derivative_bounds(axes[0], lengths[0], 2)
                if second is not None:
                    result = replace(
                        result, second_lower=second[2], second_upper=second[3]
                    )
        else:
            units = tuple(
                tuple(int(i == j) for i in range(dimension)) for j in range(dimension)
            )
            first = [
                function.derivative_bounds(coordinates, lengths, key) for key in units
            ]
            if all(item is not None for item in first):
                result = replace(
                    result,
                    partial_node_lower=torch.stack([item[0] for item in first], 1),
                    partial_node_upper=torch.stack([item[1] for item in first], 1),
                    partial_lower=torch.stack([item[2] for item in first], 1),
                    partial_upper=torch.stack([item[3] for item in first], 1),
                )
            if second_order and function.curvature is not None:
                second = [
                    [
                        function.derivative_bounds(
                            coordinates,
                            lengths,
                            tuple(a + b for a, b in zip(left, right)),
                        )
                        for right in units
                    ]
                    for left in units
                ]
                if all(item is not None for row in second for item in row):
                    result = replace(
                        result,
                        second_lower=torch.stack(
                            [
                                torch.stack([item[2] for item in row], 1)
                                for row in second
                            ],
                            1,
                        ),
                        second_upper=torch.stack(
                            [
                                torch.stack([item[3] for item in row], 1)
                                for row in second
                            ],
                            1,
                        ),
                    )
        return result

def append_grid(bounds: Bounds, coordinates: torch.Tensor) -> Bounds:
    dimension = bounds.dimension
    shape = (dimension, *((1,) * dimension))
    lengths = coordinates.new_tensor(bounds.lengths).reshape(shape)
    nodes = coordinates / lengths
    partials = (
        torch.diag(lengths.reshape(-1).reciprocal())
        .reshape(
            dimension,
            dimension,
            *((1,) * dimension),
        )
        .expand(dimension, dimension, *bounds.resolution)
    )
    node_lower = bounds.partial_node_lower
    node_upper = bounds.partial_node_upper
    second_lower = bounds.second_lower
    second_upper = bounds.second_upper
    if second_lower is not None:
        zeros = coordinates.new_zeros(dimension, *second_lower.shape[1:])
        second_lower = torch.cat((second_lower, zeros), dim=0)
        second_upper = torch.cat((second_upper, zeros), dim=0)
    return replace(
        bounds,
        node_lower=torch.cat((bounds.node_lower, nodes), dim=0),
        node_upper=torch.cat((bounds.node_upper, nodes), dim=0),
        partial_lower=torch.cat((bounds.partial_lower, partials), dim=0),
        partial_upper=torch.cat((bounds.partial_upper, partials), dim=0),
        partial_node_lower=(
            None if node_lower is None else torch.cat((node_lower, partials), dim=0)
        ),
        partial_node_upper=(
            None if node_upper is None else torch.cat((node_upper, partials), dim=0)
        ),
        second_lower=second_lower,
        second_upper=second_upper,
    )

@dataclass(frozen=True)
class PointBounds:

    points: torch.Tensor
    low: torch.Tensor
    high: torch.Tensor
    stats: dict[str, float | str] | None = None
    partial_lower: torch.Tensor | None = None
    partial_upper: torch.Tensor | None = None

    @property
    def channels(self) -> int:
        return self.low.shape[0]

    @property
    def channel_width(self):
        return (self.high - self.low).flatten(1).amax(1)

    def query(self, points):
        query = points[:, 0] if self.points.ndim == 1 else points.T.contiguous()
        lower, upper = self.bounds(query.to(dtype=self.points.dtype))
        return lower.reshape(self.channels, -1).T, upper.reshape(self.channels, -1).T

    def check(self, query: torch.Tensor) -> None:
        current = query.detach().cpu()
        if current.shape != self.points.shape or not torch.equal(
            current,
            self.points,
        ):
            raise ValueError("bounds were computed for a different query")

    def lower(self, query: torch.Tensor) -> torch.Tensor:
        self.check(query)
        return self.low.to(device=query.device).to(dtype=query.dtype)

    def upper(self, query: torch.Tensor) -> torch.Tensor:
        self.check(query)
        return self.high.to(device=query.device).to(dtype=query.dtype)

    def bounds(self, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.check(query)
        return (
            self.low.to(device=query.device).to(dtype=query.dtype),
            self.high.to(device=query.device).to(dtype=query.dtype),
        )

    @classmethod
    def from_bounds(cls, bounds: Bounds, query: torch.Tensor) -> PointBounds:
        low, high = bounds.bounds(query)
        return cls(query.detach().cpu().clone(), low, high)

def affine(
    module: nn.Module,
    bounds: Bounds,
) -> Bounds:
    if isinstance(module, nn.Identity):
        return bounds

    weight = affine_weight(module).detach().to(bounds.node_lower)
    bias = affine_bias(module, weight)
    positive = weight.clamp(min=0)
    negative = weight.clamp(max=0)
    # y̲ = W⁺ z̲ + W⁻ z̄,  ȳ = W⁺ z̄ + W⁻ z̲.
    node_lower = torch.einsum(
        "oi,i...->o...", positive, bounds.node_lower
    ) + torch.einsum("oi,i...->o...", negative, bounds.node_upper)
    node_upper = torch.einsum(
        "oi,i...->o...", positive, bounds.node_upper
    ) + torch.einsum("oi,i...->o...", negative, bounds.node_lower)
    # y̲' = W⁺ z̲' + W⁻ z̄',  ȳ' = W⁺ z̄' + W⁻ z̲'.
    partial_lower = torch.einsum(
        "oi,i...->o...", positive, bounds.partial_lower
    ) + torch.einsum("oi,i...->o...", negative, bounds.partial_upper)
    partial_upper = torch.einsum(
        "oi,i...->o...", positive, bounds.partial_upper
    ) + torch.einsum("oi,i...->o...", negative, bounds.partial_lower)
    if bias is not None:
        shape = (bias.numel(),) + (1,) * (node_lower.ndim - 1)
        bias = bias.detach().to(node_lower).reshape(shape)
        node_lower = node_lower + bias
        node_upper = node_upper + bias
    node_part_lower = None
    node_part_upper = None
    if bounds.partial_node_lower is not None and bounds.partial_node_upper is not None:
        old_part_lower = bounds.partial_node_lower
        old_part_upper = bounds.partial_node_upper
        node_part_lower = torch.einsum(
            "oi,i...->o...", positive, old_part_lower
        ) + torch.einsum("oi,i...->o...", negative, old_part_upper)
        node_part_upper = torch.einsum(
            "oi,i...->o...", positive, old_part_upper
        ) + torch.einsum("oi,i...->o...", negative, old_part_lower)
    second_lower = bounds.second_lower
    second_upper = bounds.second_upper
    if second_lower is not None and second_upper is not None:
        old_lower = second_lower
        old_upper = second_upper
        second_lower = torch.einsum(
            "oi,i...->o...", positive, old_lower
        ) + torch.einsum("oi,i...->o...", negative, old_upper)
        second_upper = torch.einsum(
            "oi,i...->o...", positive, old_upper
        ) + torch.einsum("oi,i...->o...", negative, old_lower)
    return replace(
        bounds,
        node_lower=node_lower,
        node_upper=node_upper,
        partial_lower=partial_lower,
        partial_upper=partial_upper,
        partial_node_lower=node_part_lower,
        partial_node_upper=node_part_upper,
        second_lower=second_lower,
        second_upper=second_upper,
    )

def activation(
    activation: Callable,
    bounds: Bounds,
) -> Bounds:
    if is_identity(activation):
        return bounds
    if is_relu(activation):
        cell_lower, cell_upper = bounds.cell_bounds
        # Active z̲_j > 0: keep z'; inactive z̄_j ≤ 0: 0;
        # uncertain: [min(z̲'_j, 0), max(z̄'_j, 0)].
        active = cell_lower > 0
        inactive = cell_upper <= 0
        while active.ndim < bounds.partial_lower.ndim:
            active = active.unsqueeze(1)
            inactive = inactive.unsqueeze(1)
        zero = torch.zeros_like(bounds.partial_lower)
        uncertain_lower = torch.minimum(bounds.partial_lower, zero)
        uncertain_upper = torch.maximum(bounds.partial_upper, zero)
        partial_lower = torch.where(
            active,
            bounds.partial_lower,
            torch.where(inactive, zero, uncertain_lower),
        )
        partial_upper = torch.where(
            active,
            bounds.partial_upper,
            torch.where(inactive, zero, uncertain_upper),
        )
        return replace(
            bounds,
            node_lower=torch.relu(bounds.node_lower),
            node_upper=torch.relu(bounds.node_upper),
            partial_lower=partial_lower,
            partial_upper=partial_upper,
            partial_node_lower=None,
            partial_node_upper=None,
            second_lower=None,
            second_upper=None,
        )

    if is_tanh(activation):
        node_slope_lower, node_slope_upper = tanh_prime(
            bounds.node_lower,
            bounds.node_upper,
        )
        old_node_lower, old_node_upper = bounds.partial_nodes
        node_part_lower, node_part_upper = product_bounds(
            node_slope_lower.unsqueeze(1),
            node_slope_upper.unsqueeze(1),
            old_node_lower,
            old_node_upper,
        )
        cell_lower, cell_upper = bounds.cell_bounds
        slope_lower, slope_upper = tanh_prime(cell_lower, cell_upper)
        partial_lower, partial_upper = product_bounds(
            slope_lower.unsqueeze(1),
            slope_upper.unsqueeze(1),
            bounds.partial_lower,
            bounds.partial_upper,
        )
        second_lower = None
        second_upper = None
        if bounds.second_lower is not None and bounds.second_upper is not None:
            d = bounds.dimension
            hlow = bounds.second_lower if d > 1 else bounds.second_lower[:, None, None]
            hupp = bounds.second_upper if d > 1 else bounds.second_upper[:, None, None]
            second_lower, second_upper = torch.empty_like(hlow), torch.empty_like(hupp)
            for left in range(d):
                for right in range(d):
                    low, high = tanh_hessian(
                        cell_lower,
                        cell_upper,
                        bounds.partial_lower[:, left],
                        bounds.partial_upper[:, left],
                        bounds.partial_lower[:, right],
                        bounds.partial_upper[:, right],
                        left == right,
                    )
                    extra_low, extra_high = product_bounds(
                        slope_lower,
                        slope_upper,
                        hlow[:, left, right],
                        hupp[:, left, right],
                    )
                    second_lower[:, left, right] = low + extra_low
                    second_upper[:, left, right] = high + extra_high
            if d == 1:
                second_lower, second_upper = (
                    second_lower[:, 0, 0],
                    second_upper[:, 0, 0],
                )
        return replace(
            bounds,
            node_lower=torch.tanh(bounds.node_lower),
            node_upper=torch.tanh(bounds.node_upper),
            partial_lower=partial_lower,
            partial_upper=partial_upper,
            partial_node_lower=node_part_lower,
            partial_node_upper=node_part_upper,
            second_lower=second_lower,
            second_upper=second_upper,
        )

    if is_gelu(activation):
        if isinstance(activation, nn.GELU) and activation.approximate != "none":
            raise NotImplementedError("only exact GELU interval bounds are supported")

        node_lower, node_upper = gelu_bounds(
            bounds.node_lower,
            bounds.node_upper,
        )
        # |h'| ≤ L_σ M_z with L_σ = Lip(GELU).
        derivative_magnitude = gelu_lipschitz * torch.maximum(
            bounds.partial_lower.abs(),
            bounds.partial_upper.abs(),
        )
        return replace(
            bounds,
            node_lower=node_lower,
            node_upper=node_upper,
            partial_lower=-derivative_magnitude,
            partial_upper=derivative_magnitude,
        )

    raise NotImplementedError("quadrature supports ReLU, GELU, tanh, or Identity")

def normalize(source, module):
    from lean_no.bounds.normalization import normalization_rule

    scale, low_bias, high_bias, scale_radius = normalization_rule(source, module)
    shape = (source.channels,) + (1,) * source.dimension
    scale = scale.reshape(shape)
    positive, negative = scale.clamp_min(0), scale.clamp_max(0)
    low = (
        positive * source.node_lower
        + negative * source.node_upper
        + low_bias.reshape(shape)
    )
    high = (
        positive * source.node_upper
        + negative * source.node_lower
        + high_bias.reshape(shape)
    )

    def derivatives(lower, upper):
        shape = (source.channels,) + (1,) * (lower.ndim - 1)
        return product_bounds(
            (scale.flatten() - scale_radius).reshape(shape),
            (scale.flatten() + scale_radius).reshape(shape),
            lower,
            upper,
        )

    partials = derivatives(source.partial_lower, source.partial_upper)
    nodes = derivatives(*source.partial_nodes)
    second = (
        (None, None)
        if source.second_lower is None
        else derivatives(source.second_lower, source.second_upper)
    )
    return replace(
        source,
        node_lower=low,
        node_upper=high,
        partial_lower=partials[0],
        partial_upper=partials[1],
        partial_node_lower=nodes[0],
        partial_node_upper=nodes[1],
        second_lower=second[0],
        second_upper=second[1],
    )

def add(
    left: Bounds,
    right: Bounds,
) -> Bounds:
    left_node_lower, left_node_upper = left.partial_nodes
    right_node_lower, right_node_upper = right.partial_nodes
    second_lower = None
    second_upper = None
    if left.second_lower is not None and right.second_lower is not None:
        second_lower = left.second_lower + right.second_lower
        second_upper = left.second_upper + right.second_upper
    return replace(
        left,
        node_lower=left.node_lower + right.node_lower,
        node_upper=left.node_upper + right.node_upper,
        partial_lower=left.partial_lower + right.partial_lower,
        partial_upper=left.partial_upper + right.partial_upper,
        partial_node_lower=left_node_lower + right_node_lower,
        partial_node_upper=left_node_upper + right_node_upper,
        second_lower=second_lower,
        second_upper=second_upper,
    )

def propagate(
    fno_graph: Graph,
    hidden: Bounds,
    preactivations: dict | None = None,
    channel_chunk: int | None = None,
) -> Bounds:
    from lean_no.graph import execute

    def apply(op, parents):
        source = parents[0]
        if op.kind == "affine":
            return affine(op, source)
        if op.kind == "spectral":
            result = source.rule(
                source, op, source.lengths, options={"channel_chunk": channel_chunk}
            )
            return replace(result, rule=source.rule)
        if op.kind == "normalize":
            if preactivations is not None:
                preactivations[op.name] = source
            return normalize(source, op)
        if op.kind == "add":
            return add(*parents)
        activation_fn = op.data["activation"]
        if preactivations is not None and (
            is_relu(activation_fn) or is_tanh(activation_fn)
        ):
            preactivations[op.name] = source
        return activation(activation_fn, source)

    return execute(fno_graph, hidden, apply)

def sample_input(
    fno_graph: Graph,
    resolution: int | tuple[int, ...],
    L: float | tuple[float, ...],
    function: InputBounds,
    second_order: bool = False,
) -> Bounds:
    if isinstance(resolution, int):
        resolution = (resolution,)
    if not isinstance(L, tuple):
        L = (float(L),)
    axes, coordinates = make_grid(
        resolution,
        L,
        fno_graph.device,
        fno_graph.dtype,
    )
    bounds = Bounds.from_function(function, axes, coordinates, L, second_order)
    if fno_graph.grid_embedding:
        bounds = append_grid(bounds, coordinates)
    return bounds

def validate(model, resolution, lengths):
    from lean_no.adapters.fno import build_graph

    graph = build_graph(model, normalization=True)
    if (
        graph.dimension not in (1, 2)
        or len(resolution) != graph.dimension
        or len(lengths) != graph.dimension
    ):
        raise ValueError("quadrature geometry must match the graph dimension")
    if any(not math.isfinite(length) or length <= 0 for length in lengths) or any(
        type(n) is not int or n < 1 for n in resolution
    ):
        raise ValueError("positive finite quadrature geometry required")
    if graph.dimension == 2 and any(n % 2 for n in resolution):
        raise ValueError("two-dimensional source grids must have even resolutions")
    for op in graph.ops:
        if op.kind == "spectral" and any(
            float(modes.abs().max()) >= n / 2
            for modes, n in zip(op.data["modes"], resolution)
        ):
            raise ValueError("active modes must exclude Nyquist frequencies")
