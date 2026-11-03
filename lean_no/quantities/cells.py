from __future__ import annotations

import math

import torch

from lean_no.bounds.partials import field_parts
from lean_no.function_space.fixed import (
    fixed_input,
)
from lean_no.function_space.input import InputBounds
from lean_no.propagation.linear import Bounds, PointBounds
from lean_no.quantities.norms import ResidualBounds

@torch.no_grad()
def evaluate_cells(centers, widths, evaluate, parent_upper=None, return_floor=False):
    form = evaluate(centers, widths)
    upper = torch.nan_to_num(
        form.actual.magnitude().flatten(), nan=torch.inf, posinf=torch.inf
    )
    selected = (
        upper if parent_upper is None else torch.minimum(upper, parent_upper.to(upper))
    )
    score = torch.nan_to_num(form.center.abs().flatten(), nan=0.0, posinf=0.0)
    count = min(8, len(upper))
    ids = torch.cat((selected.topk(count).indices, score.topk(count).indices)).unique()
    points = centers[ids]
    interval = evaluate(points, torch.zeros_like(points)).actual
    lower = torch.maximum(interval.lower, -interval.upper).clamp_min(0).flatten()
    best = int(lower.argmax())
    witness = dict(
        absolute_lower=float(lower[best]),
        point=points[best].tolist(),
        interval=[
            float(interval.lower.flatten()[best]),
            float(interval.upper.flatten()[best]),
        ],
    )
    if return_floor:
        floor = torch.maximum(form.actual.lower, -form.actual.upper).clamp_min(0)
        return upper.cpu(), witness, floor.flatten().cpu()
    return upper.cpu(), witness

def cell_norms(lower, upper, volume):
    return residual_norm(lower, upper), residual_l2(lower, upper, volume)

def input_cells(
    function: InputBounds, output: Bounds
) -> tuple[torch.Tensor, torch.Tensor]:
    coordinates = (
        output.axes[0]
        if output.dimension == 1
        else torch.stack(torch.meshgrid(*output.axes, indexing="ij"))
    )
    direct = function.derivative_bounds(
        coordinates,
        output.lengths[0] if output.dimension == 1 else output.lengths,
        0 if output.dimension == 1 else (0, 0),
    )
    if direct is not None:
        return direct[2], direct[3]
    coordinates = coordinates.unsqueeze(0) if output.dimension == 1 else coordinates
    bounds = Bounds.from_function(
        function,
        output.axes,
        coordinates,
        output.lengths,
    )
    return bounds.cell_bounds

def residual_norm(
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> float:
    magnitude = torch.maximum(lower.abs(), upper.abs())
    return magnitude.amax().item()

def residual_l2(lower, upper, length):
    peak = residual_norm(lower, upper)
    if peak == 0 or math.isinf(peak):
        return peak
    total = math.fsum(
        float(
            (torch.maximum(low.abs(), high.abs()).double().cpu() / peak).square().sum()
        )
        for low, high in zip(lower.flatten().split(65536), upper.flatten().split(65536))
    )
    return peak * math.sqrt(total / lower.numel() * length)

def boundary_bounds(output, pde):
    if output.dimension == 2:
        if pde != "poisson":
            raise NotImplementedError(
                "2D boundary certification currently supports Poisson"
            )
        lows, highs = [], []
        for normal in range(2):
            tangent = 1 - normal
            for side in (0.0, output.lengths[normal]):
                points = torch.stack(
                    [
                        torch.full_like(output.axes[tangent], side)
                        if axis == normal
                        else output.axes[tangent]
                        for axis in range(2)
                    ]
                )
                low, high = output.bounds(points)
                indices, _ = output.cell(points)
                magnitude = output.partial_magnitude.reshape(output.channels, 2, -1)
                radius = output.steps[tangent] * magnitude[:, tangent, indices]
                lows.append((low - radius).reshape(-1))
                highs.append((high + radius).reshape(-1))
        lower, upper = torch.cat(lows), torch.cat(highs)
        return (
            lower,
            upper,
            torch.maximum(lower.abs(), upper.abs()),
            {
                "coverage": "all four complete edges",
                "segments": lower.numel(),
            },
        )
    points = output.node_lower.new_tensor([0.0, output.lengths[0]])
    indices, _ = output.cell(points)
    boundary = PointBounds(
        points.cpu(),
        *output.bounds(points),
        output.stats,
        output.partial_lower[:, 0, indices],
        output.partial_upper[:, 0, indices],
    )
    lower, upper = boundary.bounds(points)
    lower, upper = lower.reshape(-1), upper.reshape(-1)
    if pde == "helmholtz":
        first_lower = boundary.partial_lower.reshape(-1)
        first_upper = boundary.partial_upper.reshape(-1)
        lower, upper = (
            torch.stack((lower[0] - upper[1], first_lower[0] - first_upper[1])),
            torch.stack((upper[0] - lower[1], first_upper[0] - first_lower[1])),
        )
    return lower, upper, torch.maximum(lower.abs(), upper.abs()), boundary.stats

def burgers_residual(parts, viscosity):
    if not math.isfinite(viscosity) or viscosity < 0:
        raise ValueError("viscosity must be finite and nonnegative")
    value, dx, dt, dxx = (parts[key] for key in ((0, 0), (1, 0), (0, 1), (2, 0)))
    from lean_no.bounds.interval import Interval

    result = (
        Interval(dt.lower, dt.upper)
        + Interval(value.lower, value.upper) * Interval(dx.lower, dx.upper)
        + Interval(dxx.lower, dxx.upper) * (-viscosity)
    )
    return result.lower, result.upper

def finish_bounds(
    pde,
    lower,
    upper,
    boundary,
    lengths,
    kappa=None,
):
    if not bool(
        (torch.isfinite(lower) & torch.isfinite(upper) & (lower <= upper)).all()
    ):
        raise FloatingPointError(
            "invalid residual enclosure; refine quadrature or increase precision"
        )
    linf, l2 = cell_norms(lower, upper, math.prod(lengths))
    boundary_lower, boundary_upper = boundary
    from lean_no.enclosures.cell import CellEnclosure
    from lean_no.quantities.solution import linf_interval

    cells = CellEnclosure(lower, upper, tuple(lengths))
    return ResidualBounds(
        pde,
        linf_interval(lower, upper),
        linf_interval(boundary_lower, boundary_upper),
        tuple(lengths),
        dict(
            residual=dict(complete=True, disjoint=True, leaf_count=lower.numel()),
            boundary=dict(complete=True),
        ),
        kappa=kappa,
        boundary_lower=boundary_lower,
        boundary_upper=boundary_upper,
        residual_l2=l2,
        cells=cells,
    )

def residual_norms(output, function, quantity, grid):
    pde = quantity.pde
    kappa = quantity.coefficient
    lengths = grid.lengths
    parts = field_parts(output)
    input_lower, input_upper = input_cells(function, output)
    bl, bu, _, _ = boundary_bounds(output, pde)
    value = parts[(0,) * len(lengths)]
    if value.lower.shape[0] != 1:
        raise NotImplementedError("residual certification requires one output channel")
    from lean_no.bounds.interval import Interval

    result = quantity.evaluate(
        {order: Interval(part.lower, part.upper) for order, part in parts.items()},
        Interval(input_lower, input_upper),
    )
    lower, upper = result.lower, result.upper
    return finish_bounds(
        pde,
        lower,
        upper,
        (bl, bu),
        lengths,
        kappa=kappa if pde == "helmholtz" else None,
    )

@torch.no_grad()
def burgers_norms(output, initial, grid, viscosity=0.01):
    if not math.isfinite(viscosity) or viscosity < 0:
        raise ValueError("viscosity must be finite and nonnegative")
    resolution, lengths = grid.resolution, grid.lengths
    if len(lengths) != 2 or lengths[0] != 1 or lengths[1] <= 0:
        raise ValueError("unit spatial length and positive time period required")
    parts = field_parts(output)
    lower, upper = burgers_residual(parts, viscosity)
    source = fixed_input(initial, resolution, lengths, lower.device, lower.dtype)
    value, first = parts[(0, 0)], parts[(1, 0)]
    # Restrict to t=0 nodes, then integrate the spatial defect derivative
    # across each entire x cell. No temporal variation enters this trace.
    a, ax = source[(0, 0)], source[(1, 0)]
    dx = 1 / resolution[0]
    il = value.node_lower[0, :, 0] - a.node_upper[0, :, 0]
    iu = value.node_upper[0, :, 0] - a.node_lower[0, :, 0]
    il = il + (dx * (first.lower[0, :, 0] - ax.upper[0, :, 0])).clamp(max=0)
    iu = iu + (dx * (first.upper[0, :, 0] - ax.lower[0, :, 0])).clamp(min=0)
    zeros = lower.new_zeros(2 * resolution[1])
    bl, bu = torch.cat((zeros, il)), torch.cat((zeros, iu))
    return finish_bounds(
        "burgers",
        lower,
        upper,
        (bl, bu),
        lengths,
    )
