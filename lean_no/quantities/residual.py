import math
import time

import torch

from lean_no.refine import audit_cover

def residual_form(field, centers, halfwidth, quantity, forcing=None):
    from lean_no.spectral.fourier import spectral_forms

    parts = dict(zip(field.orders, field.enclose(centers, halfwidth)))
    if quantity.pde in ("burgers", "ks"):
        return quantity.evaluate(parts)
    source = spectral_forms(
        field.forcing if forcing is None else forcing,
        centers,
        halfwidth,
        ((0,) * centers.shape[1],),
        field.periods,
    )[0]
    return quantity.evaluate(
        parts, source.actual if field.arithmetic == "interval" else source
    )

def initial_error_form(field, centers, halfwidth):
    from lean_no.propagation.derivative_cells import orders_2d
    from lean_no.spectral.fourier import spectral_forms

    value = field.enclose(centers, halfwidth, values_only=True)[0]
    initial = spectral_forms(
        field.forcing,
        centers,
        halfwidth,
        orders=orders_2d[:1],
    )[0]
    return value + initial * (-1)

@torch.no_grad()
def burgers_norm(
    field,
    spatial_cells,
    time_cells,
    viscosity=0.01,
    time_horizon=1.0,
    modes=128,
    batch=8192,
    progress=print,
):
    from lean_no.quantities.norms import BurgersStability, ResidualBounds
    from lean_no.quantities.periodic_energy import (
        periodic_energy_bounds,
    )
    from lean_no.quantities.solution import linf_interval

    if (
        field.time_period is None
        or type(modes) is not int
        or modes < 0
        or any(type(v) is not int or v < 1 for v in (spatial_cells, time_cells, batch))
        or spatial_cells <= 4 * modes
        or viscosity <= 0
        or not math.isfinite(viscosity)
        or not 0 < time_horizon <= field.time_period
    ):
        raise ValueError(
            "resolved positive grid, viscosity, and valid horizon required"
        )
    ref = field.forcing.error
    initial_parts, initial_lower, initial_upper = [], [], []
    for begin in range(0, spatial_cells, batch):
        x = (
            torch.arange(
                begin, min(begin + batch, spatial_cells), device=ref.device
            ).to(ref)
            + 0.5
        ) / spatial_cells
        centers = torch.stack((x, torch.zeros_like(x)), -1)
        widths = torch.zeros_like(centers)
        widths[:, 0] = 0.5 / spatial_cells
        error = initial_error_form(field, centers, widths)
        initial_parts.append(
            error.spatial_square_polynomial().sum(0).cpu() / spatial_cells
        )
        interval = error.lane(1).interval()
        initial_lower.append(interval.lower.cpu())
        initial_upper.append(interval.upper.cpu())
    initial = linf_interval(torch.cat(initial_lower), torch.cat(initial_upper))
    average = torch.stack(initial_parts).sum(0)
    initial_l2 = (
        (average[0] + average.new_tensor([-1, 1]) * average[3]).clamp_min(0).sqrt()
    ).tolist()

    edges = ref.new_tensor(
        [j * time_horizon / time_cells for j in range(time_cells + 1)]
    )
    sums = ref.new_zeros(time_cells, 4)
    rlo, rhi = (ref.new_zeros(time_cells) for _ in range(2))
    lower_potential, upper_potential = [], []
    for begin in range(0, spatial_cells * time_cells, batch):
        ids = torch.arange(
            begin, min(begin + batch, spatial_cells * time_cells), device=ref.device
        )
        ti, xi = ids // spatial_cells, ids % spatial_cells
        centers = torch.stack(
            (
                (xi.to(ref) + 0.5) / spatial_cells,
                (edges[ti] + edges[ti + 1]) / (2 * field.time_period),
            ),
            -1,
        )
        widths = torch.stack(
            (
                torch.full_like(centers[:, 0], 0.5 / spatial_cells),
                (edges[ti + 1] - edges[ti]) / (2 * field.time_period),
            ),
            -1,
        )
        v, vx, vs, vxx, _, _ = field.enclose(centers, widths)
        residual = vs + v * vx + vxx * (-viscosity)
        sums.index_add_(0, ti, residual.spatial_square_polynomial() / spatial_cells)
        r, slope = residual.lane(1).interval(), vx.lane(1).interval()
        rhi.scatter_reduce_(
            0, ti, r.magnitude()[:, 0], reduce="amax", include_self=True
        )
        rlo.scatter_reduce_(
            0,
            ti,
            (residual.c[:, 0].abs() - residual.r[:, 0, 1]).clamp_min(0),
            reduce="amax",
            include_self=True,
        )
        lower_potential.append((-slope.upper[:, 0] / 2).cpu())
        upper_potential.append((-slope.lower[:, 0] / 2).cpu())
    # The squared affine time coefficient is nonnegative: the maximum is
    # attained at an endpoint of the complete time slab.
    half_time = torch.diff(edges) / (2 * field.time_period)
    peak = sums[:, 0] + sums[:, 1].abs() * half_time + sums[:, 2] * half_time.square()
    residual_l2 = torch.stack(
        (
            (peak - sums[:, 3]).clamp_min(0).sqrt(),
            (peak + sums[:, 3]).clamp_min(0).sqrt(),
        ),
        -1,
    )
    lower, upper = (
        torch.cat(p).reshape(time_cells, spatial_cells)
        for p in (lower_potential, upper_potential)
    )
    growth_parts, records = [], []
    # Bound the entire continuum, while keeping eigensolver memory bounded.
    for begin in range(0, time_cells, 32):
        lo, hi = lower[begin : begin + 32], upper[begin : begin + 32]
        growth, row = periodic_energy_bounds(lo, hi, viscosity, modes)
        growth_parts.append(growth.clamp_min(0))
        records.append(row)
    growth = torch.cat(growth_parts)
    dt = torch.diff(edges).cpu()
    details = {
        **records[0],
        "maximum_potential_interval_width": float((upper - lower).max()),
        "maximum_tail_correction": max(p["maximum_tail_correction"] for p in records),
        "accumulated_exponent_interval": (growth * dt[:, None]).sum(0).tolist(),
    }
    data = BurgersStability(
        viscosity,
        field.time_period,
        tuple(edges.tolist()),
        tuple(initial_l2),
        tuple(map(tuple, residual_l2.tolist())),
        tuple(map(tuple, growth.tolist())),
        True,
        True,
    )
    coverage = {
        "residual": {
            "complete": True,
            "disjoint": True,
            "base_cells": [spatial_cells, time_cells],
            "leaf_count": spatial_cells * time_cells,
            "maximum_depth": 0,
            "spatial_l2": "exact x integrals of affine squares with bounded remainders; whole-slab time maxima",
        },
        "boundary": {
            "complete": True,
            "initial_segments": spatial_cells,
            "spatial_periodicity": "preserved by input and supported FNO operations",
        },
    }
    cert = ResidualBounds(
        "burgers",
        (float(rlo.max()), float(rhi.max())),
        initial,
        (1.0, field.time_period),
        coverage,
        burgers=data,
        residual_l2=max(pair[1] for pair in data.residual_l2),
    )
    if progress is not None:
        progress(
            dict(
                phase="direct L2 and refined energy",
                initial_l2=initial_l2,
                residual_l2_max=residual_l2.max(0).values.tolist(),
                **details,
            )
        )
    return cert, details

@torch.no_grad()
def regular_cells(
    field, base, indices, quantity, boundary=False, batch=4096, forcing=None
):

    ref = field.forcing.error
    upper_parts = []
    floor_parts = []
    witness = {"absolute_lower": 0.0}
    for begin in range(0, len(indices), batch):
        ids = indices[begin : begin + batch].to(ref.device)
        if boundary:
            normal = ids[:, 0] // 2
            side = (ids[:, 0] % 2).to(ref)
            t = (ids[:, 1].to(ref) + 0.5) / base[1]
            radius = torch.full_like(t, 0.5 / base[1])
            centers = torch.stack(
                (torch.where(normal == 0, side, t), torch.where(normal == 1, side, t)),
                -1,
            )
            widths = torch.stack(
                (
                    torch.where(normal == 0, 0.0, radius),
                    torch.where(normal == 1, 0.0, radius),
                ),
                -1,
            )
        else:
            sizes = ref.new_tensor(base)
            centers = (ids.to(ref) + 0.5) / sizes
            widths = (0.5 / sizes).expand_as(centers)
        from lean_no.quantities.cells import evaluate_cells

        def evaluate(points, radii):
            return (
                field.enclose(points, radii, values_only=True)[0]
                if boundary
                else residual_form(field, points, radii, quantity, forcing)
            )

        upper, candidate, floor = evaluate_cells(
            centers, widths, evaluate, return_floor=True
        )
        upper_parts.append(upper)
        floor_parts.append(floor)
        if candidate["absolute_lower"] >= witness["absolute_lower"]:
            witness = candidate
    return torch.cat(upper_parts), witness, torch.cat(floor_parts)

@torch.no_grad()
def interval_norms(field, quantity, grid, batch):
    from lean_no.quantities.cells import finish_bounds

    ref = field.forcing.error
    n = grid.resolution[0]
    lower, upper = [], []
    for begin in range(0, n, batch):
        x = (
            torch.arange(begin, min(begin + batch, n), device=ref.device).to(ref) + 0.5
        ) / n
        centers = x[:, None]
        halfwidth = torch.full_like(centers, 0.5 / n)
        interval = residual_form(field, centers, halfwidth, quantity).actual
        lower.append(interval.lower.T)
        upper.append(interval.upper.T)
    endpoints = ref.new_tensor([[0.0], [1.0]])
    jets = field.enclose(endpoints, torch.zeros_like(endpoints))
    value = jets[0].actual
    if quantity.pde == "poisson":
        boundary = value.lower[:, 0], value.upper[:, 0]
    else:
        derivative = jets[1].actual
        boundary = (
            torch.stack([v.lower[0, 0] - v.upper[1, 0] for v in (value, derivative)]),
            torch.stack([v.upper[0, 0] - v.lower[1, 0] for v in (value, derivative)]),
        )
    return finish_bounds(
        quantity.pde,
        torch.cat(lower, -1),
        torch.cat(upper, -1),
        boundary,
        grid.lengths,
        kappa=quantity.coefficient if quantity.pde == "helmholtz" else None,
    )

def poisson_norm(
    field,
    base_cells,
    quantity,
    boundary=False,
    progress=None,
    batch=4096,
    forcing=None,
    volume=1.0,
):
    from lean_no.quantities.cells import cell_norms

    started = time.perf_counter()
    base = (4, base_cells[0]) if boundary else base_cells
    indices = torch.cartesian_prod(torch.arange(base[0]), torch.arange(base[1]))
    upper, witness, floor = regular_cells(
        field, base, indices, quantity, boundary, batch, forcing
    )
    coverage = audit_cover([dict(level=0, indices=indices, upper=upper)], *base)
    lower = witness["absolute_lower"]
    high, l2 = cell_norms(-upper, upper, volume)
    _, l2_floor = cell_norms(floor, floor, volume)
    high = max(lower, high)
    if progress is not None:
        progress(
            dict(
                phase="boundary" if boundary else "residual",
                lower=lower,
                upper=high,
                coverage=coverage,
                seconds=time.perf_counter() - started,
            )
        )
    return (lower, high), (l2_floor, l2), coverage

@torch.no_grad()
def residual_bound(
    field,
    quantity,
    forcing=None,
    chunk_size=64,
    time_horizon=1.0,
    energy_modes=128,
    progress=None,
    index=1,
    batch=16384,
    enclosure_batch=4096,
    profile_cache=None,
):
    from lean_no.certificate import Certificate
    from lean_no.domain import FourierGrid
    from lean_no.function_space.analytic import describe
    from lean_no.graph import Field
    from lean_no.propagation.derivative_cells import DerivativeField
    from lean_no.propagation.linear import Bounds
    from lean_no.quantities import Residual
    from lean_no.quantities.cells import burgers_norms, residual_norms
    from lean_no.quantities.norms import ResidualBounds, ResidualResult

    if not isinstance(quantity, Residual):
        raise TypeError("residual_bound requires a Residual expression")
    if quantity.coefficient is not None and not math.isfinite(quantity.coefficient):
        raise ValueError("finite PDE coefficient required")
    if quantity.pde == "helmholtz" and (
        quantity.coefficient is None or quantity.coefficient < 0
    ):
        raise ValueError("Helmholtz requires a finite nonnegative wave number")
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("positive cell batch required")
    if isinstance(field, Field):
        from lean_no.quantities.ks import ks_bounds

        if quantity.pde != "ks" or forcing is not None:
            raise ValueError("rollout residuals require the unforced KS expression")
        norms, diagnostics = ks_bounds(
            field, quantity, index, batch, enclosure_batch, profile_cache
        )
        grid = FourierGrid(field.metadata.get("native_shape", (1, 1)), field.periods)
        from lean_no.inputs import InputSet
        from lean_no.methods import Method

        source = Field(field.run, field.metadata.get("input_node", field.node))
        return ResidualResult(
            norms,
            field,
            grid,
            InputSet(source),
            Method("ibp", "gaussian", "taylor_cell"),
            diagnostics=diagnostics,
        )
    if not isinstance(field, Certificate) or not isinstance(field.grid, FourierGrid):
        raise TypeError("a certified Fourier derivative field is required")
    grid, output = field.grid, field.enclosure
    required = set(quantity.required_derivatives(grid)) - {(0,) * len(grid.resolution)}
    if not required.issubset(field.derivatives):
        raise ValueError("field is missing required derivatives")
    channels = (
        output.network.out_channels
        if isinstance(output, DerivativeField)
        else output.channels
    )
    if channels != 1:
        raise ValueError("PDE residuals require one output channel")
    diagnostics = {}
    if isinstance(output, DerivativeField):
        diagnostics["coefficients"] = output.records
        if callable(forcing):
            forcing = describe(forcing).field(
                grid.lengths, output.forcing.error.device, grid.dtype
            )
        if len(grid.resolution) == 1:
            if quantity.pde not in ("poisson", "helmholtz") or forcing is not None:
                raise ValueError(
                    "one-dimensional residuals require their certified elliptic forcing"
                )
            norms = interval_norms(output, quantity, grid, chunk_size)
        elif quantity.pde == "burgers":
            norms, energy = burgers_norm(
                output,
                *grid.resolution,
                viscosity=0.01
                if quantity.coefficient is None
                else quantity.coefficient,
                time_horizon=time_horizon,
                modes=energy_modes,
                batch=chunk_size,
                progress=progress,
            )
            diagnostics["energy_stability"] = energy
        elif quantity.pde == "poisson":
            residual, l2, interior = poisson_norm(
                output,
                grid.resolution,
                quantity,
                progress=progress,
                batch=chunk_size,
                forcing=forcing,
                volume=math.prod(grid.lengths),
            )
            boundary, _, edges = poisson_norm(
                output,
                grid.resolution,
                quantity,
                boundary=True,
                progress=progress,
                batch=chunk_size,
                volume=2 * sum(grid.lengths),
            )
            norms = ResidualBounds(
                "poisson",
                residual,
                boundary,
                grid.lengths,
                dict(residual=interior, boundary=edges),
                residual_l2=l2[1],
                residual_l2_lower=l2[0],
            )
        else:
            raise ValueError("unsupported equation for this derivative field")
    elif isinstance(output, Bounds):
        from lean_no.function_space.analytic import Series
        from lean_no.inputs import InputSet

        source = field.input_bounds if forcing is None else forcing
        if isinstance(source, InputSet):
            source = source.nominal
        if isinstance(source, Series) and quantity.pde != "burgers":
            source = source.bounds()
        if callable(source):
            source = describe(source).bounds()
        if quantity.pde == "burgers":
            norms = burgers_norms(
                output,
                source,
                grid,
                0.01 if quantity.coefficient is None else quantity.coefficient,
            )
        else:
            norms = residual_norms(
                output,
                source,
                quantity,
                grid,
            )
    else:
        raise TypeError("field does not provide residual derivative enclosures")
    return ResidualResult(
        norms, output, grid, field.input_bounds, field.method, forcing, diagnostics
    )
