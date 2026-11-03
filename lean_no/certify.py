import math

import torch

from lean_no.certificate import Certificate
from lean_no.domain import DiscreteGrid, FourierGrid
from lean_no.function_space.core import (
    quadrature_crown,
    quadrature_ibp,
)
from lean_no.function_space.input import FourierInput
from lean_no.methods import (
    Method,
    grid_crown_method,
    grid_ibp_method,
)

@torch.no_grad()
def certify(
    model,
    grid,
    inputs,
    *,
    method=Method("ibp", "gaussian"),
    derivatives=0,
    chunk_size=64,
    moment_order=None,
    moment_batch=8192,
    progress=None,
    strip_radii=None,
    strip_shape=None,
    query=None,
    output_cells=None,
    integration_options=None,
):

    from itertools import product

    from lean_no.adapters.sfno import SFNOInput
    from lean_no.enclosures.bilinear import BilinearBand
    from lean_no.enclosures.grid import GridBounds
    from lean_no.function_space.input import InputBounds
    from lean_no.function_space.spherical import SphereGrid
    from lean_no.inputs import InputSet

    input_set = inputs if isinstance(inputs, InputSet) else InputSet(inputs)
    epsilon = input_set.epsilon
    if not isinstance(method, Method):
        raise TypeError("method must be a Method")
    method.validate()
    propagation, integration, strategy = (
        method.propagation,
        method.integration,
        method.strategy,
    )
    dimension = (
        len(grid.resolution)
        if isinstance(grid, (FourierGrid, DiscreteGrid))
        else 2
        if isinstance(grid, SphereGrid)
        else 1
    )
    if type(derivatives) is int:
        if not 0 <= derivatives <= 4:
            raise ValueError("derivatives must be between zero and four")
        derivatives = tuple(
            order
            for order in product(range(derivatives + 1), repeat=dimension)
            if 0 < sum(order) <= derivatives
        )
    else:
        derivatives = tuple(derivatives)
        if any(
            len(order) != dimension or any(type(n) is not int or n < 0 for n in order)
            for order in derivatives
        ):
            raise ValueError("derivative multi-indices must match the grid")
    derivative_method = (
        isinstance(grid, FourierGrid)
        and propagation == "ibp"
        and strategy in ("general", "taylor_cell")
        and integration != "taylor_quadrature"
    )
    if derivatives:
        if query is not None:
            raise ValueError("derivative certification requires a complete field")
        if (
            propagation != "ibp"
            or strategy == "linearized"
            or isinstance(grid, DiscreteGrid)
        ):
            raise ValueError("spatial derivatives require forward function-space IBP")
        if integration == "taylor_quadrature" and max(map(sum, derivatives)) > 2:
            raise ValueError(
                "Taylor quadrature requires derivatives of order at most two"
            )
    if moment_order is not None and not derivative_method:
        raise ValueError(
            "moment_order requires function-space IBP coefficient integration"
        )
    linear_methods = (
        Method("ibp", "taylor_quadrature"),
        *(
            Method("crown", rule)
            for rule in ("rectangle", "gaussian", "clenshaw_curtis", "arb")
        ),
    )
    if output_cells is not None:
        crown_cover = (
            propagation == "crown"
            and integration in ("rectangle", "gaussian", "clenshaw_curtis", "arb")
            and isinstance(grid, FourierGrid)
            and len(grid.resolution) == 1
            and (
                (type(output_cells) is int and output_cells <= grid.resolution[0])
                or isinstance(output_cells, torch.Tensor)
            )
        )
        if (
            not crown_cover
            or query is not None
            or (
                not isinstance(output_cells, torch.Tensor)
                and (type(output_cells) is not int or output_cells < 1)
            )
        ):
            raise ValueError(
                "output_cells requires a valid one-dimensional CROWN cover"
            )
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    vector_epsilon = strategy == "linearized" and isinstance(
        epsilon, (torch.Tensor, tuple, list)
    )
    if vector_epsilon:
        values = torch.as_tensor(epsilon)
        if values.ndim != 1 or not torch.isfinite(values).all() or (values < 0).any():
            raise ValueError("epsilon must contain finite nonnegative channel bounds")
    if (
        not callable(epsilon)
        and not vector_epsilon
        and (
            not isinstance(epsilon, (int, float))
            or not math.isfinite(epsilon)
            or epsilon < 0
        )
    ):
        raise ValueError("epsilon must be finite and nonnegative")
    if query is not None:
        if not isinstance(grid, FourierGrid) or (
            method not in linear_methods and not derivative_method
        ):
            raise ValueError(
                "query optimization requires Fourier Linear IBP or CROWN output/robustness"
            )
        if (
            not isinstance(query, torch.Tensor)
            or query.ndim != 2
            or query.shape != (len(query), len(grid.resolution))
            or len(query) == 0
            or query.is_complex()
            or not torch.isfinite(query).all()
        ):
            raise ValueError("query must contain finite real [N, dimension] points")
        lengths = query.new_tensor(grid.lengths)
        if ((query < 0) | (query > lengths)).any():
            raise ValueError("query lies outside the domain")
    if isinstance(grid, (FourierGrid, DiscreteGrid)):
        from lean_no.adapters.fno import build_graph

        if derivatives and getattr(model, "training", False):
            raise ValueError("derivative certification requires an eval-mode model")
        model = build_graph(model, getattr(grid, "dtype", None), normalization=True)
    if isinstance(grid, FourierGrid) and (
        strategy != "general" or integration == "taylor_quadrature"
    ):
        if any(op.kind == "normalize" for op in model.ops):
            raise NotImplementedError(
                "this construction has no native-statistic normalization rule"
            )
    inputs = input_set.prepare(model, grid, method)
    fixed = input_set.is_exact
    label = "output" if fixed else "robustness"
    if isinstance(grid, SphereGrid):
        from lean_no.propagation.spherical import spherical_bounds

        if not isinstance(inputs, SFNOInput):
            raise TypeError("pair SphereGrid with SFNOInput")
        if (
            propagation != "ibp"
            or strategy != "general"
            or derivatives
            or integration
            not in (
                "rectangle",
                "gaussian",
                "clenshaw_curtis",
                "arb",
            )
        ):
            raise ValueError(
                "SFNO requires general IBP without spatial derivative requests"
            )
        if grid.kind != "clenshaw-curtis":
            raise ValueError(
                "Clenshaw-Curtis certification requires a Clenshaw-Curtis grid"
            )
        if moment_order is not None or moment_batch != 8192:
            raise ValueError("moment options require Gaussian derivative fields")
        from lean_no.integration import Integration

        field, diagnostics = spherical_bounds(
            model,
            grid,
            inputs,
            chunk_size=chunk_size,
            progress=progress,
            strip_radii=strip_radii,
            strip_shape=(256, 512) if strip_shape is None else strip_shape,
            rule=Integration(integration, integration_options),
        )
        return Certificate(
            field,
            grid,
            input_set,
            method,
            {"layers": diagnostics},
            quantity=label,
            pointwise_width=field.channel_width,
        )
    if strip_radii is not None or strip_shape is not None:
        raise ValueError("strip options require SFNO")
    if derivative_method:
        from lean_no.integration import Integration
        from lean_no.propagation.derivatives import derivative_field

        field = derivative_field(
            model,
            grid,
            inputs,
            moment_order,
            moment_batch,
            progress,
            integration=integration,
            integration_options=integration_options,
            strategy=strategy,
            orders=derivatives,
        )
        query_width = cell_range_width = None
        if query is not None:
            query_width = field.query(query).width.amax(0)
        elif not derivatives:
            lower, upper = field.cell_bounds
            cell_range_width = (upper - lower).flatten(1).amax(1)
        return Certificate(
            field,
            grid,
            input_set,
            method,
            {"coefficients": field.records},
            quantity=label,
            derivatives=derivatives,
            query_width=query_width,
            cell_range_width=cell_range_width,
        )
    if (
        moment_order is not None
        or moment_batch != 8192
        or (
            progress is not None
            and method != grid_crown_method
            and not (method in linear_methods and propagation == "crown")
            and not (strategy == "linearized" and integration is not None)
        )
    ):
        raise ValueError(
            "moment and progress options require Gaussian derivative fields"
        )
    if isinstance(grid, DiscreteGrid):
        from lean_no.function_space.grid.crown import grid_crown
        from lean_no.function_space.grid.ibp import grid_ibp

        if not isinstance(inputs, GridBounds):
            raise TypeError("pair DiscreteGrid with GridBounds")
        if method not in (grid_ibp_method, grid_crown_method):
            raise ValueError(
                "discrete grids require IBP/CROWN without numerical integration"
            )
        if inputs.lower.shape[2:] != grid.resolution:
            raise ValueError("input shape differs from discrete grid resolution")
        if method == grid_ibp_method:
            lower, upper = grid_ibp(model, inputs.lower, inputs.upper)
        else:
            lower, upper = grid_crown(
                model,
                inputs.lower,
                inputs.upper,
                output_chunk_size=chunk_size,
                progress=progress,
            )
        enclosure = GridBounds(lower, upper)
        return Certificate(
            enclosure,
            grid,
            input_set,
            method,
            {},
            quantity=label,
            node_width=enclosure.channel_width,
        )
    if method in (grid_ibp_method, grid_crown_method):
        raise ValueError("grid methods require DiscreteGrid")
    if method in linear_methods:
        if isinstance(inputs, FourierInput):
            inputs = (
                inputs.nominal
                if fixed and (query is not None or derivatives)
                else inputs.bounds()
            )
        from lean_no.function_space.analytic import Series

        if not isinstance(inputs, (InputBounds, Series)):
            raise TypeError("Linear IBP/CROWN requires InputBounds or FourierInput")
        from lean_no.adapters.fno import build_graph

        if not isinstance(grid, FourierGrid):
            raise ValueError("Linear IBP and CROWN require rectangular geometry")
        points = None
        if query is not None:
            reference = model.reference
            points = query.to(device=reference.device, dtype=grid.dtype)
            points = (
                points[:, 0] if len(grid.resolution) == 1 else points.T.contiguous()
            )
        operation = quadrature_crown if propagation == "crown" else quadrature_ibp
        result = operation(
            model,
            grid,
            inputs,
            chunk_size=chunk_size,
            query=points,
            integration=method.integration,
            integration_options=integration_options,
            **({"derivatives": derivatives} if operation is quadrature_ibp else {}),
            **(
                {"output_cells": output_cells, "progress": progress}
                if operation is quadrature_crown
                else {}
            ),
        )
        lower, upper = (
            result.cell_bounds if query is None else (result.low, result.high)
        )
        return Certificate(
            result,
            grid,
            input_set,
            method,
            {"anchors": result.evaluated} if output_cells is not None else {},
            quantity=label,
            derivatives=derivatives,
            cell_range_width=(upper - lower).flatten(1).amax(1)
            if query is None
            else None,
            query_width=(upper - lower).flatten(1).amax(1)
            if query is not None
            else None,
        )
    if strategy == "linearized":
        from lean_no.adapters.fno import build_graph
        from lean_no.enclosures.bilinear import BilinearBand
        from lean_no.propagation.perturbation import perturbation_bounds

        if not isinstance(grid, FourierGrid) or len(grid.resolution) not in (1, 2, 3):
            raise ValueError(
                "linearization requires a one-, two-, or three-dimensional Fourier grid"
            )
        if len(grid.lengths) != len(grid.resolution) or any(
            not math.isfinite(x) or x <= 0 for x in grid.lengths
        ):
            raise ValueError("linearization requires positive finite domain lengths")
        if not isinstance(inputs, BilinearBand):
            raise TypeError("linearization requires a nominal BilinearBand")
        if tuple(inputs.nominal.lower.shape[2:]) != grid.resolution:
            raise ValueError("nominal and certificate resolutions differ")
        if progress is not None:
            integration_options = {**(integration_options or {}), "progress": progress}
        result, diagnostics = perturbation_bounds(
            build_graph(model, grid.dtype),
            inputs.nominal,
            inputs.epsilon,
            channel_chunk=chunk_size,
            constant_in_time=inputs.constant_in_time,
            integration=integration,
            integration_options=integration_options,
        )
        from dataclasses import replace

        result = replace(result, lengths=grid.lengths)
        lower, upper = result.cell_bounds
        return Certificate(
            result,
            grid,
            input_set,
            method,
            diagnostics,
            quantity=label,
            pointwise_width=result.channel_width,
            cell_range_width=(upper - lower).flatten(1).amax(1),
        )
    raise ValueError("unsupported propagation/integration pair for this geometry")
