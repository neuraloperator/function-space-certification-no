from __future__ import annotations

import torch

from lean_no.adapters.fno import build_graph
from lean_no.domain import FourierGrid
from lean_no.integration import Integration
from lean_no.propagation import crown as crown_backend
from lean_no.propagation import linear as ibp_backend

@torch.no_grad()
def quadrature_ibp(
    model,
    grid,
    inputs,
    derivatives=(),
    integration="gaussian",
    chunk_size=64,
    query=None,
    integration_options=None,
):
    if not isinstance(grid, FourierGrid):
        raise TypeError("quadrature IBP requires FourierGrid geometry")
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if callable(inputs):
        from lean_no.inputs import InputSet
        from lean_no.methods import Method

        inputs = InputSet(inputs).prepare(model, grid, Method("ibp", integration))
    if integration != "taylor_quadrature":
        from lean_no.propagation.derivatives import derivative_field

        return derivative_field(
            model,
            grid,
            inputs,
            None,
            8192,
            None,
            integration=integration,
            integration_options=integration_options,
            orders=derivatives,
        )
    from lean_no.function_space.analytic import Series
    from lean_no.propagation import taylor_quadrature

    if any(
        len(order) != len(grid.resolution) or sum(order) > 2 or min(order) < 0
        for order in derivatives
    ):
        raise ValueError(
            "Taylor quadrature derivatives must have order at most two and match the grid"
        )
    if isinstance(inputs, Series):
        from lean_no.function_space.fixed import paper_input, validate_fixed

        network = validate_fixed(
            model, grid.resolution, grid.lengths, integration, grid.dtype
        )
        source = paper_input(
            inputs, grid.resolution, grid.lengths, network.device, network.dtype
        )
        bounds = taylor_quadrature.propagate(network, source)
    else:
        bounds = taylor_quadrature.graph(
            model, inputs, grid.resolution, grid.lengths, grid.dtype
        )
    if query is None:
        return bounds
    indices, _ = bounds.cell(query)
    partials = [
        value.reshape(bounds.channels, bounds.dimension, -1)[:, :, indices]
        for value in (bounds.partial_lower, bounds.partial_upper)
    ]
    if bounds.dimension == 1:
        partials = [value[:, 0] for value in partials]
    return ibp_backend.PointBounds(
        query.detach().cpu().clone(), *bounds.bounds(query), bounds.stats, *partials
    )

@torch.no_grad()
def quadrature_crown(
    model,
    grid,
    inputs,
    chunk_size=64,
    query=None,
    integration="gaussian",
    output_cells=None,
    progress=None,
    integration_options=None,
):
    if not isinstance(grid, FourierGrid):
        raise TypeError("quadrature CROWN requires FourierGrid geometry")
    if integration not in ("rectangle", "gaussian", "clenshaw_curtis", "arb"):
        raise ValueError(f"unknown CROWN integration: {integration}")
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if callable(inputs):
        from lean_no.inputs import InputSet
        from lean_no.methods import Method

        inputs = InputSet(inputs).prepare(model, grid, Method("crown", integration))
    crown_backend.validate(model, grid.resolution, grid.lengths)
    if output_cells is not None:
        if query is not None:
            raise ValueError("output_cells requires a complete cover, not query points")
        return crown_backend.fixed_graph(
            build_graph(model, grid.dtype, normalization=True),
            grid.resolution,
            grid.lengths,
            inputs,
            output_cells,
            output_chunk_size=chunk_size,
            progress=progress,
            rule=Integration(integration, integration_options),
        )
    return crown_backend.graph(
        build_graph(model, grid.dtype, normalization=True),
        grid.resolution,
        grid.lengths,
        inputs,
        chunk_size,
        query,
        progress=progress,
        rule=Integration(integration, integration_options),
    )
