import math

import torch

from lean_no.adapters.fno import build_graph
from lean_no.bounds.operators import is_identity, is_tanh
from lean_no.bounds.partials import PartialBounds
from lean_no.propagation import linear as ibp

keys = ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2))

def fixed_input(
    initial, resolution, lengths=(1.0, 1.0), device="cpu", dtype=torch.float64
):
    if (
        len(lengths) != 2
        or lengths[0] != 1
        or not math.isfinite(lengths[1])
        or lengths[1] <= 0
    ):
        raise ValueError(
            "Burgers requires unit space and a positive finite time period"
        )
    axes = [torch.arange(n, device=device, dtype=dtype) / n for n in resolution]
    result = {}
    for orders in keys:
        value = axes[0].new_zeros(2, *resolution)
        lower, upper = value.clone(), value.clone()
        if orders[1] == 0:
            node_low, node_high, cell_low, cell_high = initial.derivative_bounds(
                axes[0], 1.0, orders[0]
            )
            value[0] = node_low[0, :, None]
            lower[0], upper[0] = cell_low[0, :, None], cell_high[0, :, None]
        if orders == (0, 0):
            value[1] = lengths[1] * axes[1][None, :]
            lower[1] = value[1]
            upper[1] = value[1] + lengths[1] / resolution[1]
        elif orders == (0, 1):
            value[1] = 1
            lower[1] = upper[1] = 1
        result[orders] = PartialBounds(value, value, lower, upper)
    return result

def validate_fixed(model, resolution, lengths, method, dtype=None):
    if method != "taylor_quadrature":
        raise ValueError("residual method must be 'taylor'")
    if getattr(model, "training", False):
        raise ValueError("model must be in eval mode")
    ibp.validate(model, resolution, lengths)
    if build_graph(model).grid_embedding:
        raise NotImplementedError(
            "fixed residual bounds require explicit input channels"
        )
    network = build_graph(model, dtype)
    activations = (
        network.lifting.activation,
        network.projection.activation,
        *(block.activation for block in network.blocks if block.activation is not None),
    )
    if any(not (is_tanh(item) or is_identity(item)) for item in activations):
        raise NotImplementedError(
            "fixed residual bounds require smooth tanh or identity"
        )
    return network

def paper_input(
    initial, resolution, lengths=(1.0, 1.0), device="cpu", dtype=torch.float64
):
    parts = fixed_input(initial, resolution, lengths, device, dtype)
    axes, _ = ibp.make_grid(resolution, lengths, device, dtype)
    value = parts[(0, 0)]
    first = [
        torch.stack([getattr(parts[key], name) for key in ((1, 0), (0, 1))], 1)
        for name in ("lower", "upper", "node_lower", "node_upper")
    ]
    second = [
        torch.stack(
            [
                torch.stack([getattr(parts[key], name) for key in row], 1)
                for row in (((2, 0), (1, 1)), ((1, 1), (0, 2)))
            ],
            1,
        )
        for name in ("lower", "upper")
    ]
    return ibp.Bounds(
        axes, lengths, value.node_lower, value.node_upper, *first, *second
    )
