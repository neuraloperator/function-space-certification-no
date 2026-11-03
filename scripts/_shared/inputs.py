from __future__ import annotations

import math
from dataclasses import dataclass

import torch

@dataclass(frozen=True)
class WitnessSearch:
    length: float
    eps: float
    full_resolution: int
    grid_resolution: int
    spatial_dimension: int = 1
    time_samples: int = 0
    time_period: float = 1.0
    fourier_modes: int = 16
    mc_samples: int = 4096
    mc_batch_size: int = 256
    mc_seed: int = 0
    decay_mode: float | None = 4.0

def center_values(case, grid, coefficients=((0.0, 1.0), (0.5, 0.0))):
    if case.spatial_dimension == 2:
        x, y = grid
        return torch.sin(math.pi * x) * torch.sin(math.pi * y) + 0.5 * torch.sin(
            2 * math.pi * x
        ) * torch.sin(2 * math.pi * y)
    return sum(
        cosine * torch.cos(2 * math.pi * mode * grid / case.length)
        + sine * torch.sin(2 * math.pi * mode * grid / case.length)
        for mode, (cosine, sine) in enumerate(coefficients, 1)
    )

def input_limits(case, coefficients=((0.0, 1.0), (0.5, 0.0))):
    limits = [
        sum(
            (abs(c) + abs(s)) * (2 * math.pi * mode / case.length) ** order
            for mode, (c, s) in enumerate(coefficients, 1)
        )
        for order in (1, 2)
    ]
    maximum_mode = case.fourier_modes if case.decay_mode is None else case.decay_mode
    limits[0] += 2 * math.pi * maximum_mode * case.eps / case.length
    limits[1] += (
        (2 * math.pi / case.length) ** 2 * maximum_mode * case.fourier_modes * case.eps
    )
    return [limits[0]], [limits[1]]

def basis_terms(case):
    axis = [(0, "cos")] + [
        (m, kind) for kind in ("cos", "sin") for m in range(1, case.fourier_modes + 1)
    ]
    if case.spatial_dimension == 1:
        return [(item,) for item in axis]
    return [(left, right) for left in axis for right in axis]

def basis_term(case, term, coordinates):
    result = 1.0
    for (mode, kind), x in zip(term, coordinates):
        scale = (
            1
            if case.decay_mode is None or mode == 0
            else min(1, case.decay_mode / mode)
        )
        result = (
            result * scale * getattr(torch, kind)(2 * math.pi * mode * x / case.length)
        )
    return result

def witness_input(case, coefficients=None):
    terms = basis_terms(case)
    values = None
    if coefficients is not None:
        coefficients = coefficients.detach().cpu().double()
        magnitude = float(coefficients.abs().sum())
        if len(coefficients) != len(terms) or magnitude > 1 + 1e-12:
            raise ValueError("witness must belong to the Fourier coefficient L1 ball")
        if magnitude > 1:
            coefficients = coefficients / (magnitude * (1 + 1e-14))
        values = coefficients.tolist()

    def evaluate(coordinates):
        result = center_values(
            case, coordinates[0] if len(coordinates) == 1 else coordinates
        )
        if values is not None:
            for value, term in zip(values, terms):
                if value:
                    result = result + case.eps * value * basis_term(
                        case, term, coordinates
                    )
        return result

    if case.spatial_dimension == 1:

        def function(x):
            return evaluate((x,))
    else:

        def function(x, y):
            return evaluate((x, y))

    return function

def basis_values(case, grid):
    coordinates = (grid,) if case.spatial_dimension == 1 else grid
    return torch.stack(
        [case.eps * basis_term(case, term, coordinates) for term in basis_terms(case)]
    )
