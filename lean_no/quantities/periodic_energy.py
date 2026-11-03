import math

import torch

def potential_matrix(potential, viscosity, modes):
    if (
        potential.ndim != 2
        or not potential.shape[0]
        or potential.shape[1] <= 4 * modes
        or type(modes) is not int
        or modes < 0
        or not math.isfinite(viscosity)
        or viscosity <= 0
        or not bool(torch.isfinite(potential).all())
    ):
        raise ValueError(
            "finite slab potentials, positive viscosity, and a resolved mode cutoff required"
        )
    n = potential.shape[-1]
    k = torch.arange(-modes, modes + 1, device=potential.device)
    difference = k[:, None] - k[None, :]
    frequency = difference.to(potential.dtype)
    coefficients = torch.fft.fft(potential, dim=-1, norm="forward")
    # Integral over [j/n,(j+1)/n]: sinc(k/n)*exp(-2*pi*i*k*(j+.5)/n).
    cell_factor = torch.sinc(frequency / n) * torch.exp(-1j * math.pi * frequency / n)
    matrix = coefficients[:, difference % n] * cell_factor
    matrix = matrix - torch.diag(viscosity * (2 * math.pi * k.to(potential.dtype)) ** 2)
    return (matrix + matrix.mH) * 0.5

def largest_eigenvalue_bounds(matrix):
    _, basis = torch.linalg.eigh(matrix)
    gram = basis.mH @ basis
    transformed = basis.mH @ matrix @ basis
    identity = torch.eye(matrix.shape[-1], device=matrix.device, dtype=matrix.dtype)
    defect = (gram - identity).abs().sum(-1).amax(-1)
    if not bool((defect < 1).all()):
        raise FloatingPointError("eigenvector basis could not be verified invertible")
    diagonal = transformed.diagonal(dim1=-2, dim2=-1).real
    radius = (transformed - torch.diag_embed(diagonal)).abs().sum(-1)
    b = (diagonal + radius).amax(-1)
    upper = b / torch.where(b >= 0, 1 - defect, 1 + defect)
    lower = (diagonal / gram.diagonal(dim1=-2, dim2=-1).real).amax(-1)
    return lower, upper

def periodic_energy_bounds(lower, upper, viscosity, modes=64):
    if lower.shape != upper.shape or not bool((lower <= upper).all()):
        raise ValueError("ordered matching potential enclosures required")
    low_matrix = potential_matrix(lower, viscosity, modes)
    high_matrix = potential_matrix(upper, viscosity, modes)
    lo, _ = largest_eigenvalue_bounds(low_matrix)
    _, a = largest_eigenvalue_bounds(high_matrix)
    maximum, minimum = upper.amax(-1), upper.amin(-1)
    c = (maximum - minimum) / 2
    d = maximum - viscosity * (2 * math.pi * (modes + 1)) ** 2
    hi = (a + d + torch.hypot(a - d, 2 * c)) / 2
    hi = torch.minimum(hi, maximum)
    return torch.stack((lo, hi), -1), {
        "spatial_cells": lower.shape[-1],
        "mode_cutoff": modes,
        "low_block_dimension": 2 * modes + 1,
        "maximum_potential_interval_width": float((upper - lower).max()),
        "maximum_tail_correction": float((hi - a).clamp_min(0).max()),
        "all_omitted_modes_bounded": True,
        "zero_mean_assumed": False,
    }
