import itertools
import math

import torch

from lean_no.bounds.affine import DualForm

def synthesis(coefficients, modes, points):
    dimension = len(modes)
    if points.ndim != 2 or points.shape[1] != dimension:
        raise ValueError("points must match the Fourier dimension")
    shape = tuple(len(axis) for axis in modes)
    result = coefficients.real.new_empty(*coefficients.shape[:-dimension], len(points))
    weighted = coefficients * torch.where(modes[-1] == 0, 1.0, 2.0)
    for start in range(0, len(points), 4096):
        part = points[start : start + 4096]
        phase = coefficients.new_ones(len(part), *shape)
        for axis, frequency in enumerate(modes):
            layout = [1] * dimension
            layout[axis] = len(frequency)
            phase *= torch.exp(
                2j
                * math.pi
                * part[:, axis].reshape(-1, *([1] * dimension))
                * frequency.reshape(1, *layout)
            )
        result[..., start : start + len(part)] = (
            weighted.flatten(-dimension) @ phase.flatten(1).T
        ).real
    return result.movedim(-1, -2)

def spectral_forms(node, centers, halfwidth, orders, periods=None):
    coefficients, modes = node.data["coefficients"], node.data["modes"]
    dimension = len(modes)
    periods = (1.0,) * dimension if periods is None else tuple(periods)
    if len(periods) != dimension or halfwidth.shape != centers.shape:
        raise ValueError("Fourier cells and periods must match")
    frequencies = []
    factors = coefficients.new_ones(len(orders), *coefficients.shape[1:])
    for axis, frequency in enumerate(modes):
        layout = [1] * dimension
        layout[axis] = len(frequency)
        omega = (2 * math.pi * frequency).reshape(layout)
        frequencies.append(omega)
        factors *= torch.stack(
            [(1j * omega / periods[axis]) ** order[axis] for order in orders]
        )
    weighted = coefficients[None] * factors[:, None]
    center = synthesis(weighted, modes, centers)
    gradient = torch.stack(
        [synthesis(weighted * (1j * omega), modes, centers) for omega in frequencies],
        -1,
    )
    remainder = torch.zeros_like(center)
    for left, right in itertools.product(range(dimension), repeat=2):
        second = synthesis(
            -weighted * frequencies[left] * frequencies[right], modes, centers
        )
        remainder += (
            second.abs() * (halfwidth[:, left] * halfwidth[:, right])[None, :, None] / 2
        )
    symmetry = torch.where(modes[-1] == 0, 1.0, 2.0)
    amplitude = weighted.abs() * symmetry
    mode_axes = tuple(range(-dimension, 0))
    for order in itertools.product(range(4), repeat=dimension):
        if sum(order) != 3:
            continue
        magnitude = amplitude
        widths = torch.ones_like(halfwidth[:, 0])
        divisor = 1
        for axis, power in enumerate(order):
            magnitude = magnitude * frequencies[axis].abs() ** power
            widths *= halfwidth[:, axis] ** power
            divisor *= math.factorial(power)
        remainder += magnitude.sum(mode_axes)[:, None] * widths[None, :, None] / divisor
    error = node.data.get("coefficient_error")
    if error is None:
        if bool((node.error != 0).any()):
            raise ValueError("derivatives require per-coefficient errors")
        inherited = torch.zeros_like(center[:, 0])
    else:
        inherited = (error[None] * factors.abs()[:, None] * symmetry).sum(mode_axes)
    return [
        DualForm(c, g, torch.stack((r, r + e), -1), halfwidth)
        for c, g, r, e in zip(center, gradient, remainder, inherited)
    ]

def spectral_disk(node, centers, radii):
    coefficients, modes = node.data["coefficients"], node.data["modes"]
    dimension = len(modes)
    exponent = torch.zeros_like(coefficients.real[0])
    for axis, frequency in enumerate(modes):
        shape = [1] * dimension
        shape[axis] = len(frequency)
        exponent += 2 * math.pi * radii[axis] * frequency.abs().reshape(shape)
    error = node.data.get("coefficient_error")
    if error is None:
        if bool((node.error != 0).any()):
            raise ValueError("complex disks require coefficient errors")
        error = torch.zeros_like(coefficients.real)
    symmetry = torch.where(modes[-1] == 0, 1.0, 2.0)
    radius = (
        ((coefficients.abs() * exponent.expm1() + error * exponent.exp()) * symmetry)
        .flatten(1)
        .sum(-1)
    )
    return synthesis(coefficients, modes, centers), radius[None].expand(
        len(centers), -1
    )
