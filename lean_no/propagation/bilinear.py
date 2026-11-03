import itertools
import math

import torch

from lean_no.enclosures.bilinear import BilinearBounds
from lean_no.spectral import real_inverse

__all__ = [
    "BilinearBounds",
    "activation",
    "affine",
    "cell_corners",
    "cell_moments",
    "cell_spectrum",
    "linear",
    "real_series",
    "relu",
    "series_bound",
    "spectral",
    "tanh",
]

def linear(values, weight):
    return torch.einsum("oi,qi...->qo...", weight, values)

def affine(state, weight, offset=None):
    from lean_no.bounds.interval import interval_linear

    lower, upper = interval_linear(
        weight, state.lower, state.upper, lambda w, v: linear(v, w)
    )
    if offset is not None:
        lower = lower + offset.reshape(1, -1, *([1] * (lower.ndim - 2)))
        upper = upper + offset.reshape(1, -1, *([1] * (lower.ndim - 2)))
    return BilinearBounds(lower, upper)

def relu(state):
    low = state.lower.amin(dim=0)
    high = state.lower.amax(dim=0)
    gap = torch.where(
        (low < 0) & (high > 0),
        -low * high / (high - low).clamp_min(torch.finfo(low.dtype).tiny),
        torch.zeros_like(low),
    )
    # Jensen bounds the upper interpolant. Subtract its maximum secant gap
    # from the lower interpolant; no derivative of the unknown input is used.
    return BilinearBounds(state.lower.relu() - gap[None], state.upper.relu())

def tanh(state):
    from lean_no.bounds.activations import tanh_second_bounds

    def gap(values, lower):
        lo, hi = values.amin(0), values.amax(0)
        second_lo, second_hi = tanh_second_bounds(lo, hi)
        curvature = second_hi.clamp_min(0) if lower else (-second_lo).clamp_min(0)
        return curvature * (hi - lo).square() / 8

    return BilinearBounds(
        state.lower.tanh() - gap(state.lower, True)[None],
        state.upper.tanh() + gap(state.upper, False)[None],
    )

def cell_corners(values, dimension):
    axes = tuple(range(-dimension, 0))
    return torch.stack(
        tuple(
            torch.roll(values, tuple(-s for s in sides), axes)
            for sides in itertools.product((0, 1), repeat=dimension)
        )
    )

def real_series(coefficients, modes, shape):
    shape = (shape, shape) if isinstance(shape, int) else tuple(shape)
    output = coefficients.new_zeros(
        *coefficients.shape[: -len(shape)], *shape[:-1], shape[-1] // 2 + 1
    )
    indices = torch.meshgrid(
        *(axis.long().remainder(n) for axis, n in zip(modes, shape)), indexing="ij"
    )
    output[(..., *indices)] = coefficients
    return real_inverse(output, shape, "forward")

def cell_moments(frequency, count):
    dtype = torch.complex128 if frequency.dtype == torch.float64 else torch.complex64
    theta = 2 * math.pi * frequency.double() / count
    average = torch.exp(-0.5j * theta) * torch.sinc(theta / (2 * math.pi))
    divisor = torch.where(theta == 0, torch.ones_like(theta), theta.square())
    right = ((1 + 1j * theta) * torch.exp(-1j * theta) - 1) / divisor
    # Integral of t exp(-i theta t); avoid subtracting nearly equal numbers.
    series = torch.zeros_like(right)
    for order in range(8, -1, -1):
        series = series * (-1j * theta) + 1 / (math.factorial(order) * (order + 2))
    right = torch.where(theta.abs() < 0.05, series, right)
    return (average - right).to(dtype), right.to(dtype)

def cell_spectrum(values, modes):
    dimension = len(modes)
    shape = values.shape[-dimension:]
    moments = [cell_moments(axis, n) for axis, n in zip(modes, shape)]
    indices = torch.meshgrid(
        *(axis.long().remainder(n) for axis, n in zip(modes, shape)), indexing="ij"
    )
    result = None
    for corner, sides in enumerate(itertools.product((0, 1), repeat=dimension)):
        spectrum = torch.fft.rfftn(
            values[corner], dim=tuple(range(-dimension, 0)), norm="forward"
        )
        term = spectrum[(..., *indices)]
        for axis, side in enumerate(sides):
            layout = [1] * dimension
            layout[axis] = -1
            term = term * moments[axis][side].reshape(layout)
        result = term if result is None else result + term
    return result

def series_bound(coefficients, modes, shape, radii):
    shape = (shape, shape) if isinstance(shape, int) else tuple(shape)
    dimension = len(shape)
    radii = (radii,) * dimension if isinstance(radii, (float, int)) else radii
    frequency = torch.meshgrid(*(2 * math.pi * axis for axis in modes), indexing="ij")
    symmetry = torch.where(modes[-1] == 0, 1.0, 2.0)
    mass = coefficients.abs() * symmetry
    spatial = tuple(range(-dimension, 0))
    speed = sum(radius * axis.abs() for radius, axis in zip(radii, frequency))
    magnitude = mass.sum(spatial)
    gain = (mass * speed).sum(spatial)
    remainder = (mass * speed.square()).sum(spatial) / 2
    layout = (*magnitude.shape, *([1] * dimension))
    value = real_series(coefficients, modes, shape).abs()
    ceiling = torch.minimum(value + gain.reshape(layout), magnitude.reshape(layout))
    local = value + remainder.reshape(layout)
    for radius, axis in zip(radii, frequency):
        local = (
            local + radius * real_series(coefficients * (1j * axis), modes, shape).abs()
        )
    return torch.minimum(local, ceiling)

def spectral(state, weight, modes, offset=None, channel_chunk=None):
    dimension = len(modes)
    shape = state.lower.shape[-dimension:]
    spatial = tuple(range(-dimension, 0))
    corners = 2**dimension
    if any(n <= 2 * int(axis.abs().max()) for axis, n in zip(modes, shape)):
        raise ValueError("the grid must resolve every retained Fourier mode")
    frequency = torch.meshgrid(*(2 * math.pi * axis for axis in modes), indexing="ij")
    halfwidth = tuple(0.5 / n for n in shape)
    coefficients = cell_spectrum((state.lower + state.upper) / 2, modes)
    product = torch.einsum("oi...,i...->o...", weight, coefficients)
    mean_radius = (state.upper - state.lower).mean(0) / 2
    symmetry = torch.where(modes[-1] == 0, 1.0, 2.0)
    mass = weight.abs() * symmetry
    phase = torch.exp(-1j * sum(h * axis for h, axis in zip(halfwidth, frequency)))
    radius_fft = torch.fft.rfftn(mean_radius, dim=spatial)
    total_radius = mean_radius.mean(spatial)
    error = torch.zeros(weight.shape[0], device=weight.device, dtype=state.lower.dtype)
    for n, direction in zip(shape, frequency):
        central = (product.abs() * direction.square() * symmetry).sum(spatial)
        uncertain = (mass * direction.square()).sum(spatial) @ total_radius
        error = error + (central + uncertain) / (8 * n**2)
    channels = weight.shape[0]
    chunk = channels if channel_chunk is None else channel_chunk
    lower = state.lower.new_empty(corners, channels, *shape)
    upper = torch.empty_like(lower)
    layout = (-1, *([1] * dimension))
    for start in range(0, channels, chunk):
        stop = min(start + chunk, channels)

        def convolve(kernel):
            spectrum = torch.fft.rfftn(kernel, dim=spatial)
            projected = torch.einsum("oi...,i...->o...", spectrum, radius_fft)
            return (real_inverse(projected, shape) / math.prod(shape)).clamp_min(0)

        kernel = series_bound(weight[start:stop] * phase, modes, shape, halfwidth)
        node_radius = convolve(kernel)
        cell_error = error[start:stop].reshape(layout)
        curvature = torch.zeros_like(node_radius)
        for n, direction in zip(shape, frequency):
            central = series_bound(
                -product[start:stop] * direction.square() * phase.conj(),
                modes,
                shape,
                halfwidth,
            )
            kernel = series_bound(
                -weight[start:stop] * direction.square(),
                modes,
                shape,
                tuple(1 / m for m in shape),
            )
            curvature = curvature + (central + convolve(kernel)) / (8 * n**2)
        cell_error = torch.minimum(cell_error, curvature)
        node_center = real_series(product[start:stop], modes, shape)
        if offset is not None:
            node_center = node_center + offset[start:stop].reshape(layout)
        lower[:, start:stop] = (
            cell_corners(node_center - node_radius, dimension) - cell_error[None]
        )
        upper[:, start:stop] = (
            cell_corners(node_center + node_radius, dimension) + cell_error[None]
        )
    return BilinearBounds(lower, upper)

def activation(state, name):
    if name == "identity":
        return state
    if name in ("relu", "tanh"):
        return {"relu": relu, "tanh": tanh}[name](state)
    if name == "gelu":
        from lean_no.bounds.activations import gelu_first, gelu_second

        center, radius = (
            (state.lower + state.upper) / 2,
            (state.upper - state.lower) / 2,
        )
        gap = gelu_second * (center.amax(0) - center.amin(0)).square() / 8
        value = torch.nn.functional.gelu(center)
        return BilinearBounds(
            value - gap[None] - gelu_first * radius,
            value + gap[None] + gelu_first * radius,
        )
    raise ValueError(f"no multilinear activation rule for {name}")
