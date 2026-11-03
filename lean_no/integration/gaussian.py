import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

@lru_cache(maxsize=32)
def gaussian_rule(order):
    if order >= 128:
        from scipy.special import roots_legendre

        return roots_legendre(order)
    return np.polynomial.legendre.leggauss(order)

def gauss_axis(cells, start, end, order, reference, method="gaussian"):
    from lean_no.integration.rules import interval_rule

    nodes, weights, _ = interval_rule(method, order)
    nodes, weights = 2 * nodes - 1, 2 * weights
    nodes = torch.as_tensor(nodes, device=reference.device, dtype=reference.dtype)
    weights = torch.as_tensor(weights, device=reference.device, dtype=reference.dtype)
    panels = torch.arange(cells, device=reference.device, dtype=reference.dtype)[
        :, None
    ]
    points = ((panels + (nodes + 1) / 2) * (end - start) / cells + start).flatten()
    weights = (
        weights[None, :].expand(cells, -1) * (end - start) / (2 * cells)
    ).flatten()
    return points, weights

@dataclass(frozen=True)
class AnalyticSource:

    evaluate: Callable
    caps: tuple
    reference: torch.Tensor

@dataclass(frozen=True)
class GaussianOptions:
    order: int = 16
    panels: tuple = (16, 16)
    batch: tuple = (32, 512)

def gaussian(source, basis=None, domain=None, options=None):
    from lean_no.integration.cell_kernel import CellKernel, cell_kernel_integral

    if isinstance(source, CellKernel):
        return cell_kernel_integral(source, "gaussian", options)
    from lean_no.integration.kernel_norm import LinearizedKernel, kernel_norm

    if isinstance(source, LinearizedKernel):
        return kernel_norm(source, "gaussian", options)
    return moment_integral(source, basis, domain, options, method="gaussian")

def moment_integral(source, basis=None, domain=None, options=None, method="gaussian"):
    if hasattr(source, "sizes"):
        from lean_no.integration.sphere import spherical_integral

        return spherical_integral(source, basis, domain, method, options)
    if hasattr(source, "node_lower"):
        from lean_no.integration.spatial import spatial

        return spatial(source, basis, domain, method, options)
    if not isinstance(source, AnalyticSource):
        raise TypeError("Gaussian integration requires analytic source bounds")
    options = GaussianOptions() if options is None else options
    if not isinstance(options, GaussianOptions):
        raise TypeError("Gaussian integration requires GaussianOptions")
    if type(options.order) is not int or options.order < (
        1 if method == "rectangle" else 2
    ):
        raise ValueError("Gaussian order must be at least two")
    if any(type(n) is not int or n < 1 for n in (*options.panels, *options.batch)):
        raise ValueError("Gaussian panels and batches must be positive integers")
    dimension = len(basis) if basis is not None else 0
    if (
        not 1 <= dimension <= 3
        or len(options.panels) != dimension
        or len(options.batch) != dimension
    ):
        raise ValueError("quadrature partitions must match the Fourier dimension")
    if domain is None or len(domain) != dimension or any(b <= a for a, b in domain):
        raise ValueError("quadrature requires nonempty intervals")
    if len(source.caps) != dimension:
        raise ValueError("one analytic cap per Fourier axis is required")
    if any(not torch.isfinite(c).all() or (c < 0).any() for c in source.caps):
        raise ValueError("analytic caps must be finite and nonnegative")
    if method == "rectangle":
        from dataclasses import replace

        options = replace(options, order=1)
    ref = source.reference
    axes = [
        gauss_axis(n, a, b, options.order, ref, method)
        for n, (a, b) in zip(options.panels, domain)
    ]
    dtype = torch.complex128 if ref.dtype == torch.float64 else torch.complex64
    center = torch.zeros(
        len(ref), *(len(axis) for axis in basis), device=ref.device, dtype=dtype
    )
    inherited = torch.zeros_like(ref)
    labels, modes = "xyz"[:dimension], "klm"[:dimension]
    equation = (
        labels + "c," + ",".join(a + b for a, b in zip(labels, modes)) + "->c" + modes
    )
    error_equation = labels + "c," + ",".join(labels) + "->c"
    batches = [
        range(0, len(axis[0]), count) for axis, count in zip(axes, options.batch)
    ]
    for starts in itertools.product(*batches):
        points, weights, transforms = [], [], []
        for start, count, (values, quadrature), frequency in zip(
            starts, options.batch, axes, basis
        ):
            x, w = values[start : start + count], quadrature[start : start + count]
            points.append(x)
            weights.append(w)
            transforms.append(
                torch.exp(-2j * math.pi * x[:, None] * frequency) * w[:, None]
            )
        value, radius = source.evaluate(*points)
        center += torch.einsum(equation, value.to(dtype), *transforms)
        if radius is not None:
            inherited += torch.einsum(error_equation, radius, *weights)
    volume = math.prod(b - a for a, b in domain)
    remainder = gaussian_error(source.caps, volume, options.order, method)
    if remainder.ndim > 1:
        inherited = inherited.reshape(-1, *([1] * dimension))
    return center, remainder + inherited

def gaussian_error(caps, volume, order, method="gaussian"):
    degree = (
        2 * order
        if method == "gaussian"
        else order
        if method == "clenshaw_curtis"
        else 1
    )
    return 2 * volume * sum(caps) * 0.25**degree / 0.75
