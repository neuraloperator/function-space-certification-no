import math
from functools import lru_cache

import torch

from lean_no.spectral.spherical import legendre_basis

def clenshaw_curtis(source, basis=None, domain=None, options=None):
    from lean_no.integration.cell_kernel import CellKernel, cell_kernel_integral

    if isinstance(source, CellKernel):
        return cell_kernel_integral(source, "clenshaw_curtis", options)
    from lean_no.integration.kernel_norm import LinearizedKernel, kernel_norm

    if isinstance(source, LinearizedKernel):
        return kernel_norm(source, "clenshaw_curtis", options)
    from lean_no.integration.gaussian import AnalyticSource, moment_integral

    if isinstance(source, AnalyticSource):
        return moment_integral(source, basis, domain, options, method="clenshaw_curtis")
    if hasattr(source, "node_lower"):
        from lean_no.integration.spatial import spatial

        return spatial(source, basis, domain, "clenshaw_curtis", options)
    from lean_no.integration.sphere import spherical_integral

    return spherical_integral(source, basis, domain, "clenshaw_curtis", options)

def spherical_moments(source, grid, degrees, orders, split=True, moment_cache=None):
    from lean_no.propagation.spherical import evaluate

    if source.kind == "spectral":
        old = source.data["coefficients"]
        result = old.new_zeros((old.shape[0], degrees, orders))
        lmax, mmax = min(degrees, old.shape[1]), min(orders, old.shape[2])
        result[:, :lmax, :mmax] = old[:, :lmax, :mmax]
        return result, True
    if (
        grid.latitude.dtype != source.sizes.dtype
        or grid.latitude.device != source.sizes.device
    ):
        raise ValueError("field and quadrature must share a device and dtype")
    if grid.nlon < 2 * (min(degrees, orders) - 1) + 1:
        raise ValueError("Q is too small for retained modes; modes are never reduced")
    dtype = torch.complex64 if source.sizes.dtype == torch.float32 else torch.complex128
    # Integration is linear through the reference's fixed affine operations.
    # Cache only normalized nonlinear leaves, on CPU: old residual branches
    # need not be reevaluated over the entire Q-by-Q sphere in later blocks.
    normalized = source.kind == "affine" and "normalization" in source.data
    moment_cache = {} if moment_cache is None else moment_cache
    cache_key = (source, id(grid), degrees, orders)
    cached = moment_cache.get(cache_key) if split and normalized else None
    if cached is not None and cached[0] is grid and cached[1:3] == (degrees, orders):
        return cached[3].to(device=source.error.device, copy=True), cached[4]
    if split and source.kind in ("affine", "add", "concat") and not normalized:
        parts = [
            spherical_moments(p, grid, degrees, orders, moment_cache=moment_cache)
            for p in source.parents
        ]
        if source.kind == "affine":
            result = torch.einsum(
                "oi,ilm->olm", source.data["weight"].to(dtype), parts[0][0]
            )
            result[:, 0, 0] += source.data["bias"] * math.sqrt(4 * math.pi)
        elif source.kind == "add":
            result = sum(p[0] for p in parts)
        else:
            result = torch.cat([p[0] for p in parts])
        return result, all(p[1] for p in parts)
    result = torch.zeros(
        (len(source.error), degrees, orders), dtype=dtype, device=source.error.device
    )
    for start in range(0, len(grid.latitude), source.chunk_size):
        latitude = grid.latitude[start : start + source.chunk_size]
        cache = {}
        values = evaluate(source, latitude, grid.longitude, True, cache)
        modes = torch.fft.rfft(values, dim=1, norm="forward")[:, :orders]
        basis = cache.get((degrees, orders))
        if basis is None:
            basis = legendre_basis(latitude, degrees, orders).to(result.dtype)
        weighted = basis * grid.quadrature[start : start + len(latitude)][None, None, :]
        result += 2 * math.pi * torch.einsum("tmc,lmt->clm", modes, weighted)
    if split and normalized and len(source.error) > 8:
        moment_cache[cache_key] = (
            grid,
            degrees,
            orders,
            result.to(device="cpu", copy=True),
            False,
        )
    return result, False

def integration_error(sizes, grid, degrees):
    mass = grid.quadrature.sum() / 2
    probability = grid.quadrature / (2 * mass)
    edges = torch.cat((probability.new_zeros(1), probability.cumsum(0)))
    edges = (1 - 2 * edges).clamp(-1, 1).acos()
    separation = torch.maximum(
        (grid.latitude - edges[:-1]).abs(), (grid.latitude - edges[1:]).abs()
    )
    distance = (probability * separation).sum() + math.pi / grid.nlon
    degree = torch.arange(degrees, device=sizes.device, dtype=sizes.dtype)
    derivative = sizes[1, :, None] + degree * sizes[0, :, None]
    bound = 4 * math.pi * (distance * derivative + (1 - mass).abs() * sizes[0, :, None])
    return torch.minimum(bound, 4 * math.pi * (1 + mass) * sizes[0, :, None])

@lru_cache(maxsize=32)
def strip_alias_logs(q, radii):
    import numpy as np
    from scipy.special import logsumexp

    radius = np.asarray(radii)
    n = q - 1
    frequencies = np.arange(q, 4 * q + 1, 2, dtype=np.float64)
    folded = frequencies % (2 * n)
    folded = np.minimum(folded, 2 * n - folded)
    difference = np.abs(2 / (1 - frequencies**2) - 2 / (1 - folded**2))
    active = difference > 0
    finite = logsumexp(
        np.log(difference[active])[None] - radius[:, None] * frequencies[active][None],
        axis=1,
    )
    # Every cosine integral and its alias have modulus <= 2.
    tail = math.log(4) - (4 * q + 2) * radius - np.log(-np.expm1(-2 * radius))
    return np.logaddexp(finite, tail) + 1e-12

def strip_integration_error(source, grid, degrees):
    from lean_no.bounds.positive import positive_map

    if grid.kind != "clenshaw-curtis" or grid.nlon % 2:
        return None
    from lean_no.spectral.spherical_strip import nominal_affine_terms

    terms, bias = nominal_affine_terms(source)
    if not any("signed_strip" in leaf.data for leaf in terms):
        return None
    parts = []
    for coordinate in ("theta", "phi"):
        radii = torch.cat(
            [
                leaf.data["signed_strip"][coordinate][0]
                for leaf in terms
                if "signed_strip" in leaf.data
            ]
        ).unique(sorted=True)
        cap = bias.to(radii).abs()[None].expand(len(radii), -1).clone()
        for leaf, weight in terms.items():
            local = source.error.new_full((len(radii), len(leaf.error)), math.inf)
            if leaf.kind == "spectral":
                coefficients = leaf.data["coefficients"]
                degree = torch.arange(
                    coefficients.shape[1], device=radii.device, dtype=radii.dtype
                )
                factors = radii.new_full((coefficients.shape[2],), 2.0)
                factors[0] = 1
                amplitude = (coefficients.abs().square() * factors).sum(-1).sqrt()
                amplitude *= ((2 * degree + 1) / (4 * math.pi)).sqrt()
                local = (radii[:, None] * degree).exp() @ amplitude.T
            if "signed_strip" in leaf.data:
                saved_radii, saved_caps = leaf.data["signed_strip"][coordinate]
                # Maximum modulus on a periodic coordinate strip bounds
                # all narrower strips by its wider boundary bound.
                candidates = torch.where(
                    saved_radii[:, None, None] >= radii[None, :, None],
                    saved_caps[:, None],
                    math.inf,
                ).amin(0)
                local = torch.minimum(local, candidates)
            cap += positive_map(local, weight.abs().T)
        degree = torch.arange(degrees, device=radii.device, dtype=radii.dtype)
        growth = radii[:, None] * degree
        if coordinate == "theta":
            cosine = radii.new_tensor(
                strip_alias_logs(len(grid.latitude), tuple(radii.cpu().tolist()))
            )
            correction = math.log(4 * math.pi) + cosine
        else:
            qr = grid.nlon * radii
            correction = math.log(8 * math.pi) - qr - torch.log(-torch.expm1(-qr))
        parts.append(
            (cap.log()[:, :, None] + growth[:, None] + correction[:, None, None]).amin(
                0
            )
        )
    logs = torch.logaddexp(*parts)
    return logs.exp()
