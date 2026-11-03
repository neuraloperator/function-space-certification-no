import math
from contextlib import contextmanager
from functools import lru_cache

import numpy as np
import torch
from scipy.special import sph_legendre_p_all

from lean_no.bounds.activations import (
    conservative_gelu_bounds,
    gelu_first,
    gelu_linear,
    gelu_second,
    gelu_sizes,
    gelu_third,
)
from lean_no.function_space.spherical import SphereGrid

__all__ = [
    "basis_cache_capacity",
    "gelu_bounds",
    "gelu_first",
    "gelu_linear",
    "gelu_second",
    "gelu_sizes",
    "gelu_third",
    "harmonic_eval",
    "harmonic_sizes",
    "legendre_basis",
    "sfno_grid",
]

# g'(x)=Phi(x)+x*phi(x) is extremal at +/-sqrt(2), and
# g''(x)=(2-x*x)*phi(x) has maximum absolute value at zero.
# Rounded upward from (1+erf(1))/2+exp(-1)/sqrt(pi) and sqrt(2/pi).
# g'''=x*(x^2-4)*phi(x), with its largest modulus at
# x^2=(7-sqrt(33))/2.

gelu_bounds = conservative_gelu_bounds


def sfno_grid(
    q,
    device="cpu",
    dtype=torch.float32,
    gauss_nodes=None,
    rule="composite-gauss",
    nlon=None,
):
    if type(q) is not int or q < 4:
        raise ValueError("q must be an integer >= 4")
    nlon = q if nlon is None else nlon
    if type(nlon) is not int or nlon < 3:
        raise ValueError("longitude count must be an integer >= 3")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("SFNO grids require float32 or float64")
    maximum = (
        (4 if dtype == torch.float32 else 8) if gauss_nodes is None else gauss_nodes
    )
    if type(maximum) is not int or maximum not in (1, 2, 4, 8, 16, 32):
        raise ValueError("gauss_nodes must be 1, 2, 4, 8, 16 or 32")
    count = math.gcd(q, maximum)
    if rule == "clenshaw-curtis":
        if q % 2 or nlon % 2:
            raise ValueError(
                "Clenshaw-Curtis SFNO quadrature requires even latitude and longitude counts"
            )
        from scipy.fft import dct

        degree = q - 1
        cosine_integrals = np.zeros(q)
        cosine_integrals[::2] = 1 / (1 - np.arange(0, q, 2, dtype=np.float64) ** 2)
        quadrature = 2 / degree * dct(cosine_integrals, type=1)
        quadrature[[0, -1]] /= 2
        return SphereGrid(
            torch.linspace(0, math.pi, q, device=device, dtype=dtype),
            torch.tensor(quadrature, device=device, dtype=dtype),
            nlon,
            rule,
            count,
        )
    if rule != "composite-gauss":
        raise ValueError("unknown SFNO quadrature rule")
    nodes, weights = np.polynomial.legendre.leggauss(count)
    step = math.pi / (q // count)
    latitude = (np.arange(q // count)[:, None] + (nodes + 1) / 2).reshape(-1) * step
    quadrature = np.tile(weights, q // count) * step / 2 * np.sin(latitude)
    return SphereGrid(
        torch.tensor(latitude, dtype=dtype, device=device),
        torch.tensor(quadrature, dtype=dtype, device=device),
        nlon,
        "composite-gauss",
        count,
    )

@lru_cache(maxsize=128)
def cached_basis(theta, degrees, orders, dtype=np.float64):
    latitude = np.frombuffer(theta, dtype=np.float64)
    return sph_legendre_p_all(degrees - 1, orders - 1, latitude)[0, :, :orders].astype(
        dtype
    )

@contextmanager
def basis_cache_capacity(maxsize):
    global cached_basis
    previous = cached_basis
    cached_basis = lru_cache(maxsize=maxsize)(previous.__wrapped__)
    try:
        yield
    finally:
        cached_basis.cache_clear()
        cached_basis = previous

def legendre_basis(latitude, degrees, orders):
    theta = latitude.detach().to(device="cpu", dtype=torch.float64).numpy().tobytes()
    result = torch.as_tensor(
        cached_basis(
            theta,
            degrees,
            orders,
            np.float32 if latitude.dtype == torch.float32 else np.float64,
        ),
        dtype=latitude.dtype,
        device=latitude.device,
    )
    if not torch.isfinite(result).all():
        raise FloatingPointError("nonfinite normalized spherical basis")
    return result

def harmonic_eval(coefficients, latitude, longitude, cartesian=False, cache=None):
    latitude, longitude = (
        latitude.to(coefficients.real),
        longitude.to(coefficients.real),
    )
    degrees, orders = coefficients.shape[-2:]
    cache = {} if cache is None else cache
    key = (degrees, orders)
    if key not in cache:
        cache[key] = legendre_basis(latitude, degrees, orders).to(coefficients.dtype)
    values = torch.einsum("clm,lmt->ctm", coefficients, cache[key])
    if cartesian:
        if len(longitude) < 2 * (min(degrees, orders) - 1) + 1:
            raise ValueError("longitude grid aliases retained spherical orders")
        return torch.fft.irfft(
            values, n=len(longitude), dim=-1, norm="forward"
        ).permute(1, 2, 0)
    phase = torch.exp(
        1j * longitude[:, None] * torch.arange(orders, device=longitude.device)
    )
    factors = longitude.new_full((orders,), 2.0)
    factors[0] = 1
    return torch.einsum("ctm,tm,m->tc", values, phase, factors).real

def harmonic_sizes(coefficients):
    degree = torch.arange(
        coefficients.shape[-2],
        device=coefficients.device,
        dtype=coefficients.real.dtype,
    )
    factors = degree.new_full((coefficients.shape[-1],), 2.0)
    factors[0] = 1
    energy = (coefficients.abs().square() * factors).sum(-1).sqrt()
    scale = ((2 * degree + 1) / (4 * math.pi)).sqrt()
    amplitude = energy * scale
    return torch.stack((amplitude.sum(-1), (amplitude * degree).sum(-1)))
