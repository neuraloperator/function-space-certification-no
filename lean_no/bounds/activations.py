import math
from collections.abc import Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from lean_no.bounds.interval import Interval, product_bounds
from lean_no.bounds.relaxations import tanh_rule

# Outward decimal upper bounds used by the nominal-error enclosures.
gelu_first = 1.1289041451852
gelu_second = 0.7978845608029
gelu_third = 0.7787696350558
# Real-axis extrema of GELU derivatives five through seven. Exact rational
# stationary-root isolation and Arb interval evaluation validate these caps in
# tests/test_arithmetic.py; no sampled maxima enter the proof.
gelu_high_caps = {5: 2.842526, 6: 7.180962, 7: 16.447552}
# Retain the existing Fourier rule's evaluation order.
gelu_lipschitz = 0.5 * (1.0 + math.erf(1.0)) + math.exp(-1.0) / math.sqrt(math.pi)
gelu_minimizer = -0.7517915246935645
gelu_minimum = -0.16997120747990367

_GAUSS = {}

def complex_gelu(z, order=8):
    if order == 0:
        from scipy.special import erfc

        argument = (-z.detach() / math.sqrt(2)).cpu().numpy()
        with np.errstate(over="ignore", invalid="ignore"):
            phi = torch.as_tensor(0.5 * erfc(argument), device=z.device, dtype=z.dtype)
        density = (-0.5 * z.square()).exp() / math.sqrt(2 * math.pi)
        zero = torch.zeros_like(z.real)
        return z * phi, phi + z * density, zero, zero
    if order < 2:
        raise ValueError("complex GELU order must be zero or at least two")
    key = (z.device, z.real.dtype, order)
    if key not in _GAUSS:
        nodes, weights = np.polynomial.legendre.leggauss(order)
        _GAUSS[key] = (
            torch.as_tensor((nodes + 1) / 2, device=z.device, dtype=z.real.dtype),
            torch.as_tensor(weights / 2, device=z.device, dtype=z.real.dtype),
        )
    nodes, weights = _GAUSS[key]
    a, b = z.real, z.imag
    integral = torch.zeros_like(z)
    err = torch.zeros_like(a)
    for j in range(4):
        u = (j + nodes) / 4
        zz = a[..., None] + 1j * b[..., None] * u
        integral += ((-0.5 * zz.square()).exp() * weights).sum(-1) / (
            4 * math.sqrt(2 * math.pi)
        )
        # A radius-1/2 disk around the panel midpoint has h/rho=1/4.
        lo = (a.abs() - b.abs() / 2).clamp(min=0)
        height = b.abs() * ((j + 0.5) / 4 + 0.5)
        cap = torch.exp((height.square() - lo.square()) / 2) / math.sqrt(2 * math.pi)
        err += 2 * cap / 4 * 0.25 ** (2 * order) / 0.75
    phi = 0.5 * (1 + torch.erf(a / math.sqrt(2))) + 1j * b * integral
    error = b.abs() * err
    value = z * phi
    slope = phi + z * (-0.5 * z.square()).exp() / math.sqrt(2 * math.pi)
    return value, slope, z.abs() * error, error

def gelu_series(center, degree):
    density = (-center.square() / 2).exp() / math.sqrt(2 * math.pi)
    normal = [density]
    if degree:
        normal.append(-center * density)
    for n in range(2, degree + 1):
        normal.append(-(center * normal[-1] + normal[-2]) / n)
    if center.is_complex():
        # Complex erfc is a primitive under the same exact-special-function
        # assumption as real GELU; no extra Gauss truncation is composed here.
        value, slope, e0, e1 = complex_gelu(center, order=0)
    else:
        value = F.gelu(center)
        slope = 0.5 * torch.erfc(-center / math.sqrt(2)) + center * density
        e0 = e1 = torch.zeros_like(center)
    result = [value]
    if degree:
        result.append(slope)
    result.extend(
        normal[n - 2] / (n * (n - 1)) - normal[n] for n in range(2, degree + 1)
    )
    return torch.stack(result, -1), e0, e1

def gelu_coefficients(center, degree):
    return gelu_series(center, degree)[0]

def gelu_remainder_log_coefficient(order, height=None):
    if order < 2:
        raise ValueError("at least the second derivative is required")

    def log_cap(n):
        return (n - 1) / 2 * math.log(2) + math.lgamma((n + 1) / 2) - math.log(math.pi)

    if height is None:
        a, b = log_cap(order - 2), log_cap(order)
    else:
        # Cauchy-Schwarz under the Gaussian Fourier measure bounds its
        # j-th derivative on |Im z| <= H by sqrt(A_(2j)*A_0)*exp(H^2).
        a = 0.5 * (log_cap(2 * order - 4) + log_cap(0))
        b = 0.5 * (log_cap(2 * order) + log_cap(0))
    hi, lo = max(a, b), min(a, b)
    result = hi + math.log1p(math.exp(lo - hi)) - math.lgamma(order + 1) + 1e-12
    return result if height is None else result + height.square()

def local_gelu_log_coefficient(center, variation, order, complex_domain=False):
    modulus = center.abs() + variation
    if complex_domain or center.is_complex():
        height = (center.imag.abs() if center.is_complex() else 0) + variation
    else:
        height = torch.zeros_like(variation)
    near = (center.real.abs() - variation).clamp(min=0)
    z = modulus.log()
    previous, current = torch.zeros_like(z), z
    lower = previous if order == 2 else current if order == 3 else None
    for n in range(2, order + 1):
        previous, current = (
            current,
            torch.logaddexp(z + current, previous) - math.log(n),
        )
        if n == order - 2:
            lower = current
    polynomial = torch.logaddexp(lower - math.log(order * (order - 1)), current)
    result = (
        polynomial + (height.square() - near.square()) / 2 - math.log(2 * math.pi) / 2
    )
    return torch.nan_to_num(
        result + 1e-12, nan=torch.inf, posinf=torch.inf, neginf=-torch.inf
    )

def is_relu(activation: Callable) -> bool:
    if isinstance(activation, str):
        return activation == "relu"
    return isinstance(activation, nn.ReLU) or activation in {F.relu, torch.relu}

def is_gelu(activation: Callable) -> bool:
    if isinstance(activation, str):
        return activation == "gelu"
    return isinstance(activation, nn.GELU) or activation is F.gelu

def is_tanh(activation: Callable) -> bool:
    if isinstance(activation, str):
        return activation == "tanh"
    return isinstance(activation, nn.Tanh) or activation in {F.tanh, torch.tanh}

def is_identity(activation: Callable) -> bool:
    if isinstance(activation, str):
        return activation == "identity"
    return isinstance(activation, nn.Identity)

def gelu_bounds(
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    lower_value = F.gelu(lower)
    upper_value = F.gelu(upper)
    output_lower = torch.minimum(lower_value, upper_value)
    output_upper = torch.maximum(lower_value, upper_value)
    crosses = (lower <= gelu_minimizer) & (upper >= gelu_minimizer)
    minimum = lower.new_tensor(gelu_minimum)
    output_lower = torch.where(
        crosses,
        torch.minimum(output_lower, minimum),
        output_lower,
    )
    return output_lower, output_upper

def crown_relu_rule(
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    active = lower >= 0
    unstable = (lower < 0) & (upper > 0)

    lower_slope = active.to(lower.dtype)
    upper_slope = active.to(lower.dtype)
    upper_bias = torch.zeros_like(lower)

    unstable_lower = lower[unstable]
    unstable_upper = upper[unstable]
    upper_slope[unstable] = unstable_upper / (unstable_upper - unstable_lower)
    upper_bias[unstable] = -unstable_lower * upper_slope[unstable]
    lower_slope[unstable] = (unstable_upper > -unstable_lower).to(lower.dtype)
    return lower_slope, upper_slope, upper_bias

def crown_tanh_rule(
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return tanh_rule(lower, upper)

def tanh_prime(
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    left = 1 - torch.tanh(lower).square()
    right = 1 - torch.tanh(upper).square()
    low = torch.minimum(left, right)
    high = torch.maximum(left, right)
    high = torch.where((lower <= 0) & (upper >= 0), torch.ones_like(high), high)
    return low, high

def tanh_second_bounds(
    lower: torch.Tensor, upper: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    left = torch.tanh(lower)
    right = torch.tanh(upper)
    left = -2 * left * (1 - left.square())
    right = -2 * right * (1 - right.square())
    low, high = torch.minimum(left, right), torch.maximum(left, right)
    root = math.atanh(1 / math.sqrt(3))
    peak = 4 / (3 * math.sqrt(3))
    low = torch.where((lower <= root) & (upper >= root), -peak, low)
    high = torch.where((lower <= -root) & (upper >= -root), peak, high)
    return low, high

def tanh_hessian(lower, upper, first_low, first_high, last_low, last_high, diagonal):
    low, high = product_bounds(first_low, first_high, last_low, last_high)
    if diagonal:
        low = low.clamp(min=0)
    return product_bounds(*tanh_second_bounds(lower, upper), low, high)

def gelu_derivative(x, order):
    if order == 0:
        return F.gelu(x)
    density = torch.exp(-x.square() / 2) / math.sqrt(2 * math.pi)
    if order == 1:
        return 0.5 * torch.erfc(-x / math.sqrt(2)) + x * density
    if order == 2:
        return (2 - x.square()) * density
    if order == 3:
        return (x**3 - 4 * x) * density
    if order == 4:
        return (-(x**4) + 7 * x.square() - 4) * density
    raise ValueError("GELU derivative order must be between zero and four")

def gelu_range(value, order):
    if order == 0:
        midpoint = (value.lower + value.upper) / 2
        radius = (value.upper - value.lower) / 2 * gelu_range(value, 1).magnitude()
        return Interval.around(F.gelu(midpoint), radius)
    roots = (
        (-math.sqrt(2), math.sqrt(2)),
        (-2.0, 0.0, 2.0),
        tuple(
            s * math.sqrt((7 + t * math.sqrt(33)) / 2) for s in (-1, 1) for t in (-1, 1)
        ),
        (-3.0, -math.sqrt(2), 0.0, math.sqrt(2), 3.0),
    )[order - 1]
    candidates = [
        gelu_derivative(value.lower, order),
        gelu_derivative(value.upper, order),
    ]
    candidates += [
        gelu_derivative(
            torch.full_like(value.lower, root).clamp(value.lower, value.upper), order
        )
        for root in roots
    ]
    candidates = torch.stack(candidates)
    return Interval(candidates.amin(0), candidates.amax(0))

def gelu_sizes(sizes):
    return torch.stack((sizes[0], gelu_first * sizes[1]))

def gelu_linear(center, radius):
    lower, upper = center - radius, center + radius

    def derivative(value):
        return 0.5 * (1 + torch.erf(value / math.sqrt(2))) + value * torch.exp(
            -value.square() / 2
        ) / math.sqrt(2 * math.pi)

    left, right = derivative(lower), derivative(upper)
    low, high = torch.minimum(left, right), torch.maximum(left, right)
    for value in (-math.sqrt(2), math.sqrt(2)):
        inside = (lower <= value) & (value <= upper)
        candidate = derivative(center.new_tensor(value))
        low = torch.where(inside, torch.minimum(low, candidate), low)
        high = torch.where(inside, torch.maximum(high, candidate), high)
    slope = (low + high) / 2
    shift = torch.nn.functional.gelu(center) - slope * center
    return slope, shift, (high - low) / 2

def conservative_gelu_bounds(lower, upper):
    left, right = torch.nn.functional.gelu(lower), torch.nn.functional.gelu(upper)
    low, high = torch.minimum(left, right), torch.maximum(left, right)
    # The stationary minimum lies in (-1, 0) and GELU(x) >= -0.18 globally.
    crossing = (lower < 0) & (upper > -1)
    return torch.where(crossing, torch.minimum(low, low.new_tensor(-0.18)), low), high

def relu_remainder(lower, upper, radius, slope):
    result = torch.zeros_like(lower)
    for center in (
        lower,
        upper,
        torch.zeros_like(lower).clamp(lower, upper),
        radius.expand_as(lower).clamp(lower, upper),
        (-radius).expand_as(lower).clamp(lower, upper),
    ):
        for error in (-radius, radius, (-center).clamp(-radius, radius)):
            value = ((center + error).relu() - center.relu() - slope * error).abs()
            result = torch.maximum(result, value)
    return result
