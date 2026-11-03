import math

import torch

from lean_no.bounds.activations import (
    gelu_coefficients,
    gelu_remainder_log_coefficient,
    local_gelu_log_coefficient,
)
from lean_no.graph import walk
from lean_no.spectral.spherical import gelu_first, legendre_basis
from lean_no.spectral.spherical_strip import theta_derivative

@torch.no_grad()
def spherical_taylor_cells(field, centers, half_widths, degree=8):
    if degree < 1:
        raise ValueError("Taylor degree must be positive")
    indices = [(a, k - a) for k in range(degree + 1) for a in range(k, -1, -1)]
    lookup = {pair: i for i, pair in enumerate(indices)}
    pairs = [
        (i, j, lookup[a + c, b + d])
        for i, (a, b) in enumerate(indices)
        for j, (c, d) in enumerate(indices)
        if a + b + c + d <= degree
    ]
    left_index, right_index, output_index = torch.tensor(pairs, device=centers.device).T
    orders = torch.tensor([a + b for a, b in indices], device=centers.device)
    order_range = torch.arange(degree + 1, device=centers.device)
    tail_mask = (order_range[:, None] + order_range[None, :] > degree).to(centers)
    cache, geometry = {}, {}

    def multiply(left, right):
        product = torch.zeros_like(left)
        product.scatter_add_(
            -1,
            output_index.expand(*left.shape[:-1], -1),
            left[..., left_index] * right[..., right_index],
        )
        a = left.new_zeros((*left.shape[:-1], degree + 1))
        b = torch.zeros_like(a)
        a.scatter_add_(-1, orders.expand_as(left), left.abs())
        b.scatter_add_(-1, orders.expand_as(right), right.abs())
        tail = torch.einsum("...i,ij,...j->...", a, tail_mask, b)
        return product, tail

    def activation(polynomial, error):
        series = gelu_coefficients(polynomial[..., 0], degree)
        argument = polynomial.clone()
        argument[..., 0] = 0
        variation = argument.abs().sum(-1)
        result = torch.zeros_like(polynomial)
        result[..., 0] = series[..., -1]
        remainder = torch.zeros_like(error)
        for k in range(degree - 1, -1, -1):
            result, tail = multiply(result, argument)
            remainder = remainder * variation + tail
            result[..., 0] += series[..., k]
        coefficient = torch.minimum(
            variation.new_tensor(gelu_remainder_log_coefficient(degree + 1)),
            local_gelu_log_coefficient(polynomial[..., 0], variation, degree + 1),
        )
        analytic_tail = torch.exp(coefficient + (degree + 1) * variation.log())
        analytic_tail = torch.where(variation == 0, 0.0, analytic_tail)
        return result, remainder + analytic_tail + gelu_first * error

    def apply(node, visit):
        if node.kind == "spectral":
            coefficients = node.data["coefficients"]
            degrees, harmonic_orders = coefficients.shape[-2:]
            key = degrees, harmonic_orders
            if key not in geometry:
                basis = legendre_basis(centers[:, 0], degrees, degrees)
                bases = [basis]
                for _ in range(degree):
                    bases.append(
                        theta_derivative(bases[-1])
                        if degrees > 1
                        else torch.zeros_like(basis)
                    )
                modes = torch.arange(
                    harmonic_orders, device=centers.device, dtype=centers.dtype
                )
                phase = torch.exp(1j * centers[:, 1, None] * modes)
                factors = torch.full_like(modes, 2)
                factors[0] = 1
                geometry[key] = (
                    [v[:, :harmonic_orders].to(coefficients) for v in bases],
                    modes,
                    phase,
                    factors,
                )
            bases, modes, phase, factors = geometry[key]
            values = []
            derivatives = [
                torch.einsum("clm,lmt->ctm", coefficients, basis) for basis in bases
            ]
            for a, b in indices:
                value = torch.einsum(
                    "ctm,tm,m->tc", derivatives[a], phase * (1j * modes) ** b, factors
                ).real
                factor = (
                    half_widths[:, 0] ** a
                    * half_widths[:, 1] ** b
                    / (math.factorial(a) * math.factorial(b))
                )
                values.append(value * factor[:, None])
            polynomial = torch.stack(values, -1)
            l = torch.arange(degrees, device=centers.device, dtype=centers.dtype)
            amplitude = (coefficients.abs().square() * factors).sum(-1).sqrt()
            amplitude *= ((2 * l + 1) / (4 * math.pi)).sqrt()
            derivative_cap = (amplitude * l ** (degree + 1)).sum(-1)
            error = (
                half_widths.sum(-1, keepdim=True) ** (degree + 1)
                / math.factorial(degree + 1)
                * derivative_cap
            )
            result = polynomial, error
        else:
            parents = [visit(parent) for parent in node.parents]
            if node.kind == "affine":
                polynomial, error = parents[0]
                weight, bias = node.data["weight"], node.data["bias"]
                polynomial = torch.einsum("nck,oc->nok", polynomial, weight)
                polynomial[..., 0] += bias
                result = polynomial, error @ weight.abs().T
            elif node.kind == "activation":
                result = activation(*parents[0])
            elif node.kind == "add":
                result = tuple(a + b for a, b in zip(*parents))
            elif node.kind == "concat":
                result = tuple(torch.cat(parts, dim=1) for parts in zip(*parents))
            else:
                raise NotImplementedError(node.kind)
        return result

    polynomial, remainder = walk(apply, cache)(field)
    return polynomial[:, 0, 0], polynomial[:, 0, 1:].abs().sum(-1) + remainder[:, 0]
