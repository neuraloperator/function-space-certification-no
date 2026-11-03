import itertools
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from lean_no.bounds.activations import complex_gelu, gelu_derivative, gelu_range
from lean_no.bounds.affine import (
    DualForm,
    compose,
    tanh_compose,
    tanh_derivative,
    tanh_derivative_magnitude,
)
from lean_no.bounds.derivatives import compose_jet
from lean_no.bounds.interval import Interval
from lean_no.bounds.normalization import (
    normalization_adjoint,
    normalization_affine_bounds,
)
from lean_no.bounds.operators import crown_tanh_rule
from lean_no.propagation.derivative_cells import orders_2d
from lean_no.propagation.perturbation import activation_bounds
from lean_no.quantities.ks import orders as ks_orders

def test_tanh_lines_encloseactivation_bounds() -> None:
    lower = torch.tensor([-2.0, -0.5, 0.4])
    upper = torch.tensor([-0.3, 0.7, 1.8])
    low_slope, low_bias, high_slope, high_bias = crown_tanh_rule(lower, upper)
    ratio = torch.linspace(0, 1, 1001)[:, None]
    points = lower + ratio * (upper - lower)
    values = torch.tanh(points)
    assert torch.all(low_slope * points + low_bias <= values + 2e-6)
    assert torch.all(high_slope * points + high_bias >= values - 2e-6)

@pytest.mark.parametrize("groups,near_constant", [(1, False), (4, True)])
def test_normalization_affine_bounds_and_signed_backward(groups, near_constant):
    torch.manual_seed(200 + groups)
    center = torch.randn(4, 3, 4, dtype=torch.float64)
    if near_constant:
        center = center * 1e-8 + 0.2
    radius = torch.rand_like(center) * 0.03
    lo, hi = center - radius, center + radius
    weight = torch.tensor([1.0, -2.0, 0.0, 0.3], dtype=torch.float64)
    bias = torch.tensor([0.1, -0.4, 0.3, 0.2], dtype=torch.float64)
    scale, offset, error = normalization_affine_bounds(
        lo, hi, weight, bias, 1e-5, groups
    )
    objective = torch.randn(3, 4, 3, 4, dtype=torch.float64)
    backward = normalization_adjoint(objective, scale, groups)
    affine_bias = (objective * offset[None, :, None, None]).sum((1, 2, 3))
    residual = (objective.abs() * error).sum((1, 2, 3))
    lower = (
        (backward.clamp_min(0) * lo + backward.clamp_max(0) * hi).sum((1, 2, 3))
        + affine_bias
        - residual
    )
    upper = (
        (backward.clamp_min(0) * hi + backward.clamp_max(0) * lo).sum((1, 2, 3))
        + affine_bias
        + residual
    )
    samples = [lo + torch.rand_like(lo) * (hi - lo) for _ in range(20)]
    for values in (lo, hi, *samples):
        actual = F.group_norm(values[None], groups, weight, bias, eps=1e-5)[0]
        centered = values.reshape(groups, -1)
        centered = (centered - centered.mean(-1, keepdim=True)).reshape_as(values)
        predicted = scale[:, None, None] * centered + offset[:, None, None]
        assert ((actual - predicted).abs() <= error + 1e-11).all()
        targets = (objective * actual).sum((1, 2, 3))
        assert (lower <= targets + 1e-11).all()
        assert (targets <= upper + 1e-11).all()
    torch.testing.assert_close(error[2], torch.zeros_like(error[2]))

@pytest.mark.parametrize(
    "activation", [torch.nn.ReLU(), torch.nn.Tanh(), torch.nn.GELU()]
)
def test_frozen_remainder_including_gelu_negative_slopes(activation):
    lo = torch.tensor([-3.0, -1.3, -0.8, -0.1, 0.2, 1.5], dtype=torch.float64)
    hi = lo + 0.12
    radius = torch.full_like(lo, 0.15)
    slope, error = activation_bounds(type(activation).__name__.lower(), lo, hi, radius)
    for f, g in itertools.product(torch.linspace(0, 1, 13), torch.linspace(-1, 1, 13)):
        nominal = lo + f * (hi - lo)
        delta = g * radius
        actual = (
            activation(nominal + delta) - activation(nominal) - slope * delta
        ).abs()
        assert (actual <= error + 1e-14).all()

def partial(value, points, order):
    for axis, count in enumerate(order):
        for _ in range(count):
            value = torch.autograd.grad(value.sum(), points, create_graph=True)[0][
                :, axis
            ]
    return value

@pytest.mark.parametrize(
    "activation,orders", [("tanh", orders_2d), ("gelu", ks_orders)]
)
def test_chain_rule(activation, orders):
    points = torch.tensor(
        [[0.1, 0.3], [-0.4, 0.7]], dtype=torch.float64, requires_grad=True
    )
    t, x = points.unbind(-1)
    value = torch.sin(t + 0.3 * x) + 0.2 * t * x + 0.1 * x**4
    truth = value.tanh() if activation == "tanh" else torch.nn.functional.gelu(value)
    jet = [
        DualForm.constant(
            partial(value, points, order)[:, None], torch.zeros_like(points)
        )
        for order in orders
    ]
    output = compose_jet(jet, orders, tanh_compose if activation == "tanh" else compose)
    for order, form in zip(orders, output):
        expected = partial(truth, points, order)[:, None]
        torch.testing.assert_close(form.c, expected, rtol=1e-11, atol=1e-12)
        assert bool((form.r == 0).all())

def test_interval_arithmetic_has_exact_extrema():
    lo = torch.tensor([-3.0, -1.0, 2.0], dtype=torch.float64)
    hi = torch.tensor([-2.0, 2.0, 4.0], dtype=torch.float64)
    a, b = Interval(lo, hi), Interval(lo.flip(0), hi.flip(0))
    product = a * b
    vertices = torch.stack(
        [x * y for x, y in itertools.product((a.lower, a.upper), (b.lower, b.upper))]
    )
    torch.testing.assert_close(product.lower, vertices.amin(0))
    torch.testing.assert_close(product.upper, vertices.amax(0))
    squared = a.square()
    torch.testing.assert_close(squared.lower, lo.new_tensor([4.0, 0.0, 4.0]))
    torch.testing.assert_close(squared.upper, lo.new_tensor([9.0, 4.0, 16.0]))
    weight = lo.new_tensor([[2.0, -3.0, 0.0], [-1.0, 0.5, 4.0]])
    actual = a.linear(weight)
    points = lo.new_tensor(list(itertools.product(*zip(lo.tolist(), hi.tolist()))))
    truth = points @ weight.T
    torch.testing.assert_close(actual.lower, truth.amin(0))
    torch.testing.assert_close(actual.upper, truth.amax(0))

@pytest.mark.parametrize("activation", ["tanh", "gelu"])
def test_derivative_enclosures_against_autograd(activation):
    lo = torch.tensor([-2.0, -0.8, -0.1, 0.6], dtype=torch.float64)
    hi = lo + 0.4
    points = (
        lo[:, None] + torch.linspace(0, 1, 23, dtype=lo.dtype) * (hi - lo)[:, None]
    ).requires_grad_()
    values = points.tanh() if activation == "tanh" else F.gelu(points)
    for order in range(1, 7 if activation == "tanh" else 5):
        values = torch.autograd.grad(values.sum(), points, create_graph=True)[0]
        if activation == "tanh":
            torch.testing.assert_close(
                tanh_derivative(points, order), values, atol=2e-10, rtol=2e-10
            )
            cap = tanh_derivative_magnitude(Interval(lo, hi), order)
            assert (values.abs() <= cap[:, None] + 2e-10).all()
        else:
            torch.testing.assert_close(
                gelu_derivative(points, order), values, atol=2e-10, rtol=2e-10
            )
            if order <= 4:
                bounds = gelu_range(Interval(lo, hi), order)
                assert (bounds.lower[:, None] <= values + 2e-10).all()
                assert (bounds.upper[:, None] >= values - 2e-10).all()

def test_complex_gelu_against_scipy():
    from scipy.special import erf

    z = torch.tensor([-6 + 9j, -1 + 1j, 0j, 3 - 7j, 15 + 12j], dtype=torch.complex128)
    expected = z.numpy() * (1 + erf(z.numpy() / math.sqrt(2))) / 2
    value, slope, _, _ = complex_gelu(z, order=0)
    expected_slope = (1 + erf(z.numpy() / math.sqrt(2))) / 2 + z.numpy() * np.exp(
        -(z.numpy() ** 2) / 2
    ) / math.sqrt(2 * math.pi)
    torch.testing.assert_close(
        value, torch.from_numpy(expected), rtol=3e-11, atol=1e-12
    )
    torch.testing.assert_close(
        slope, torch.from_numpy(expected_slope), rtol=3e-11, atol=1e-12
    )

def test_affine_forms_preserve_correlated_cancellation():
    h = torch.tensor([[0.1, 0.2]], dtype=torch.float64)
    form = DualForm.constant(torch.tensor([[0.7]], dtype=torch.float64), h)
    form.g[0, 0] = h.new_tensor([2.0, -3.0])
    form.r[..., 1] = 0.01
    zero = form + form * (-1)
    torch.testing.assert_close(zero.g, torch.zeros_like(zero.g))
    assert float(zero.radius()[..., 1]) == pytest.approx(0.02)
    interval = (form * form).lane(1).interval()
    for x, y, error in itertools.product(
        [-0.1, 0.0, 0.1], [-0.2, 0.0, 0.2], [-0.01, 0.01]
    ):
        truth = (0.7 + 2 * x - 3 * y + error) ** 2
        assert float(interval.lower) <= truth + 1e-12 <= float(interval.upper) + 1e-12

def test_high_gelu_caps_by_complete_root_isolation_and_arb_intervals():
    from fractions import Fraction

    from flint import arb, ctx, fmpq

    from lean_no.bounds.activations import gelu_high_caps

    # GELU^(n)(x)=P_n(x)*exp(-x*x/2)/sqrt(2*pi).
    polynomials = {
        5: (0, 18, 0, -11, 0, 1),
        6: (18, 0, -51, 0, 16, 0, -1),
        7: (0, -120, 0, 115, 0, -22, 0, 1),
        8: (-120, 0, 465, 0, -225, 0, 29, 0, -1),
    }
    numerators = {
        5: (43596768197, 133720391716, 236169520404),
        6: (81004692324, 166230481437, 264003275520),
        7: (37634582279, 114282488490, 195836782776, 290034021376),
    }
    denominator = 2**36

    def evaluate(coefficients, x):
        result = 0
        for a in reversed(coefficients):
            result = result * x + a
        return result

    for n in (5, 6, 7):
        p = polynomials[n]
        recurrence = tuple(
            (j + 1) * p[j + 1] if j + 1 < len(p) else 0 for j in range(len(p) + 1)
        )
        recurrence = tuple(a - (p[j - 1] if j else 0) for j, a in enumerate(recurrence))
        assert recurrence == polynomials[n + 1]
        brackets = [(a, a + 1) for a in numerators[n]] + [
            (-a - 1, -a) for a in numerators[n]
        ]
        if n == 6:
            brackets.append((-1, 1))
        brackets.sort()
        assert all(b < c for (_, b), (c, _) in zip(brackets, brackets[1:]))
        # Disjoint sign changes equal the polynomial degree, so every real
        # stationary point is covered, with no reliance on numerical roots.
        assert len(brackets) == len(polynomials[n + 1]) - 1
        for a, b in brackets:
            lo, hi = Fraction(a, denominator), Fraction(b, denominator)
            assert (
                evaluate(polynomials[n + 1], lo) * evaluate(polynomials[n + 1], hi) < 0
            )
            with ctx.workprec(160):
                x = arb(fmpq(a + b, 2 * denominator), fmpq(b - a, 2 * denominator))
                value = evaluate(p, x) * (-x * x / 2).exp() / (2 * arb.pi()).sqrt()
                assert abs(value).upper() < arb(gelu_high_caps[n])
