import math
from types import SimpleNamespace

import pytest
import torch
from flint import acb, arb, ctx

from lean_no.certify import certify
from lean_no.domain import FourierGrid
from lean_no.function_space import InputBounds, coord, sin_pi
from lean_no.integration import integrate
from lean_no.integration.arb import integrate_square
from lean_no.integration.gaussian import AnalyticSource, GaussianOptions, gauss_axis
from lean_no.methods import Method
from tests.helpers import averaging_model

def test_rectangle_integral():
    # The exact integral of x on [0, 1] is 1/2, enclosed using its derivative.
    nodes = torch.arange(8, dtype=torch.float64)[None] / 8
    source = SimpleNamespace(
        node_lower=nodes,
        node_upper=nodes,
        dimension=1,
        resolution=(8,),
        lengths=(1.0,),
        steps=(1 / 8,),
        partial_magnitude=torch.ones(1, 1, 8, dtype=torch.float64),
    )
    lower, upper = integrate(source, method="rectangle")
    assert lower.item() <= 0.5 <= upper.item()
    torch.testing.assert_close(
        upper - lower, torch.tensor([0.125], dtype=torch.float64)
    )

@pytest.mark.parametrize("batch", [(3, 5)])
def test_gaussian_integral(batch):
    # A constant's Fourier coefficients are known independently of the quadrature.
    ref = torch.ones(1, dtype=torch.float64)
    modes = torch.tensor([0.0, 1.0], dtype=ref.dtype)

    def evaluate(x, y):
        center = ref.expand(len(x), len(y), 1)
        return center, torch.full_like(center, 0.03)

    # Panel radius = 2/4; |exp(-2πikz)| <= exp(π) for these modes.
    cap = ref * torch.exp(ref * torch.pi)
    source = AnalyticSource(evaluate, (cap, cap), ref)
    center, radius = integrate(
        source,
        (modes, modes),
        ((0.0, 1.0), (0.0, 1.0)),
        "gaussian",
        GaussianOptions(order=12, panels=(4, 4), batch=batch),
    )
    expected = torch.zeros_like(center)
    expected[:, 0, 0] = 1
    assert ((center - expected).abs() <= radius[:, None, None]).all()
    assert 0.03 <= radius.item() < 0.030001
    torch.testing.assert_close(center, expected, atol=1e-13, rtol=1e-13)

def test_nested_integral_complex_outer_enclosures():
    with ctx.workdps(30):
        result = integrate_square(
            lambda z, analytic: z[0] ** 2 * z[1] ** 3, arb("1e-15")
        )
        assert result.contains(acb(1) / 12)
        result = integrate_square(
            lambda z, analytic: (2 * (z[0] + z[1])).exp_pi_i(), arb("1e-15")
        )
        assert result.contains(0)
        assert float(abs(result)) < 1e-12

@pytest.mark.parametrize("order", [4])
def test_gaussian_original_ibp_contains_known_relu_integral(order):
    model = averaging_model(1)
    model.lifting.non_linearity = torch.nn.ReLU()
    source = InputBounds.exact((sin_pi(2 * coord()),), [1.0], [2 * math.pi])
    result = certify(
        model,
        FourierGrid((64,)),
        source,
        integration_options={"order": order},
        method=Method("ibp", "gaussian"),
    )
    low, high = result.query(torch.linspace(0, 1, 21, dtype=torch.float64)[:, None])
    assert (low <= 1 / math.pi).all() and (high >= 1 / math.pi).all()
    assert result.enclosure.stats["quadrature_order"] == order

def test_gaussian_uses_original_source_band_and_preserves_kernel_cancellation():
    model = averaging_model(1)
    with torch.no_grad():
        conv = model.fno_blocks.convs[0]
        conv.weight.tensor[0, 0, 0] = 0
        conv.weight.tensor[0, 0, 1] = 0.5
    inputs = InputBounds((-1 + 0 * coord(),), (1 + 0 * coord(),), [1.0], [0.0])
    for order in (None, 2):
        if order:
            field = certify(
                model,
                FourierGrid((1024,)),
                inputs,
                integration_options={"order": order},
                method=Method("ibp", "gaussian"),
            )
        else:
            field = certify(model, FourierGrid((1024,)), inputs)
        width = float((field.enclosure.node_upper - field.enclosure.node_lower).max())
        assert 4 / math.pi - 1e-12 <= width < 1.29

@pytest.mark.parametrize("method", ["gaussian", "clenshaw_curtis"])
def test_quadrature_polynomial_on_a_nonunit_interval(method):
    ref = torch.zeros(1, dtype=torch.float64)
    x, w = gauss_axis(3, 0.3, 1.7, 8, ref, method)
    assert (w > 0).all()
    for degree in (0, 1, 6):
        truth = (1.7 ** (degree + 1) - 0.3 ** (degree + 1)) / (degree + 1)
        assert float((w * x**degree).sum()) == pytest.approx(truth, abs=3e-13)

def test_nonzero_complex_fourier_moments():
    modes = torch.tensor([0.0, 1.0], dtype=torch.float64)
    c = torch.tensor([0.3 - 0.2j], dtype=torch.complex128)

    def evaluate(x, y):
        z = (2j * math.pi * (x[:, None] + y[None, :])).exp()
        return (c[:, None, None] * z).real.permute(1, 2, 0), torch.zeros(
            len(x), len(y), 1, dtype=x.dtype
        )

    # On radius 2/8 in each coordinate the exponential grows at most exp(pi).
    cap = c.abs() * math.exp(math.pi)
    center, radius = integrate(
        AnalyticSource(evaluate, (cap, cap), c.abs()),
        (modes, modes),
        ((0.0, 1.0),) * 2,
        "gaussian",
        GaussianOptions(12, (8, 8), (3, 5)),
    )
    expected = torch.zeros_like(center)
    expected[:, 1, 1] = c / 2
    assert ((center - expected).abs() <= radius[:, None, None] + 2e-14).all()
    assert radius.max() < 1e-8

def test_analytic_gaussian_remainder_encloses_nonpolynomial_integral():
    ref = torch.ones(1, dtype=torch.float64)

    def evaluate(x, y):
        return (4 * x).exp()[:, None, None].expand(-1, len(y), 1), torch.zeros(
            len(x), len(y), 1, dtype=x.dtype
        )

    # Four panels use complex disks of radius 1/2: |exp(4z)| <= exp(6).
    source = AnalyticSource(
        evaluate, (ref * math.exp(6), ref * math.exp(6)), ref * math.exp(4)
    )
    errors = []
    observed = []
    for order in (2, 4):
        center, radius = integrate(
            source,
            (ref * 0, ref * 0),
            ((0.0, 1.0),) * 2,
            "gaussian",
            GaussianOptions(order, (4, 4), (4, 4)),
        )
        error = abs(float(center.real) - math.expm1(4) / 4)
        assert error <= float(radius) + 1e-12
        assert radius.min() > 0
        errors.append(float(radius))
        observed.append(error)
    assert (
        observed[0] > 1e-4
    )  # A missing remainder must fail independently of roundoff.
    assert errors[1] < errors[0] and observed[1] < observed[0]

def test_tanh_complex_disk_rejects_a_pole():
    from lean_no.integration.coefficients import tanh_disk

    with pytest.raises(ArithmeticError, match="pole"):
        tanh_disk(
            torch.tensor([0.0], dtype=torch.float64),
            torch.tensor([2.0], dtype=torch.float64),
        )
