import math

import pytest
import torch

from lean_no.bounds.affine import DualForm
from lean_no.certify import certify
from lean_no.domain import FourierGrid
from lean_no.evaluation import sample_residual
from lean_no.function_space.analytic import describe
from lean_no.kernel import mode_weights
from lean_no.methods import Method
from lean_no.propagation.derivative_cells import DerivativeField
from lean_no.quantities import Residual
from lean_no.quantities.residual import burgers_norm, residual_bound, residual_form
from lean_no.quantities.solution import solution_bound
from tests.helpers import averaging_model, burgers_identity, burgers_model

@pytest.mark.parametrize("nonlinear,pde", [(True, "poisson"), (True, "helmholtz")])
def test_gaussian_derivatives_and_endpoint_traces(nonlinear, pde):
    model = averaging_model(1)
    with torch.no_grad():
        model.fno_blocks.fno_skips[0].conv.weight.fill_(0.25)
    if nonlinear:
        model.projection.non_linearity = torch.nn.Tanh()

    def forcing(x):
        return 1 + torch.sin(2 * math.pi * x)

    field = certify(
        model,
        FourierGrid((256,), dtype=torch.float64),
        forcing,
        moment_order=16,
        derivatives=2,
        method=Method("ibp", "gaussian"),
    )
    assert isinstance(field.enclosure, DerivativeField)
    x = torch.linspace(0, 1, 1025, dtype=torch.float64, requires_grad=True)
    exact = 1.25 + 0.25 * torch.sin(2 * math.pi * x)
    if nonlinear:
        exact = exact.tanh()
    derivative = torch.autograd.grad(exact.sum(), x, create_graph=True)[0]
    second = torch.autograd.grad(derivative.sum(), x)[0]
    target = -second - forcing(x)
    coefficient = None if pde == "poisson" else 1.7
    if coefficient is not None:
        target -= coefficient**2 * exact
    result = residual_bound(field, Residual(pde, coefficient), chunk_size=256)
    lower, upper = result.query(x.detach()[:, None])
    assert (lower[:, 0] <= target + 1e-11).all()
    assert (upper[:, 0] >= target - 1e-11).all()
    lo, hi = field.query(x.detach()[::64, None])
    assert (lo[:, 0] <= exact[::64] + 1e-11).all()
    assert (hi[:, 0] >= exact[::64] - 1e-11).all()
    samples = sample_residual(result, points=x.detach()[::64, None], batch_size=5)
    assert samples["kind"] == "enclosure_midpoint"
    assert (samples["lower"][:, 0] <= target[::64] + 1e-11).all()
    assert (samples["upper"][:, 0] >= target[::64] - 1e-11).all()
    norms = result.bounds
    boundary = exact.detach()[[0, -1]] if pde == "poisson" else torch.zeros(2)
    assert (norms.boundary_lower <= boundary + 1e-11).all()
    assert (norms.boundary_upper >= boundary - 1e-11).all()
    assert solution_bound(result).solution_error_upper >= 0
    assert all(
        row["gauss_remainder_max"] > 0 for row in field.diagnostics["coefficients"]
    )

def test_nonlinear_residual_against_analytic_coefficients_and_autograd():
    model = burgers_model()
    model.projection.non_linearity = torch.nn.Tanh()
    with torch.no_grad():
        w = model.fno_blocks.convs[0].weight.tensor
        ids = torch.arange(w.numel(), dtype=torch.float64).reshape(w.shape)
        w.copy_(0.02 * torch.complex(ids.cos(), ids.sin()))
    f = DerivativeField.from_profile(
        model, describe(lambda x: torch.sin(2 * math.pi * x))
    )
    f.certify_coefficients(order=16, progress=None)
    # The lifting is .3*sin(2pi*x)+.2*P*s; its Fourier coefficients are analytic.
    weight, (kx, ks) = mode_weights(model.fno_blocks.convs[0], torch.float64)
    moment = torch.zeros_like(weight[0, 0])
    for i, k in enumerate(kx.tolist()):
        for j, l in enumerate(ks.tolist()):
            if l == 0 and abs(k) == 1:
                moment[i, j] = -0.15j * k
            if k == 0:
                moment[i, j] += (
                    0.1 * f.time_period
                    if l == 0
                    else 0.2 * f.time_period * 1j / (2 * math.pi * l)
                )
    node = f.coefficients["blocks.0.spectral"]
    exact_coefficient = weight[0, 0] * moment
    assert (
        (node.data["coefficients"][0] - exact_coefficient).abs()
        <= node.data["coefficient_error"][0] + 1e-12
    ).all()
    torch.manual_seed(2)
    c = torch.rand(43, 2, dtype=torch.float64, requires_grad=True)
    phase = (
        2
        * math.pi
        * (
            c[:, 0, None, None] * kx[None, :, None]
            + c[:, 1, None, None] * ks[None, None, :]
        )
    )
    spectral = (
        exact_coefficient
        * torch.where(ks == 0, 1.0, 2.0)[None, :]
        * torch.exp(1j * phase)
    ).real.sum((-1, -2))
    value = (
        spectral
        + 0.3 * torch.sin(2 * math.pi * c[:, 0])
        + 0.2 * f.time_period * c[:, 1]
    ).tanh()
    grad = torch.autograd.grad(value.sum(), c, create_graph=True)[0]
    xx = torch.autograd.grad(grad[:, 0].sum(), c)[0][:, 0]
    truth = grad[:, 1] / f.time_period + value * grad[:, 0] - 0.01 * xx
    actual = (
        residual_form(f, c.detach(), torch.zeros_like(c), Residual("burgers", 0.01))
        .lane(1)
        .interval()
    )
    assert (actual.lower[:, 0] <= truth + 1e-10).all()
    assert (actual.upper[:, 0] >= truth - 1e-10).all()

@pytest.mark.parametrize("spatial_axis", [0, 1])
def test_spatial_integration_preserves_time_and_encloses_remainders(spatial_axis):
    h = torch.tensor([[0.2, 0.3]], dtype=torch.float64)
    form = DualForm.constant(torch.tensor([[0.4]], dtype=torch.float64), h)
    form.g[0, 0] = form.c.new_tensor([1.2, -0.8])
    form.r[..., 1] = 0.01
    if spatial_axis == 1:
        form.h = form.h.flip(-1)
        form.g = form.g.flip(-1)
    c, b, a, radius = form.spatial_square_polynomial(spatial_axis)[0]
    assert float(c) == pytest.approx(0.4**2 + 1.2**2 * 0.2**2 / 3)
    assert float(b) == pytest.approx(2 * 0.4 * -0.8)
    assert float(a) == pytest.approx(0.8**2)
    for t in torch.linspace(-0.3, 0.3, 101, dtype=torch.float64):
        for error in (-0.01, 0.0, 0.01):
            actual = (0.4 - 0.8 * t + error) ** 2 + 1.2**2 * 0.2**2 / 3
            expected = c + b * t + a * t * t
            assert expected - radius - 1e-12 <= actual <= expected + radius + 1e-12

def test_manufactured_direct_norms_and_complete_cover():
    model = burgers_identity()
    field = DerivativeField.from_profile(
        model, describe(lambda x: 0 * x), time_period=2.3
    )
    field.certify_coefficients(order=16, progress=None)
    cert, _ = burgers_norm(field, 32, 8, modes=2, time_horizon=0.9, progress=None)
    data = cert.burgers
    assert data.initial_l2 == (0, 0)
    # v=t, r=1: sup in time of the spatial norm is exactly one, not sqrt(dt).
    for pair in data.residual_l2:
        assert pair == pytest.approx((1, 1))
    assert solution_bound(cert).solution_error_upper == pytest.approx(0.9)
    assert cert.coverage["residual"]["leaf_count"] == 256
    assert (
        cert.coverage["residual"]["complete"] and cert.coverage["residual"]["disjoint"]
    )

def test_2d_poisson_cells_enclose_autograd_laplacian():
    model = averaging_model(2)
    model.projection.non_linearity = torch.nn.Tanh()
    with torch.no_grad():
        model.fno_blocks.fno_skips[0].conv.weight.fill_(0.25)

    def forcing(x, y):
        return 1 + torch.sin(2 * math.pi * x) + 0.5 * torch.cos(2 * math.pi * y)

    field = certify(
        model,
        FourierGrid((32, 32), (1.0, 1.0)),
        forcing,
        derivatives=2,
        moment_order=16,
        method=Method("ibp", "gaussian", strategy="taylor_cell"),
    )
    centers = torch.tensor(
        [[0.13, 0.21], [0.47, 0.62], [0.81, 0.83]], dtype=torch.float64
    )
    h = torch.full_like(centers, 0.003)
    bounded = (
        residual_form(field.enclosure, centers, h, Residual("poisson"))
        .lane(1)
        .interval()
    )
    for shift in (-1.0, 0.0, 1.0):
        p = (centers + shift * h).requires_grad_()
        f = forcing(*p.T)
        u = (1 + 0.25 * f).tanh()
        grad = torch.autograd.grad(u.sum(), p, create_graph=True)[0]
        laplacian = sum(
            torch.autograd.grad(grad[:, axis].sum(), p, retain_graph=True)[0][:, axis]
            for axis in range(2)
        )
        truth = -laplacian - f
        assert (bounded.lower[:, 0] <= truth + 1e-10).all()
        assert (truth <= bounded.upper[:, 0] + 1e-10).all()
    norms = residual_bound(field, Residual("poisson"), chunk_size=256).bounds
    assert torch.isfinite(torch.tensor(norms.residual_linf))
    assert norms.residual_linf >= float(truth.detach().abs().max())

def test_gaussian_coefficients_keep_inherited_layer_error(monkeypatch):
    import lean_no.integration.coefficients as gaussian
    from tests.helpers import fno

    model = fno(layers=2, channels=1)
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.zero_()
            if ("lifting" in name or "projection" in name) and "weight" in name:
                p.fill_(1)
        for block in range(2):
            model.fno_blocks.convs[block].weight.tensor[0, 0, 0] = 1
            model.fno_blocks.fno_skips[block].conv.weight.fill_(0.25)
    original = gaussian.integrate
    calls = 0

    def biased_moment(*args, **kwargs):
        nonlocal calls
        center, radius = original(*args, **kwargs)
        calls += 1
        if calls == 1:
            center = center.clone()
            center[:, 0] += 0.001
            radius = radius + 0.001
        return center, radius

    monkeypatch.setattr(gaussian, "integrate", biased_moment)
    field = certify(
        model,
        FourierGrid((16,)),
        lambda x: 0.3,
        derivatives=2,
        moment_order=16,
        method=Method("ibp", "gaussian"),
    )
    lo, hi = field.query(torch.tensor([[0.17], [0.81]], dtype=torch.float64))
    exact = 0.3 * 1.25**2
    assert (lo <= exact + 1e-12).all() and (exact <= hi + 1e-12).all()
    assert (hi - lo).min() > 0.002

def test_cell_l2_norm_uses_physical_volume_not_peak():
    from lean_no.quantities.cells import residual_l2

    lower = torch.tensor([-3.0, 0.0, -1.0, 0.0], dtype=torch.float64)
    upper = torch.tensor([-2.0, 0.0, 2.0, 0.0], dtype=torch.float64)
    assert residual_l2(lower, upper, 2.3) == pytest.approx(math.sqrt(2.3 * (9 + 4) / 4))
