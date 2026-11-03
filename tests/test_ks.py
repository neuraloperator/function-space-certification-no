import math

import numpy as np
import pytest
import torch
from scipy.special import erf, eval_hermitenorm

from lean_no.bounds.activations import gelu_coefficients, gelu_remainder_log_coefficient
from lean_no.enclosures.cauchy_jet import profile_forms
from lean_no.enclosures.taylor import polydisk, spatial_taylor
from lean_no.graph import affine, fourier, gelu, make_field
from lean_no.propagation.derivative_cells import enclose_jet
from lean_no.propagation.rollout import (
    initial_state_handoff,
    rollout,
    values,
)
from lean_no.quantities.ks import (
    cell_bounds,
    geometry,
    ks_residual,
    orders,
    paper_profile,
    point_witness,
    space_period,
    split,
    time_period,
)
from lean_no.refine import audit_cover
from lean_no.refine import refine as refine_cells
from lean_no.spectral.strip import StripGauss

def example():
    ref = torch.zeros(1, dtype=torch.float64)
    modes = torch.arange(2, dtype=torch.float64)
    return fourier(
        torch.tensor(
            [[[0.3, 0.2 - 0.1j], [0.1 + 0.05j, 0.02 - 0.03j]]], dtype=torch.complex128
        ),
        (modes, modes),
        ref,
        metadata=geometry,
    )

def test_terminal_is_active_left_exact_nonperiodic_and_inherits_error():
    p = paper_profile()
    ref = p.error
    source = make_field(
        "input",
        {"u": p, "tp": 0.1 / (32 / 34), "lx": 6 * math.pi},
        (),
        ref.new_zeros(3),
        ref.new_tensor([1.5, 0.1, 6 * math.pi]),
    )
    source = affine(source, ref.new_tensor([[0.7, 2.0, 0.1]]), ref.new_tensor([-0.3]))
    source = gelu(source)
    source = make_field("pad", {}, (source,), ref.new_tensor([0.0123]), source.size)
    terminal = initial_state_handoff(source)
    t, x = ref.new_tensor([[32 / 34]]), torch.linspace(0, 1, 101, dtype=ref.dtype)[None]
    direct = values(source, t, x, active=True)
    actual = values(terminal, t * 0, x)
    torch.testing.assert_close(actual, direct, atol=1e-15, rtol=1e-15)
    assert values(source, t, x).abs().max() == 0
    assert (actual[0, 0, -1] - actual[0, 0, 0]).abs() > 0.1
    assert terminal.error.item() == 0.0123

def test_real_centered_complex_polydisk_contains_independent_values():
    base = example()
    field = gelu(
        affine(
            base,
            torch.tensor([[1.2], [-0.7]], dtype=torch.float64),
            torch.tensor([-0.4, 0.6], dtype=torch.float64),
        )
    )
    centers = torch.tensor([[0.13, 0.17], [0.4, 0.72]], dtype=torch.float64)
    radii = torch.tensor([[0.001, 0.002], [0.003, 0.004]], dtype=torch.float64)
    v, g, r = polydisk(field, centers, radii, True)
    rng = np.random.default_rng(91)
    c = base.data["coefficients"].numpy()[0]
    for _ in range(30):
        delta = (
            (rng.uniform(-1, 1, (2, 2)) + 1j * rng.uniform(-1, 1, (2, 2)))
            / math.sqrt(2)
            * radii.numpy()
        )
        z = centers.numpy() + delta
        f = np.zeros(2, dtype=complex)
        for i in range(2):
            for j in range(2):
                phase = 2j * math.pi * (i * z[:, 0] + j * z[:, 1])
                f += (
                    (1 if j == 0 else 2)
                    * (c[i, j] * np.exp(phase) + c[i, j].conjugate() * np.exp(-phase))
                    / 2
                )
        pre = f[:, None] * [1.2, -0.7] + [-0.4, 0.6]
        actual = 0.5 * pre * (1 + erf(pre / math.sqrt(2)))
        linear = v.numpy() + (g.numpy() * delta[:, None]).sum(-1)
        assert np.all(np.abs(actual - linear) <= r.numpy() + 2e-15)

def test_adaptive_gauss_matches_analytic_fourier_moments():
    p = example()
    rule = StripGauss(
        order=12, cells=((16, 64),) * 4, batch=16, progress=lambda *args, **kw: None
    )
    p = make_field(
        p.kind,
        dict(p.data),
        p.parents,
        p.error,
        p.sizes,
        {**p.metadata, "modes": ((16, 33),)},
    )
    c, e = rule(p, 0)
    exact = torch.zeros_like(c)
    # Real part introduces conjugate partners on the zero-spatial-frequency line.
    exact[0, 8, 0] = 0.3
    exact[0, 9, 0] = (0.1 + 0.05j) / 2
    exact[0, 7, 0] = (0.1 - 0.05j) / 2
    exact[0, 8, 1] = 0.2 - 0.1j
    exact[0, 9, 1] = 0.02 - 0.03j
    assert ((c - exact).abs() <= e[:, None, None] + 3e-15).all()
    assert e.item() > 0

def sine_field(error=0.0):
    c = torch.zeros(1, 2, 3, dtype=torch.complex128)
    c[0, 1, 2] = -0.5j
    t = torch.arange(2, dtype=torch.float64)
    x = torch.arange(3, dtype=torch.float64)
    ce = torch.full_like(c.real, error)
    result = fourier(
        c, (t, x), (ce * torch.tensor([1, 2, 2])).sum((-1, -2)), metadata=geometry
    )
    result.data["coefficient_error"] = ce
    return result

def test_physical_fourier_jets_and_manufactured_ks_residual():
    source = sine_field()
    t = torch.tensor([0.13, 0.4], dtype=torch.float64)
    x = torch.tensor([0.12, 0.63], dtype=torch.float64)
    nominal, actual = enclose_jet(source, t, x)
    phase = 2 * math.pi * (t[:, None] + 2 * x[None])
    assert time_period == pytest.approx(0.1 * 34 / 32)
    assert space_period == pytest.approx(6 * math.pi)
    wt, wx = 2 * math.pi / (0.1 * 34 / 32), 4 * math.pi / (6 * math.pi)
    for j, (dt, dx) in enumerate(orders):
        exact = wt**dt * wx**dx * torch.sin(phase + (dt + dx) * math.pi / 2)
        torch.testing.assert_close(
            nominal[j].lower[..., 0], exact, atol=2e-11, rtol=2e-12
        )
        torch.testing.assert_close(actual[j].upper, nominal[j].lower)
    expected = (
        wt * phase.cos()
        + phase.sin() * wx * phase.cos()
        - wx**2 * phase.sin()
        + 0.01 * wx**4 * phase.sin()
    )
    assert_contains(ks_residual(actual), expected[..., None])
    from lean_no.quantities.ks import cell_bounds

    indices = torch.cartesian_prod(torch.arange(8), torch.arange(16))
    upper, witness = cell_bounds(
        source, (8, 16), torch.zeros(128, dtype=torch.int64), indices
    )
    assert upper.shape == (128,)
    assert upper.max() >= witness["absolute_lower"]

def test_nonlinear_whole_cells_cover_autograd_at_interior_points():
    source = gelu(
        affine(
            sine_field(),
            torch.tensor([[1.2], [-0.8]], dtype=torch.float64),
            torch.tensor([-0.4, 0.2], dtype=torch.float64),
        )
    )
    source = gelu(
        affine(
            source,
            torch.tensor([[0.7, -0.3]], dtype=torch.float64),
            torch.tensor([0.1], dtype=torch.float64),
        )
    )
    tc, xc, ht, hx = 0.3, 0.45, 0.002, 0.003
    bounds = enclose_jet(
        source,
        torch.tensor([tc], dtype=torch.float64),
        torch.tensor([xc], dtype=torch.float64),
        ht,
        hx,
    )[1]
    for dt, dx in [(-1, -1), (0, 0), (0.2, -0.7), (1, 1)]:
        t = torch.tensor([tc + dt * ht], dtype=torch.float64, requires_grad=True)
        x = torch.tensor([xc + dx * hx], dtype=torch.float64, requires_grad=True)
        value = _evaluate(source, t, x)
        time = torch.autograd.grad(value.sum(), t, create_graph=True)[0] / time_period
        jets = [value, time.reshape(1, 1, 1)]
        for order in range(1, 5):
            value = torch.autograd.grad(value.sum(), x, create_graph=True)[0]
            jets.append(value.reshape(1, 1, 1) / space_period**order)
        for interval, exact in zip(bounds, jets):
            assert_contains(interval, exact, 2e-10)

def lanes(jets):
    return tuple([p.lane(i) for p in jets] for i in (0, 1))

def uncertain_pair():
    p = paper_profile()
    p.data["coefficient_error"] = torch.full_like(p.data["coefficients"].real, 1e-8)
    p.error[:] = 5e-8
    w, b = (
        torch.tensor([[1.2]], dtype=torch.float64),
        torch.tensor([-0.3], dtype=torch.float64),
    )
    hidden = affine(p, w, b)
    hidden.data["scale_error"] = p.error.new_tensor([2e-6])
    hidden.data["_bound_bias_error"] = p.error.new_tensor([1e-6])
    hidden = hidden.with_error(hidden.error + 2e-6 * (p.size + p.error) + 1e-6)
    source = initial_state_handoff(gelu(hidden), retain_derivatives=True)
    actual = paper_profile()
    phase = torch.arange(
        actual.data["coefficients"].numel(), dtype=torch.float64
    ).reshape_as(actual.data["coefficients"])
    actual.data["coefficients"] += 1e-8 * torch.exp(1j * phase)
    actual = gelu(affine(actual, w + 1e-6, b - 5e-7))
    return source, actual

@pytest.mark.parametrize("width", [0.001])
def test_cauchy_profile_jets_contain_independent_uncertain_autograd_derivatives(width):
    source, actual = uncertain_pair()
    x = torch.tensor([0.21, 0.63], dtype=torch.float64)
    h = torch.full_like(x, width)
    nominal, bounded = lanes(profile_forms(source, x, h, degree=16))
    for shift in (-1.0, 0.0, 1.0):
        point = (x + shift * h).requires_grad_()
        v = values(actual, point[:, None] * 0, point[:, None])[:, 0, 0, 0]
        derivatives = [v, torch.zeros_like(v)]
        derivative = v
        for order in range(1, 5):
            derivative = torch.autograd.grad(
                derivative.sum(), point, create_graph=True
            )[0]
            derivatives.append(derivative / space_period**order)
        for form, truth in zip(bounded, derivatives):
            interval = form.interval()
            assert (interval.lower[:, 0] <= truth + 1e-10).all()
            assert (interval.upper[:, 0] >= truth - 1e-10).all()
    assert any(bool((b.r > n.r).any()) for b, n in zip(bounded, nominal))
    assert bounded[1].r.count_nonzero() == 0

def assert_contains(interval, values, tolerance=1e-10):
    assert torch.all(values >= interval.lower - tolerance)
    assert torch.all(values <= interval.upper + tolerance)

def _evaluate(field, time, space):
    return values(field, time[None], space[None])[0]

def test_small_native_fft_parity_and_multiple_handoffs():
    pytest.importorskip("ks_training")
    from ks_training.model import SmallKS, make_input

    torch.manual_seed(91)
    torch.set_num_threads(4)
    model = SmallKS().double().eval()
    t = torch.arange(34, dtype=torch.float64)[None] / 34
    x = torch.arange(128, dtype=torch.float64)[None] / 128

    def integrate(source, layer):
        assert layer == 0
        sampled = values(source, t, x)[0].permute(2, 0, 1)
        c = torch.fft.rfft2(sampled, norm="forward")[
            :, torch.arange(-8, 8).remainder(34), :33
        ]
        return c, source.error * 0

    profile = paper_profile()
    initial = values(profile, t[:, :1] * 0, x)[0, 0, :, 0][None]
    with torch.no_grad():
        expected = model(make_input(initial))
        output, _, trace = rollout(model.state_dict(), profile, integrate, **geometry)
        actual = values(output, t[:, :32], x)[0].permute(2, 0, 1)[None]
    torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-10)
    assert len(trace) == 2
    profile.error[:] = 1e-5
    for _ in range(3):
        output, budget, _ = rollout(model.state_dict(), profile, integrate, **geometry)
        assert output.error > profile.error
        assert budget[1:].count_nonzero() == 0
        profile = initial_state_handoff(output)
        torch.testing.assert_close(
            values(output, t.new_tensor([[32 / 34]]), x, active=True),
            values(profile, t[:, :1] * 0, x),
            rtol=2e-12,
            atol=2e-12,
        )

def _polynomial_value(model, scaled):
    v = torch.zeros_like(model.coefficients[..., 0])
    for coefficient in model.coefficients.flip(-1).unbind(-1):
        v = v * scaled + coefficient
    return v

def test_scaled_hermite_derivatives_match_independent_scipy():
    x = torch.tensor([-3.0, -0.1, 0.0, 0.7, 4.0], dtype=torch.float64)
    actual = gelu_coefficients(x, 32).numpy()
    phi = np.exp(-(x.numpy() ** 2) / 2) / math.sqrt(2 * math.pi)
    for n in range(2, 33):
        expected = (
            (-1) ** n
            * (eval_hermitenorm(n - 2, x.numpy()) - eval_hermitenorm(n, x.numpy()))
            * phi
            / math.factorial(n)
        )
        np.testing.assert_allclose(actual[:, n], expected, rtol=2e-12, atol=2e-18)
        assert np.max(np.abs(expected)) <= math.exp(gelu_remainder_log_coefficient(n))

@pytest.mark.parametrize("degree", [8])
def test_real_polynomial_enclosures_contain_independent_off_center_values(degree):
    p = paper_profile()
    f = gelu(
        affine(
            p,
            torch.tensor([[1.2]], dtype=torch.float64),
            torch.tensor([-0.3], dtype=torch.float64),
        )
    )
    t = torch.zeros(3, dtype=torch.float64)
    x = torch.tensor([0.13, 0.4, 0.72], dtype=torch.float64)
    h = torch.tensor([0.02, 0.03, 0.04], dtype=torch.float64)
    model = spatial_taylor(f, t, x, h, degree)
    for s in torch.linspace(-1, 1, 101, dtype=torch.float64):
        point = x + s * h
        truth = values(f, t[:, None], point[:, None])[:, 0, 0]
        error = (truth - _polynomial_value(model, s)).abs()
        assert (error <= model.remainder + 2e-14).all()
    assert torch.isfinite(model.remainder).all()

def _refine_residual(field, leaves, base, witness, seconds, batch=8):
    def bound(level, indices, parent):
        return cell_bounds(field, base, level, indices, parent)

    return refine_cells(bound, leaves, witness, seconds, batch)

def _refinement_case():
    field = fourier(
        torch.tensor([[[-1j]]], dtype=torch.complex128),
        (
            torch.tensor([1.0], dtype=torch.float64),
            torch.tensor([0.0], dtype=torch.float64),
        ),
        torch.zeros(1, dtype=torch.float64),
        metadata=geometry,
    )
    nt, nx = 4, 2
    ids = torch.cartesian_prod(torch.arange(nt), torch.arange(nx))
    t, x = (ids[:, 0].double() + 0.5) * split / nt, (ids[:, 1].double() + 0.5) / nx
    _, jets = enclose_jet(field, t, x, split / (2 * nt), 1 / (2 * nx), paired=True)
    leaves = [
        {"level": 0, "indices": ids, "upper": ks_residual(jets).magnitude().reshape(-1)}
    ]
    witness = point_witness(
        field,
        torch.zeros(1, dtype=torch.float64),
        torch.tensor([0.5], dtype=torch.float64),
    )
    return field, leaves, witness

def test_refinement_tightens_analytic_residual_and_preserves_partition():
    field, leaves, witness = _refinement_case()
    old_upper = max(float(b["upper"].max()) for b in leaves)
    result, new_witness, _ = _refine_residual(
        field, leaves, [4, 2], witness, seconds=0.2, batch=8
    )
    coverage = audit_cover(result, 4, 2)
    new_upper = max(float(b["upper"].max()) for b in result)
    exact = 2 * math.pi / time_period
    assert new_witness["absolute_lower"] <= exact + 1e-12
    assert exact - 1e-12 <= new_upper < old_upper
    assert coverage["leaf_count"] > 8
    assert (coverage["leaf_count"] - 8) % 3 == 0

def test_terminal_residual_l2_matches_analytic_physical_integral():
    from lean_no.quantities.ks import terminal_residual_l2

    # u=sin(2*pi*(tau+2*xi)); integrate its KS residual over physical x.
    source = sine_field()
    result = terminal_residual_l2(source)
    wt, wx = 2 * math.pi / time_period, 4 * math.pi / space_period
    a = -(wx**2) + 0.01 * wx**4
    exact = math.sqrt(space_period * ((wt**2 + a**2) / 2 + wx**2 / 8))
    assert result["l2_lower"] <= exact <= result["l2_upper"]
    assert result["relative_gap"] <= 0.099
    assert result["local_time"] == pytest.approx(0.1)
    assert result["complete_spatial_cover"]
    assert result["trace"] == "active-left"

def test_terminal_residual_l2_uses_exact_time_endpoint():
    from lean_no.quantities.ks import terminal_residual_l2

    # u=cos(2*pi*t/time_period), constant in space, so r=u_t.
    coefficients = torch.zeros(1, 2, 1, dtype=torch.complex128)
    coefficients[0, 1, 0] = 1.0
    source = fourier(
        coefficients,
        (torch.arange(2, dtype=torch.float64), torch.zeros(1, dtype=torch.float64)),
        torch.zeros(1, dtype=torch.float64),
        metadata=geometry,
    )
    result = terminal_residual_l2(source)
    expected = (
        math.sqrt(space_period)
        * (2 * math.pi / time_period)
        * abs(math.sin(2 * math.pi * split))
    )
    assert result["l2_lower"] == pytest.approx(expected, rel=1e-12)
    assert result["l2_upper"] == pytest.approx(expected, rel=1e-12)

def test_terminal_residual_l2_refines_and_covers_high_frequency_field():
    from lean_no.quantities.ks import terminal_residual_l2

    # One high-frequency mode forces spatial refinement; its integral is analytic.
    source = fourier(
        torch.tensor([[[-0.5j]]], dtype=torch.complex128),
        (
            torch.tensor([1.0], dtype=torch.float64),
            torch.tensor([500.0], dtype=torch.float64),
        ),
        torch.zeros(1, dtype=torch.float64),
        metadata=geometry,
    )
    result = terminal_residual_l2(source)
    wt, wx = 2 * math.pi / time_period, 1000 * math.pi / space_period
    a = -(wx**2) + 0.01 * wx**4
    exact = math.sqrt(space_period * ((wt**2 + a**2) / 2 + wx**2 / 8))
    assert result["l2_lower"] <= exact <= result["l2_upper"]
    assert result["relative_gap"] <= 0.099
    assert result["maximum_depth"] > 0
    assert result["spatial_cells"] > 2048
    assert result["complete_spatial_cover"]
