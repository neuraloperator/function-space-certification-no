import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from lean_no.adapters.sfno import SFNOInput, SFNOModel
from lean_no.enclosures.spherical_taylor import spherical_taylor_cells
from lean_no.function_space.spherical import sphere_grid
from lean_no.graph import Graph, Operator, make_field
from lean_no.integration.clenshaw_curtis import (
    integration_error,
    spherical_moments,
    strip_alias_logs,
    strip_integration_error,
)
from lean_no.propagation.spherical import (
    evaluate,
    gelu_field,
    harmonic_field,
    linear_field,
    spherical_bounds,
    sum_field,
)
from lean_no.quantities.spherical import regional_maximum
from lean_no.spectral.spherical import (
    harmonic_eval,
    harmonic_sizes,
    sfno_grid,
)

@pytest.mark.parametrize("q", [16])
def test_clenshaw_curtis_positive_and_cosine_exact(q):
    grid = sfno_grid(q, dtype=torch.float64, rule="clenshaw-curtis")
    assert (grid.quadrature > 0).all()
    assert grid.latitude[0] == 0 and grid.latitude[-1] == math.pi
    for frequency in range(q):
        actual = (grid.quadrature * (frequency * grid.latitude).cos()).sum()
        expected = 0.0 if frequency % 2 else 2 / (1 - frequency**2)
        assert abs(float(actual) - expected) < 3e-14

@pytest.mark.parametrize("q", [16])
def test_cosine_remainder_including_infinite_alias_tail(q):
    grid = sfno_grid(q, dtype=torch.float64, gauss_nodes=32, rule="clenshaw-curtis")
    for frequency in (q - 2, q, q + 2, 2 * (q - 1), 4 * (q - 1), 8 * (q - 1) + 2):
        orders = torch.arange(65, dtype=torch.float64)
        sizes = frequency ** orders[:, None]
        source = make_field(
            "input",
            {"input": SimpleNamespace(evaluate=lambda p: (frequency * p[:, :1]).cos())},
            (),
            torch.zeros(1, dtype=torch.float64),
            sizes,
            {"basis": "spherical"},
        )
        actual, _ = spherical_moments(source, grid, 1, 1)
        expected = math.sqrt(math.pi) * 2 / (1 - frequency**2)
        error = abs(float(actual[0, 0, 0].real) - expected)
        radius = integration_error(sizes, grid, 1)[0, 0] / math.sqrt(4 * math.pi)
        assert error <= radius + 1e-12

@pytest.mark.parametrize("coordinate", ["x", "z"])
def test_analytic_maximum(coordinate):
    coefficients = torch.zeros((1, 2, 2), dtype=torch.complex128)
    if coordinate == "x":
        coefficients[0, 1, 1] = -math.sqrt(2 * math.pi / 3)
        expected = 1.0  # interior maximum at (latitude, longitude) = (0, 0)
    else:
        coefficients[0, 1, 0] = math.sqrt(4 * math.pi / 3)
        expected = math.sin(math.radians(17))  # northern boundary
    field = harmonic_field(coefficients, torch.tensor([0.001], dtype=torch.float64))
    result, _ = regional_maximum(field, (-12, 17, -23, 19), cells=32)
    assert result["maximum_lower"] <= expected - 0.001 + 1e-12
    assert result["maximum_upper"] >= expected + 0.001 - 1e-12
    assert result["gap"] <= 0.02

def test_single_cell_covers_whole_region():
    coefficients = torch.zeros((1, 2, 2), dtype=torch.complex128)
    coefficients[0, 1, 1] = -math.sqrt(2 * math.pi / 3)
    field = harmonic_field(coefficients)
    result, _ = regional_maximum(field, (-12, 17, -23, 19), cells=1)
    assert result["maximum_lower"] <= 1 <= result["maximum_upper"]
    assert result["gap"] > 0.1
    assert result["complete_uniform_cover"]
    assert result["evaluated_cells"] == 1

@pytest.mark.parametrize("enclose", [spherical_taylor_cells])
def test_nonlinear_cells_cover_independent_points(enclose):
    generator = torch.Generator().manual_seed(419)
    coefficients = (
        torch.randn((3, 4, 4), generator=generator, dtype=torch.complex128) * 0.2
    )
    source = harmonic_field(coefficients)
    hidden = gelu_field(source)
    weight = torch.randn((3, 3), generator=generator, dtype=torch.float64)
    hidden = gelu_field(
        linear_field(hidden, weight, torch.zeros(3, dtype=torch.float64))
    )
    field = linear_field(
        sum_field(source, hidden),
        torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.float64),
        torch.zeros(1, dtype=torch.float64),
    )
    centers = torch.tensor([[1.0, 4.2], [1.4, 4.0]], dtype=torch.float64)
    widths = torch.tensor([[0.01, 0.015], [0.03, 0.02]], dtype=torch.float64)
    nominal, radius = enclose(field, centers, widths)
    exact = evaluate(field, centers[:, 0], centers[:, 1], False, {})[:, 0]
    torch.testing.assert_close(nominal, exact)
    for center, width, value, error in zip(centers, widths, nominal, radius):
        fractions = (
            torch.rand((100, 2), generator=generator, dtype=torch.float64) * 2 - 1
        )
        points = center + fractions * width
        samples = evaluate(field, points[:, 0], points[:, 1], False, {})[:, 0]
        assert torch.all((samples - value).abs() <= error + 1e-12)

def spectral_model(grid, weights):
    op = Operator(
        "spectral",
        "spectral",
        (0,),
        dict(weight=weights, degrees=weights.shape[-1], orders=weights.shape[-1]),
    )
    return SFNOModel(Graph((op,), {"basis": "spherical"}), grid, grid, ("a",))

def test_lipschitz_spectral():
    grid = sfno_grid(16, dtype=torch.float64, rule="clenshaw-curtis")
    coefficients = torch.zeros(1, 1, 1, dtype=torch.complex128)
    weights = torch.tensor([[[0.0, 1.0]]], dtype=torch.complex128)
    model = spectral_model(grid, weights)
    widths = []
    for budget in (0.0, 0.02, 1.0):
        inputs = SFNOInput(coefficients, ("a",), 0.1, budget)
        result, stages = spherical_bounds(model, grid, inputs)
        widths.append(float(result.width))
        # The degree-one field below is unchanged by this spectral filter.
        points = grid.points
        actual = min(budget, 0.1) * points[:, :1].cos()
        lower, upper = result.query(points)
        assert (lower <= actual).all() and (actual <= upper).all()
    # Uniform propagation intentionally uses only the everywhere error radius.
    assert widths[0] == widths[1] == widths[2] > 0
    assert widths[2] == pytest.approx(2 * math.sqrt(3) * 0.1)
    # Spatially constant perturbations remain admissible even with Lip(a)=0.
    average = spectral_model(grid, torch.ones(1, 1, 1, dtype=torch.complex128))
    result = spherical_bounds(average, grid, SFNOInput(coefficients, ("a",), 0.1, 0.0))[
        0
    ]
    assert float(result.width) == pytest.approx(0.2)

def test_full_budget():
    # a0=0.05*cos(theta), a=-a0: Lip(a)=M but Lip(a-a0)=2M.
    # There is no spare M-M0 budget, yet this input is admissible.
    coefficients = torch.zeros(1, 2, 2, dtype=torch.complex128)
    coefficients[0, 1, 0] = 0.05 * math.sqrt(4 * math.pi / 3)
    inputs = SFNOInput(coefficients, ("a",), 0.1, 0.05)
    inputs.admit(0.1, 0.05)
    grid = sfno_grid(16, dtype=torch.float64, rule="clenshaw-curtis")
    model = spectral_model(grid, torch.tensor([[[0.0, 1.0]]], dtype=torch.complex128))
    result, stages = spherical_bounds(model, grid, inputs)
    points = torch.tensor([[0.0, 0.0], [0.6, 1.0], [math.pi, 2.0]], dtype=torch.float64)
    lower, upper = result.query(points)
    actual = -inputs.evaluate(points)
    assert (lower <= actual).all() and (actual <= upper).all()

@pytest.mark.parametrize("q", [16])
def test_strip_cosine_alias_sum_with_geometric_tail(q):
    radii = (0.001, 0.01, 0.1, 1.0)
    bound = np.exp(strip_alias_logs(q, radii))
    frequencies = np.arange(q, 200 * q, 2, dtype=np.float64)
    folded = frequencies % (2 * (q - 1))
    folded = np.minimum(folded, 2 * (q - 1) - folded)
    observed = np.array(
        [
            np.sum(
                np.abs(2 / (1 - frequencies**2) - 2 / (1 - folded**2))
                * np.exp(-r * frequencies)
            )
            for r in radii
        ]
    )
    assert np.all(observed <= bound * (1 + 1e-13))

def test_real_and_complex_harmonic_normalization():
    from lean_no.spectral.spherical_strip import complex_legendre

    coefficients = torch.zeros(1, 2, 2, dtype=torch.complex128)
    coefficients[0, 0, 0] = 0.3 * math.sqrt(4 * math.pi)
    coefficients[0, 1, 0] = 0.2 * math.sqrt(4 * math.pi / 3)
    coefficients[0, 1, 1] = -0.4 * math.sqrt(2 * math.pi / 3)
    theta = torch.tensor([0.0, 0.7, math.pi], dtype=torch.float64)
    phi = torch.tensor([0.4, 1.2, 5.0], dtype=torch.float64)
    actual = harmonic_eval(coefficients, theta, phi, False)[:, 0]
    exact = 0.3 + 0.2 * theta.cos() + 0.4 * theta.sin() * phi.cos()
    torch.testing.assert_close(actual, exact, atol=1e-14, rtol=1e-13)
    z = torch.tensor([0.2 + 0.03j, 1.5 + 0.1j, 2.9 + 0.01j], dtype=torch.complex128)
    basis = complex_legendre(z, 4, 4)
    torch.testing.assert_close(basis[0, 0], torch.ones_like(z) / math.sqrt(4 * math.pi))
    torch.testing.assert_close(basis[1, 0], z.cos() * math.sqrt(3 / (4 * math.pi)))
    torch.testing.assert_close(basis[1, 1], -z.sin() * math.sqrt(3 / (8 * math.pi)))

def test_sfno_adapter_matches_native_multilayer_network():
    pytest.importorskip("torch_harmonics")
    from lean_no.adapters.sfno import adapt_backbone

    backbone = BackboneFixture().eval()
    adapted = adapt_backbone(backbone, ("a", "b"))
    inputs = fixture_input()
    values = (
        inputs.evaluate(adapted.input_grid.points)
        .T.reshape(1, 2, *adapted.input_grid.shape)
        .float()
    )
    torch.testing.assert_close(
        adapted.native(values), backbone(values), atol=1e-5, rtol=1e-4
    )
    # The uncertainty budget is for the full input, not the deviation's gradient.
    field, _ = spherical_bounds(
        adapted,
        sfno_grid(16, dtype=torch.float64, rule="clenshaw-curtis"),
        inputs,
        chunk_size=8,
    )
    assert torch.isfinite(field.channel_width).all()

class SpectralFixture(nn.Module):
    def __init__(self, sht, isht, channels):
        super().__init__()
        self.forward_transform, self.inverse_transform = sht, isht
        self.operator_type, self.separable, self.num_groups = "dhconv", False, 1
        self.scale_residual = (sht.nlat, sht.nlon, sht.grid) != (
            isht.nlat,
            isht.nlon,
            isht.grid,
        )
        self.weight = nn.Parameter(
            (0.2 * math.sqrt(3 / channels))
            * torch.randn(1, channels, channels, sht.lmax, dtype=torch.complex64)
        )

    def forward(self, x):
        modes = self.forward_transform(x.float())
        residual = self.inverse_transform(modes).to(x) if self.scale_residual else x
        mixed = torch.einsum("bilm,iol->bolm", modes, self.weight[0])
        return self.inverse_transform(mixed).to(x), residual

class FilterFixture(nn.Module):
    def __init__(self, spectral):
        super().__init__()
        self.filter = spectral

    def forward(self, x):
        return self.filter(x)

class BlockFixture(nn.Module):
    def __init__(self, sht, isht, channels):
        super().__init__()
        self.filter = FilterFixture(SpectralFixture(sht, isht, channels))
        self.norm0 = nn.InstanceNorm2d(channels, eps=1e-6, affine=True)
        self.norm1 = nn.InstanceNorm2d(channels, eps=1e-6, affine=True)
        self.act_layer0 = nn.GELU()
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, 2 * channels, 1),
            nn.GELU(),
            nn.Conv2d(2 * channels, channels, 1),
            nn.Identity(),
        )
        self.drop_path = nn.Identity()
        self.outer_skip = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x):
        x, residual = self.filter(x)
        x = self.norm0(x)
        x = self.act_layer0(x)
        x = self.mlp(x)
        x = self.norm1(x)
        return self.drop_path(x) + self.outer_skip(residual)

class BackboneFixture(nn.Module):
    def __init__(self):
        layers, degrees, channels = 2, 4, 2
        super().__init__()
        import torch_harmonics as th

        torch.manual_seed(0)
        self.inp_chans, self.out_chans = 2, 2
        self.embed_dim, self.big_skip = channels, True
        self.encoder = nn.Sequential(
            nn.Conv2d(self.inp_chans, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, self.out_chans, 1, bias=False),
        )
        self.pos_drop = nn.Identity()
        self.residual_transform = nn.Conv2d(
            self.inp_chans, self.out_chans, 1, bias=False
        )
        outer = (12, 24)
        inner = (8, 16)
        orders = 5
        self.trans_down = th.RealSHT(
            *outer, lmax=degrees, mmax=orders, grid="equiangular"
        )
        self.itrans_up = th.InverseRealSHT(
            *outer, lmax=degrees, mmax=orders, grid="equiangular"
        )
        self.trans = th.RealSHT(
            *inner, lmax=degrees, mmax=orders, grid="legendre-gauss"
        )
        self.itrans = th.InverseRealSHT(
            *inner, lmax=degrees, mmax=orders, grid="legendre-gauss"
        )
        self.blocks = nn.ModuleList(
            [
                BlockFixture(
                    self.trans_down if i == 0 else self.trans,
                    self.itrans_up if i == layers - 1 else self.itrans,
                    channels,
                )
                for i in range(layers)
            ]
        )

    def forward(self, x):
        residual = self.residual_transform(x)
        x = self.encoder(x)
        for block in self.blocks:
            x = block(x)
        return self.decoder(x) + residual

def fixture_input():
    values = torch.zeros(2, 3, 3, dtype=torch.complex128)
    values[:, 0, 0] = torch.tensor([0.2, -0.3], dtype=torch.float64) * math.sqrt(
        4 * math.pi
    )
    values[0, 1, 1] = 0.05 + 0.02j
    values[1, 2, 0] = -0.07
    return SFNOInput(values, ("a", "b"), 0.0, harmonic_sizes(values)[1] + 0.004)

@pytest.mark.parametrize("q,seed", [(16, 71)])
@pytest.mark.parametrize("harmonic_skip", [False, True])
def test_strip_bound_encloses_nonlinear_spherical_moment_error(q, seed, harmonic_skip):
    torch.manual_seed(seed)
    # Nonlinear degree-seven inputs reach orders that alias on the coarse grid.
    coefficients = torch.randn(2, 8, 8, dtype=torch.complex128) * 0.2
    source = gelu_field(harmonic_field(coefficients, chunk_size=32))
    source = linear_field(
        source,
        torch.tensor([[1.0, -0.7], [0.3, 0.8]], dtype=torch.float64),
        torch.tensor([0.2, -0.4], dtype=torch.float64),
    )
    source = gelu_field(source)
    from lean_no.spectral.spherical_strip import signed_pointwise_strip

    radii = torch.tensor([0.1], dtype=torch.float64)
    source.data["signed_strip"] = {
        label: (
            radii,
            signed_pointwise_strip(
                source, radii, (16, 32), 8, coordinate, latitude_batch=3
            ),
        )
        for coordinate, label in enumerate(("theta", "phi"))
    }
    if harmonic_skip:
        source = sum_field(source, harmonic_field(coefficients, chunk_size=32))
    grid = sfno_grid(q, dtype=torch.float64, rule="clenshaw-curtis")
    actual, _ = spherical_moments(source, grid, 5, 5)
    reference, _ = spherical_moments(
        source, sphere_grid((192, 384), dtype=torch.float64), 5, 5
    )
    factors = torch.tensor([1.0, 2.0, 2.0, 2.0, 2.0], dtype=torch.float64)
    observed = ((actual - reference).abs().square() * factors).sum(-1).sqrt()
    bound = strip_integration_error(source, grid, 5)
    degree = torch.arange(5, dtype=torch.float64)
    coefficient_bound = bound * ((2 * degree + 1) / (4 * math.pi)).sqrt()
    assert torch.isfinite(bound).all()
    assert (observed <= coefficient_bound + 2e-12).all()
    assert observed.max() > 1e-9

def test_high_degree_complex_harmonics_against_jacobi_polynomials():
    import mpmath as mp

    from lean_no.spectral.spherical_strip import complex_legendre

    theta = torch.tensor([0.3 + 0.01j, 1.4 + 0.04j], dtype=torch.complex128)
    actual = complex_legendre(theta, 240, 240)
    with mp.workdps(100):
        for degree, order in ((239, 0), (239, 120), (239, 239)):
            scale = mp.sqrt(
                (2 * degree + 1)
                / (4 * mp.pi)
                * mp.factorial(degree - order)
                / mp.factorial(degree + order)
            )
            scale *= mp.factorial(degree + order) / (2**order * mp.factorial(degree))
            exact = torch.tensor(
                [
                    complex(
                        scale
                        * (-mp.sin(complex(z))) ** order
                        * mp.jacobi(degree - order, order, order, mp.cos(complex(z)))
                    )
                    for z in theta
                ],
                dtype=torch.complex128,
            )
            torch.testing.assert_close(
                actual[degree, order], exact, atol=1e-13, rtol=2e-10
            )

def test_spherical_harmonic_moments_remain_exact_without_sampling(monkeypatch):
    import lean_no.propagation.spherical as propagation

    coefficients = torch.zeros((2, 3, 3), dtype=torch.complex128)
    coefficients[:, 0, 0] = 1.0
    coefficients[:, 2, 1] = 0.1 + 0.2j
    source = harmonic_field(coefficients)
    grid = sfno_grid(16, dtype=torch.float64, rule="clenshaw-curtis")

    def no_sampling(*args, **kwargs):
        pytest.fail("harmonic moments should use their exact coefficients")

    monkeypatch.setattr(propagation, "evaluate", no_sampling)
    actual, exact = spherical_moments(source, grid, 3, 3)
    assert exact
    torch.testing.assert_close(actual, coefficients)

def test_spherical_affine_split_reuses_normalized_leaf(monkeypatch):
    import lean_no.propagation.spherical as propagation

    coefficients = torch.zeros((9, 3, 3), dtype=torch.complex128)
    coefficients[:, 0, 0] = 1.0
    coefficients[:, 2, 1] = 0.1 + 0.2j
    nonlinear = gelu_field(harmonic_field(coefficients))
    normalized = linear_field(
        nonlinear,
        torch.eye(9, dtype=torch.float64),
        torch.zeros(9, dtype=torch.float64),
    )
    normalized.data["normalization"] = True
    source = sum_field(normalized, harmonic_field(coefficients))
    grid = sfno_grid(16, dtype=torch.float64, rule="clenshaw-curtis")
    cache = {}
    actual, _ = spherical_moments(source, grid, 3, 3, moment_cache=cache)
    assert cache, "normalized nonlinear moments must be cached on CPU"
    reference, _ = spherical_moments(source, grid, 3, 3, split=False)
    torch.testing.assert_close(actual, reference, atol=1e-10, rtol=1e-10)

    def no_sampling(*args, **kwargs):
        pytest.fail("cached normalized leaf must not be sampled again")

    monkeypatch.setattr(propagation, "evaluate", no_sampling)
    repeated, _ = spherical_moments(source, grid, 3, 3, moment_cache=cache)
    torch.testing.assert_close(repeated, actual)
