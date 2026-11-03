import itertools
import math
import weakref

import pytest
import torch
from neuralop.models import FNO

from lean_no.adapters.fno import build_graph, cast_module, native
from lean_no.certify import certify
from lean_no.domain import FourierGrid
from lean_no.evaluation import reference_error
from lean_no.function_space.analytic import describe
from lean_no.graph import affine, fourier, gelu
from lean_no.inputs import InputSet
from lean_no.integration import integrate
from lean_no.integration.gaussian import AnalyticSource, GaussianOptions
from lean_no.methods import Method
from lean_no.propagation.derivative_cells import DerivativeField, derivative_orders
from lean_no.quantities import Residual
from lean_no.quantities.norms import ResidualBounds, ResidualResult
from lean_no.quantities.residual import residual_bound
from lean_no.spectral.fourier import spectral_forms
from tests.helpers import averaging_model, burgers_identity

@pytest.mark.parametrize(
    "dimension,mlp,norm", [(1, False, None), (2, True, "group_norm")]
)
def test_graph_parity(dimension, mlp, norm):
    torch.manual_seed(82)
    model = cast_module(
        FNO(
            n_modes=(2,) * dimension,
            in_channels=2,
            out_channels=2,
            hidden_channels=4,
            n_layers=3,
            non_linearity=torch.nn.GELU(),
            use_channel_mlp=mlp,
            positional_embedding="grid",
            norm=norm,
            factorization=None,
        ).eval(),
        torch.float64,
    )
    graph = build_graph(model, normalization=True)
    values = torch.randn(1, 2, *((8,) * dimension), dtype=torch.float64)
    expected = model(values)
    torch.testing.assert_close(native(graph, values), expected, atol=2e-6, rtol=2e-6)
    assert all(
        not isinstance(value, torch.nn.Module)
        for op in graph.ops
        for value in op.data.values()
    )
    before = native(graph, values)
    with torch.no_grad():
        model.lifting.fcs[0].weight.add_(1)
    torch.testing.assert_close(native(graph, values), before)

@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_fourier_dimension(dimension):
    lengths = (1.7, 2.3, 0.8)[:dimension]
    modes = tuple(
        torch.tensor([0.0, float(axis + 1)], dtype=torch.float64)
        for axis in range(dimension)
    )
    coefficients = torch.zeros(1, *((2,) * dimension), dtype=torch.complex128)
    coefficients[(0, *((1,) * dimension))] = -0.5j
    source = fourier(coefficients, modes, torch.zeros(1, dtype=torch.float64))
    centers = torch.rand(7, dimension, dtype=torch.float64)
    orders = derivative_orders(dimension, 4)
    forms = spectral_forms(source, centers, torch.zeros_like(centers), orders, lengths)
    frequency = torch.tensor(
        [axis + 1 for axis in range(dimension)], dtype=torch.float64
    )
    phase = 2 * math.pi * (centers * frequency).sum(-1)
    for order, form in zip(orders, forms):
        factor = math.prod(
            (2 * math.pi * k / length) ** n
            for k, length, n in zip(frequency, lengths, order)
        )
        expected = factor * torch.sin(phase + sum(order) * math.pi / 2)
        torch.testing.assert_close(form.c[:, 0], expected, rtol=1e-11, atol=1e-9)
        assert torch.count_nonzero(form.r) == 0

@pytest.mark.parametrize("dimension", [1, 3])
def test_moment_dimension(dimension):
    ref = torch.ones(1, dtype=torch.float64)
    modes = (torch.tensor([0.0, 1.0], dtype=torch.float64),) * dimension

    def evaluate(*axes):
        center = ref.expand(*(len(axis) for axis in axes), 1)
        return center, torch.full_like(center, 0.03)

    source = AnalyticSource(evaluate, (ref * math.exp(math.pi),) * dimension, ref)
    center, radius = integrate(
        source,
        modes,
        ((0.0, 1.0),) * dimension,
        "gaussian",
        GaussianOptions(12, (4,) * dimension, (5,) * dimension),
    )
    expected = torch.zeros_like(center)
    expected[(0, *((0,) * dimension))] = 1
    assert ((center - expected).abs() <= radius.reshape(-1, *([1] * dimension))).all()
    assert 0.03 <= float(radius) < 0.030001

def test_method_selection():
    model = averaging_model()
    inputs = InputSet(lambda x: torch.sin(2 * math.pi * x))
    for degree, order in itertools.product((1, 2), (8, 16)):
        result = certify(
            build_graph(model),
            FourierGrid((16,)),
            inputs,
            derivatives=degree,
            method=Method("ibp", "gaussian"),
            moment_order=order,
        )
        assert result.enclosure.arithmetic == "interval"
        assert result.enclosure.forcing.data["coefficients"].ndim == 2
        assert result.input_bounds is inputs
    for old in ("interval_derivatives", "taylor_cells", "cell_taylor", "linearized"):
        with pytest.raises(ValueError, match="propagation"):
            certify(model, FourierGrid((16,)), inputs, method=Method(old, "gaussian"))
    for degree in (0, 1, 2, 3, 4):
        result = certify(model, FourierGrid((16,)), inputs, derivatives=degree)
        assert result.method == Method("ibp", "gaussian")
        assert result.enclosure.arithmetic == "interval"

def test_tiny_query_radius():
    model = averaging_model()
    model.projection.non_linearity = torch.nn.Tanh()
    source = describe(lambda x: 0.0).field((1.0,))
    result = DerivativeField(build_graph(model), source, "interval")
    radius = torch.tensor([1e-180], dtype=torch.float64)
    moment = fourier(
        torch.ones(1, 1, dtype=torch.complex128),
        (torch.zeros(1, dtype=torch.float64),),
        radius,
    )
    moment.data["coefficient_error"] = radius[:, None]
    result.coefficients["blocks.0.spectral"] = moment
    bound = result.query(torch.tensor([[0.3]], dtype=torch.float64))
    assert torch.equal(bound.lower, bound.upper)
    assert 0 < float(bound.width) < 2e-180

def test_exact_burgers(monkeypatch):
    def reject_fft(*args, **kwargs):
        raise AssertionError("exact profiles must not be sampled and transformed")

    monkeypatch.setattr(torch.fft, "rfft", reject_fft)

    def initial(x):
        return torch.sin(2 * math.pi * x) + 0.5 * torch.cos(4 * math.pi * x)

    result = certify(
        burgers_identity(),
        FourierGrid((16, 8), (1.0, 2.3)),
        initial,
        derivatives=2,
        moment_order=8,
        method=Method("ibp", "gaussian", strategy="taylor_cell"),
    )
    field = result.enclosure
    assert field.forcing.data["coefficients"].shape == (1, 4, 1)
    centers = torch.tensor([[0.13, 0.27], [0.45, 0.73]], dtype=torch.float64)
    jet = field.enclose(centers, torch.zeros_like(centers))
    x = centers[:, 0]
    expected = (
        initial(x) + 2.3 * centers[:, 1],
        2 * math.pi * (torch.cos(2 * math.pi * x) - torch.sin(4 * math.pi * x)),
        torch.ones_like(x),
        -((2 * math.pi) ** 2) * torch.sin(2 * math.pi * x)
        - 8 * math.pi**2 * torch.cos(4 * math.pi * x),
    )
    for form, value in zip(jet, expected):
        torch.testing.assert_close(form.c[:, 0], value, atol=1e-10, rtol=1e-11)

def test_residual_contract():
    for strategy in ("general", "taylor_cell"):
        field = certify(
            averaging_model(),
            FourierGrid((16,)),
            lambda x: 1.0,
            derivatives=2,
            moment_order=8,
            method=Method("ibp", "gaussian", strategy=strategy),
        )
        result = residual_bound(field, Residual("poisson"))
        assert isinstance(result, ResidualResult)
        assert isinstance(result.bounds, ResidualBounds)
        assert "residual" not in result.diagnostics
        assert result.bounds.coverage["residual"]["complete"]
        assert result.grid is field.grid
        comparison = reference_error(
            result,
            points=torch.tensor([[0.1], [0.5], [0.9]], dtype=torch.float64),
            reference=torch.ones(3, dtype=torch.float64),
        )
        # Compare the retained output (one), not the Poisson residual (-one).
        torch.testing.assert_close(
            comparison["errors"], torch.zeros(3, 1, dtype=torch.float64)
        )

def test_field_lifetime():
    ref = torch.zeros(1, dtype=torch.float64)
    field = gelu(
        affine(
            fourier(torch.ones(1, 1, 1, dtype=torch.complex128), (ref, ref), ref),
            ref.new_ones(1, 1),
            ref,
        )
    )
    assert all(
        isinstance(parent, int) for op in field.graph.ops for parent in op.parents
    )
    lifetime = weakref.ref(field.run)
    del field
    assert lifetime() is None

def test_burgers_taylor():
    def initial(x):
        return torch.sin(2 * math.pi * x) + 0.5 * torch.cos(4 * math.pi * x)

    grid = FourierGrid((16, 8), (1.0, 2.3))
    point = torch.tensor([[0.13, 0.27], [0.45, 1.7]], dtype=torch.float64)
    expected = initial(point[:, 0]) + point[:, 1]
    for inputs in (initial, describe(initial)):
        result = certify(
            burgers_identity(),
            grid,
            inputs,
            derivatives=2,
            method=Method("ibp", "taylor_quadrature"),
        )
        bound = result.query(point)
        assert (bound.lower[:, 0] <= expected).all()
        assert (expected <= bound.upper[:, 0]).all()
