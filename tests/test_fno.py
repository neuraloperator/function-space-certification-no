import itertools
import math

import pytest
import torch
from neuralop.models import FNO

from lean_no.adapters.fno import build_graph
from lean_no.certify import certify
from lean_no.domain import DiscreteGrid, FourierGrid
from lean_no.enclosures.bilinear import BilinearBand, BilinearBounds
from lean_no.function_space import InputBounds, coord
from lean_no.inputs import InputSet
from lean_no.methods import Method
from tests.helpers import averaging_model, fno

def constant_output(net, value):
    # NeuralOperator's native spectral forward allocates complex64 even for a
    # double fno. Evaluate its constant/DC action without that roundoff.
    from lean_no.bounds.operators import affine_bias, affine_weight
    from lean_no.graph import activation_value
    from lean_no.kernel import mode_weights

    graph = build_graph(net)

    def mlp(group, value):
        for index, layer in enumerate(group.layers):
            value = affine_weight(layer) @ value
            bias = affine_bias(layer, value)
            if bias is not None:
                value = value + bias
            if index + 1 < len(group.layers):
                value = activation_value(group.activation, value)
        return value

    value = mlp(graph.lifting, value)
    for block in graph.blocks:
        weight, modes = mode_weights(block.conv, torch.float64)
        zero = tuple(int((axis == 0).nonzero()[0]) for axis in modes)
        next_value = weight[(slice(None), slice(None), *zero)].real @ value
        if block.conv.bias is not None:
            next_value = next_value + block.conv.bias.flatten()
        if block.skip is not None:
            next_value = (
                next_value
                + affine_weight(block.skip) @ value
                + affine_bias(block.skip, value)
            )
        value = next_value
        if block.activation is not None:
            for _ in range(block.activation_repetitions):
                value = activation_value(block.activation, value)
    return mlp(graph.projection, value)

@pytest.mark.parametrize(
    "dimension,layers,activation,integration",
    [
        (1, 4, torch.nn.Tanh(), "gaussian"),
        (1, 2, torch.nn.ReLU(), "arb"),
        (2, 3, torch.nn.GELU(), None),
        (3, 1, torch.nn.Identity(), None),
    ],
)
def test_general_model_contains_constant_witnesses(
    dimension, layers, activation, integration
):
    torch.manual_seed(14)
    net = fno(dimension, layers, activation, inputs=2, outputs=2)
    shape = (4, 6, 4)[:dimension]
    values = torch.empty(2, *shape, dtype=torch.float64)
    values[0], values[1] = 0.3, -0.2
    corners = values[None].expand(2**dimension, *values.shape)
    eps = torch.tensor([0.01, 0.02], dtype=torch.float64)
    result = certify(
        net,
        FourierGrid(shape, (1.5,) * dimension),
        BilinearBand(BilinearBounds(corners, corners), eps),
        method=Method("ibp", integration, strategy="linearized"),
    )
    assert result.diagnostics["implementation"] == "graph"
    assert result.pointwise_width.shape == (2,)
    for a, b in itertools.product((-1, 0, 1), repeat=2):
        perturbed = values.clone()
        perturbed[0] += a * eps[0]
        perturbed[1] += b * eps[1]
        exact = constant_output(net, perturbed[(slice(None), *((0,) * dimension))])
        actual = exact.reshape(-1, *((1,) * dimension)).expand_as(perturbed)
        torch.testing.assert_close(
            actual, net(perturbed[None])[0], atol=2e-7, rtol=2e-7
        )
        assert (result.enclosure.lower[0] <= actual + 1e-12).all()
        assert (actual <= result.enclosure.upper[0] + 1e-12).all()
    points = torch.rand(10, dimension, dtype=torch.float64) * 1.5
    lower, upper = result.query(points)
    assert lower.shape == upper.shape == (10, 2)

@pytest.mark.parametrize("propagation", ["ibp", "crown"])
def test_discrete_linear_bounds_equal_exact_signed_matrix(propagation):
    from lean_no.enclosures.grid import GridBounds

    n = 8
    model = averaging_model()
    with torch.no_grad():
        model.fno_blocks.convs[0].weight.tensor[0, 0, 0] = 0.7
        model.fno_blocks.convs[0].weight.tensor[0, 0, 1] = 0.4 + 0.3j
        model.fno_blocks.fno_skips[0].conv.weight.zero_()
    center = torch.linspace(-0.4, 0.7, n, dtype=torch.float64)
    radius = torch.linspace(0.01, 0.07, n, dtype=torch.float64)
    lower, upper = center - radius, center + radius
    phase = (
        2 * math.pi * (torch.arange(n)[:, None] - torch.arange(n)[None, :]).double() / n
    )
    matrix = (0.7 + 0.8 * phase.cos() - 0.6 * phase.sin()) / n
    exact_lo = matrix.clamp_min(0) @ lower + matrix.clamp_max(0) @ upper
    exact_hi = matrix.clamp_min(0) @ upper + matrix.clamp_max(0) @ lower
    certificate = certify(
        model,
        DiscreteGrid((n,)),
        GridBounds(lower[None, None], upper[None, None]),
        chunk_size=3,
        method=Method(propagation, None),
    )
    lo, hi = certificate.query(torch.arange(n)[:, None])
    torch.testing.assert_close(lo[:, 0], exact_lo)
    torch.testing.assert_close(hi[:, 0], exact_hi)

@pytest.mark.parametrize(
    "dimension,propagation,activation,integration",
    [
        (1, "ibp", torch.nn.ReLU(), "gaussian"),
        (1, "ibp", torch.nn.ReLU(), "arb"),
        (1, "crown", torch.nn.Tanh(), "gaussian"),
        (1, "crown", torch.nn.ReLU(), "arb"),
        (2, "ibp", torch.nn.GELU(), "gaussian"),
        (2, "crown", torch.nn.ReLU(), "gaussian"),
    ],
)
def test_continuum_bounds_contain_admissible_constant_inputs(
    dimension, propagation, activation, integration
):
    model = fno(dimension, layers=2, activation=activation)
    center, eps = 0.3, 0.02
    result = certify(
        model,
        FourierGrid((16,) * dimension, (1.0,) * dimension),
        InputBounds(
            lower=lambda x: (x if dimension == 1 else x[0]).new_full(
                (1, *(x if dimension == 1 else x[0]).shape), center - eps
            ),
            upper=lambda x: (x if dimension == 1 else x[0]).new_full(
                (1, *(x if dimension == 1 else x[0]).shape), center + eps
            ),
            supremum=torch.tensor([center + eps]),
            lipschitz=torch.zeros(1) if dimension == 1 else torch.zeros(1, dimension),
        ),
        method=Method(propagation, integration),
    )
    points = torch.stack(
        [torch.zeros(dimension), torch.full((dimension,), 0.37), torch.ones(dimension)]
    ).double()
    lo, hi = result.query(points)
    for delta in (-eps, 0.0, eps):
        value = constant_output(
            model, torch.tensor([center + delta], dtype=torch.float64)
        )
        assert (lo <= value + 1e-10).all()
        assert (value <= hi + 1e-10).all()
    assert torch.isfinite(result.cell_range_width).all()

@pytest.mark.parametrize("integration", ["taylor_quadrature", "arb"])
def test_optional_backends_enclose_a_known_smooth_operator(integration):
    model = averaging_model()
    with torch.no_grad():
        model.fno_blocks.fno_skips[0].conv.weight.fill_(0.25)
    model.projection.non_linearity = torch.nn.Tanh()
    field = certify(
        model,
        FourierGrid((128,)),
        lambda x: 1 + torch.sin(2 * math.pi * x),
        **{"derivatives": 2} if integration == "arb" else {},
        method=Method("ibp", integration),
    )
    points = torch.linspace(0, 1, 65, dtype=torch.float64)[:, None]
    actual = (1.25 + 0.25 * torch.sin(2 * math.pi * points)).tanh()
    lo, hi = field.query(points)
    assert (lo <= actual + 1e-11).all() and (actual <= hi + 1e-11).all()
    assert (hi - lo).max() < 0.2

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_interval_derivatives_match_analytic_values():
    model = averaging_model().cuda()
    with torch.no_grad():
        model.fno_blocks.fno_skips[0].conv.weight.fill_(0.25)
    field = certify(
        model,
        FourierGrid((64,)),
        lambda x: 1 + torch.sin(2 * math.pi * x),
        moment_order=16,
        derivatives=2,
        method=Method("ibp", "gaussian"),
    )
    points = torch.tensor([[0.0], [0.23], [1.0]], dtype=torch.float64, device="cuda")
    lo, hi = field.query(points)
    expected = 1.25 + 0.25 * torch.sin(2 * math.pi * points)
    assert (lo <= expected + 1e-11).all() and (expected <= hi + 1e-11).all()

def pointwise_model():
    model = (
        FNO(
            n_modes=(2,),
            in_channels=1,
            out_channels=2,
            hidden_channels=1,
            n_layers=1,
            lifting_channel_ratio=1,
            projection_channel_ratio=1,
            non_linearity=torch.nn.Identity(),
            use_channel_mlp=False,
            positional_embedding=None,
            fno_skip="linear",
            factorization=None,
        )
        .double()
        .eval()
    )
    with torch.no_grad():
        for name, value in model.named_parameters():
            value.zero_()
            if "weight" in name and any(
                s in name for s in ("lifting", "projection", "skip")
            ):
                value.fill_(1)
        model.projection.fcs[-1].weight[1].fill_(-2)
    return model

@pytest.mark.parametrize("cells", [7])
def test_cover_contains_nonperiodic_values_and_both_domain_endpoints(cells):
    model = pointwise_model()
    length, eps = 2.7, 0.01
    x = coord()
    forcing = InputBounds(
        (x + x**2 - eps,),
        (x + x**2 + eps,),
        (length + length**2 + eps,),
        (1 + 2 * length,),
    )
    result = certify(
        model,
        FourierGrid((64,), (length,)),
        forcing,
        chunk_size=4,
        output_cells=cells,
        method=Method("crown", "rectangle"),
    )
    points = torch.linspace(0, length, 401, dtype=torch.float64)[:, None]
    lower, upper = result.query(points)
    for delta in [-eps, 0.0, eps]:
        value = points + points.square() + delta
        expected = torch.cat((value, -2 * value), 1)
        assert torch.all(lower <= expected + 1e-12)
        assert torch.all(expected <= upper + 1e-12)
    cover = result.enclosure
    assert int(cover.starts[0]) == 0 and int(cover.stops[-1]) == 64
    assert torch.equal(cover.starts[1:], cover.stops[:-1])
    assert torch.all(cover.stops > cover.starts)
    assert len(cover.starts) <= cells
    with pytest.raises(ValueError, match="finite"):
        result.query(torch.tensor([[torch.nan]], dtype=torch.float64))
    with pytest.raises(ValueError, match="domain"):
        result.query(torch.tensor([[length + 0.01]], dtype=torch.float64))

def test_normalized_discrete_crown_contains_native_network_outputs():
    from lean_no.enclosures.grid import GridBounds

    model = fno(layers=2, activation=torch.nn.ReLU(), norm="group_norm")
    center = torch.randn(1, 1, 16, dtype=torch.float64)
    radius = 0.01
    result = certify(
        model,
        DiscreteGrid((16,)),
        GridBounds(center - radius, center + radius),
        chunk_size=4,
        method=Method("crown", None),
    )
    lo, hi = result.query(torch.arange(16)[:, None])
    for delta in (
        -torch.ones_like(center),
        torch.ones_like(center),
        torch.rand_like(center) * 2 - 1,
    ):
        exact = model(center + radius * delta)[0].T
        assert (lo <= exact + 1e-10).all() and (exact <= hi + 1e-10).all()

@pytest.mark.parametrize("propagation", ["ibp", "crown"])
def test_nonperiodic_2d_convolution_has_the_correct_mean(propagation):
    model = averaging_model(2)
    model.projection.non_linearity = torch.nn.Tanh()
    with torch.no_grad():
        model.fno_blocks.fno_skips[0].conv.weight.fill_(0.25)

    def base(x):
        return (torch.sin(math.pi * x[0]) * torch.sin(math.pi * x[1]))[None]

    source = InputBounds(
        lower=base,
        upper=base,
        supremum=[1.0],
        lipschitz=torch.tensor([[math.pi, math.pi]]),
    )
    result = certify(
        model,
        FourierGrid((32, 32), (1.0, 1.0)),
        source,
        method=Method(propagation, "gaussian"),
    )
    points = torch.tensor([[0.0, 0.0], [0.23, 0.61], [1.0, 1.0]], dtype=torch.float64)
    exact = (
        4 / math.pi**2
        + 0.25 * torch.sin(math.pi * points[:, 0]) * torch.sin(math.pi * points[:, 1])
    ).tanh()[:, None]
    lo, hi = result.query(points)
    assert (lo <= exact + 1e-11).all() and (exact <= hi + 1e-11).all()

@pytest.mark.parametrize(
    "activation,shared",
    [("relu", False), ("relu", True), ("tanh", False), ("tanh", True)],
)
def test_perturbation_encloses_analytic_spectral_operator(activation, shared):
    from lean_no.graph import Graph, Operator

    channels = 2 if shared else 1
    ref = torch.zeros((), dtype=torch.float64)
    spectral = torch.zeros(1, channels, 1, 1, dtype=torch.complex128)
    spectral[0, 0, 0, 0] = 0.7
    model = Graph(
        (
            Operator(
                "lift",
                "affine",
                (0,),
                dict(
                    weight=torch.eye(channels, dtype=ref.dtype),
                    bias=ref.new_zeros(channels),
                ),
            ),
            Operator(
                "mean",
                "spectral",
                (1,),
                dict(
                    weight=spectral,
                    modes=(ref.new_tensor([0.0]), ref.new_tensor([0.0])),
                    bias=ref.new_zeros(1),
                ),
            ),
            Operator(
                "local",
                "affine",
                (1,),
                dict(
                    weight=ref.new_tensor([[-0.4, 0.2] if shared else [-0.4]]),
                    bias=ref.new_zeros(1),
                ),
            ),
            Operator("sum", "add", (2, 3)),
            Operator("nonlinear", "activation", (4,), dict(activation=activation)),
        ),
        dict(dimension=2, input_channels=channels, output_channels=1),
    )

    def profile(x):
        return torch.sin(2 * math.pi * x) + 0.5 * torch.cos(4 * math.pi * x)

    def forcing(x, y):
        return torch.sin(2 * math.pi * x) + 0.5 * torch.cos(4 * math.pi * y)

    period = 17 / 16 if shared else 1.0
    grid = FourierGrid((16, 16), (1.0, period))
    points = torch.cartesian_prod(
        torch.linspace(0, 1, 25, dtype=ref.dtype),
        torch.linspace(0, period, 27, dtype=ref.dtype),
    )
    x, y = points.T
    nominal = profile(x) if shared else forcing(x, y)
    activation_fn = torch.relu if activation == "relu" else torch.tanh
    for epsilon in (0.0, 0.01, 0.1):
        certificate = certify(
            model,
            grid,
            InputSet(profile if shared else forcing, epsilon=epsilon),
            method=Method("ibp", None, strategy="linearized"),
        )
        lower, upper = certificate.query(points)
        # Constants and a nonconstant admissible perturbation have exact means.
        for sign in (-1, 1):
            for varying in (False, True):
                oscillation = torch.cos(2 * math.pi * x)
                if not shared:
                    oscillation = oscillation * torch.cos(2 * math.pi * y)
                delta = (
                    sign * epsilon * (0.5 + 0.5 * oscillation)
                    if varying
                    else sign * epsilon
                )
                mean = sign * epsilon * (0.5 if varying else 1.0)
                exact = activation_fn(
                    0.7 * mean - 0.4 * (nominal + delta) + (0.2 * y if shared else 0)
                )
                assert (lower[:, 0] <= exact + 1e-12).all()
                assert (upper[:, 0] >= exact - 1e-12).all()
        assert (
            torch.isfinite(certificate.pointwise_width).all()
            and (certificate.pointwise_width >= 0).all()
        )

def test_shared_grid_concretization_matches_vertex_extrema():
    from lean_no.function_space.grid.crown import AffineBound, CrownBounds, concretize

    torch.manual_seed(71)
    center = torch.randn(1, 2, 3, 4, dtype=torch.float64)
    center[:, 0] = center[:, 0, :, :1].expand(-1, -1, 4).clone()
    radius = torch.zeros_like(center)
    radius[:, 0] = 0.2
    coefficient = torch.randn(1, 5, 2, 3, 4, dtype=torch.float64)
    bias = torch.randn(1, 5, dtype=torch.float64)
    affine = AffineBound(coefficient, bias)
    low, high = concretize(
        CrownBounds(affine, affine), center - radius, center + radius, (1,)
    )
    values = []
    for signs in itertools.product((-1, 1), repeat=3):
        sample = center.clone()
        sample[:, 0] += 0.2 * center.new_tensor(signs)[:, None]
        values.append(bias + (coefficient * sample.unsqueeze(1)).flatten(2).sum(-1))
    values = torch.stack(values)
    torch.testing.assert_close(low, values.amin(0))
    torch.testing.assert_close(high, values.amax(0))

def test_tanh_grid_crown_contains_shared_spatial_witnesses():
    from lean_no.adapters.fno import native
    from lean_no.function_space.grid.crown import grid_crown

    torch.manual_seed(72)
    model = fno(2, 2, torch.nn.Tanh(), inputs=2, outputs=1)
    center = torch.randn(1, 2, 4, 4, dtype=torch.float64)
    center[:, 0] = center[:, 0, :, :1].expand(-1, -1, 4).clone()
    radius = torch.zeros_like(center)
    radius[:, 0] = 0.1
    lower, upper = grid_crown(
        model,
        center - radius,
        center + radius,
        output_chunk_size=5,
        shared_input_axes=(1,),
    )
    graph = build_graph(model)
    for signs in itertools.product((-1, 0, 1), repeat=4):
        sample = center.clone()
        sample[:, 0] += 0.1 * center.new_tensor(signs)[:, None]
        value = native(graph, sample)
        assert (lower <= value + 1e-12).all()
        assert (upper >= value - 1e-12).all()

@pytest.mark.parametrize("method", ["ibp", "crown"])
def test_grid_negative_nyquist_bin_matches_native(method):
    from lean_no.adapters.fno import cast_module, native
    from lean_no.function_space.grid.crown import grid_crown
    from lean_no.function_space.grid.ibp import grid_ibp

    torch.manual_seed(73)
    model = FNO(
        n_modes=(4, 4),
        in_channels=2,
        out_channels=1,
        hidden_channels=2,
        n_layers=1,
        positional_embedding=None,
        non_linearity=torch.nn.Identity(),
        use_channel_mlp=False,
        fno_skip="linear",
    )
    model = cast_module(model, torch.float64).eval()
    center = torch.randn(1, 2, 4, 5, dtype=torch.float64)
    function = grid_ibp if method == "ibp" else grid_crown
    low, high = function(model, center, center)
    expected = native(build_graph(model), center)
    torch.testing.assert_close(low, expected, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(high, expected, rtol=1e-10, atol=1e-10)

def test_response_row_releases_tensor_without_cyclic_gc():
    import gc
    import weakref

    from lean_no.propagation.perturbation import Response

    enabled = gc.isenabled()
    gc.disable()
    try:
        response = Response("basis", 2, data=0)
        row = response.row(0, 2, (4, 4), torch.zeros((), dtype=torch.float64))
        reference = weakref.ref(row)
        del row
        assert reference() is None
    finally:
        if enabled:
            gc.enable()

def test_linearized_bounds_enclose_pointwise_network():
    from lean_no.adapters.fno import native
    from lean_no.propagation.perturbation import perturbation_bounds

    torch.manual_seed(123)
    graph = build_graph(fno(2, 2, torch.nn.Tanh()), torch.float64)
    with torch.no_grad():
        for op in graph.ops:
            if op.kind == "spectral":
                op.weight.zero_()
        corners = torch.randn(4, graph.in_channels, 4, 4, dtype=torch.float64) * 0.1
        nominal = BilinearBounds(corners, corners)
        output, _ = perturbation_bounds(graph, nominal, 0.01)
        for a, b in [(0.0, 0.0), (0.2, 0.7), (0.5, 0.5), (1.0, 1.0)]:
            weights = corners.new_tensor(
                [(1 - a) * (1 - b), (1 - a) * b, a * (1 - b), a * b]
            )
            value = (corners * weights[:, None, None, None]).sum(0)
            lower = (output.lower * weights[:, None, None, None]).sum(0)
            upper = (output.upper * weights[:, None, None, None]).sum(0)
            for sign in [-1, 1]:
                sample = value.clone()
                sample[0] += sign * 0.01
                exact = native(graph, sample[None])[0]
                assert (lower <= exact + 1e-12).all()
                assert (exact <= upper + 1e-12).all()
